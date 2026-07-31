#!/usr/bin/env python3
"""Speak a line of text through one of the bake-off's TTS engines.

The command-line half of Claude Code voice mode: hooks pipe a turn's text in
here, this renders it to a WAV and plays it. Under WSL that playback goes to
Windows rather than PulseAudio; see `use_windows_player`.

Kokoro is the default and runs locally, but loading its ONNX model costs ~3.4s,
which is too much to pay per turn. So the first call starts a small warm daemon
(`--serve`) that owns the loaded model and answers over a unix socket; later
calls reuse it and cost ~1-2s. The daemon exits on its own after an idle spell,
and `--shutdown` ends it immediately.

The cloud engines are one-shot HTTPS calls with no warm-up to manage. They need
this script to run under `venv-ui/bin/python`, which has their SDKs installed.
ElevenLabs is deliberately absent: it is the only engine that will not return a
WAV, and its free plan already blocks the library voices worth using.

Playback starts before the whole utterance has been rendered; see `speak`.

    say.py "text to speak"                   # kokoro, am_michael, speed 1.2
    say.py --engine cartesia "text"          # a cloud engine instead
    say.py --no-play --out /tmp/a.wav "text" # render the lot to one file instead
    say.py --stop                            # cut off whatever is playing
    say.py --shutdown                        # ...and drop the warm model
"""
import argparse
import fcntl
import json
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import wave

BAKE = os.environ.get("BAKE_DIR") or os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("VOICE_DIR") or os.path.expanduser("~/.claude/voice")
SOCK = f"{STATE}/kokoro.sock"
LOCK = f"{STATE}/kokoro.lock"
SPEAK_PID = f"{STATE}/speaking.pgid"
STOPPED = f"{STATE}/stopped"
# When this run began, used to tell a barge-in aimed at us from one aimed at the
# utterance before us. See `claim_speaker`.
STARTED = time.time()
DAEMON_LOG = f"{STATE}/kokoro-daemon.log"
DEFAULT_OUT = f"{STATE}/say.wav"

# Chosen by ear 2026-07-31 from seven Kokoro voices; every speech-rate figure in
# this project's notes was measured at this pairing.
DEFAULT_VOICE = "am_michael"
DEFAULT_SPEED = 1.2

# The warm model holds ~400MB, so it gives that back once a session goes quiet.
# A later turn simply pays the load again and restarts it.
IDLE_EXIT_SECONDS = float(os.environ.get("VOICE_IDLE_EXIT", 1800))
MODEL_LOAD_TIMEOUT = 30.0
# Generous: this bounds one chunk's synthesis, and only fires on a wedged worker.
WORKER_REPLY_TIMEOUT = 120.0


# --- spoken-text cleanup -------------------------------------------------

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]+`")
_URL = re.compile(r"https?://\S+")
_PATH_LINE = re.compile(r"\b[\w./-]+\.(py|ts|tsx|js|json|sh|md|yaml|yml):\d+\b")
_MD_PUNCT = re.compile(r"[*_#>|\[\]]+")
_WS = re.compile(r"\s+")


def sanitize(text):
    """Strip everything unlistenable from markdown, leaving speakable prose.

    Fenced code, inline code, URLs, and file:line references all read as noise
    through a TTS engine, so they come out entirely rather than being narrated.
    """
    text = _FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PATH_LINE.sub(" ", text)
    text = _MD_PUNCT.sub("", text)
    text = _WS.sub(" ", text)
    return text.strip()


def pad_silence(path, seconds=0.7):
    """Prepend silence so a cold audio device cannot swallow the first word.

    An output that has been idle loses the opening fraction of a second while it
    wakes, and this happens on both playback routes: WSLg's RDPSink suspends when
    idle, and Windows' own endpoint clips a cold first utterance the same way
    (heard 2026-07-31 on an unpadded clip). Padding costs a little dead air and
    makes what gets lost silence instead of speech.
    """
    with wave.open(path, "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    quiet = b"\x00" * int(params.framerate * seconds) * params.sampwidth * params.nchannels
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(quiet + frames)


# --- warm Kokoro daemon --------------------------------------------------


def _serve():
    """Own a loaded Kokoro model and answer synth requests over a unix socket.

    One request per connection, JSON in and JSON out, mirroring the worker's own
    line protocol. Started on demand by `_kokoro`; never run by hand.
    """
    os.makedirs(STATE, exist_ok=True)
    # Only one daemon may own the socket, and now that voice mode is per session
    # several conversations can race to start one. The loser must exit rather
    # than unlink a live socket out from under the winner. The lock is held for
    # the whole run and released when the process ends.
    lock = open(LOCK, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # another daemon already owns it
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(4)
    srv.settimeout(IDLE_EXIT_SECONDS)

    log = open(DAEMON_LOG, "ab")
    worker = subprocess.Popen(
        [f"{BAKE}/venv-kokoro/bin/python", f"{BAKE}/engines/kokoro_worker.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        text=True, bufsize=1,
    )
    # Bounded like every other read from the worker: a hang here would hold both
    # the lock and the socket, so no session could ever start a working daemon.
    if not select.select([worker.stdout], [], [], MODEL_LOAD_TIMEOUT)[0] \
            or not worker.stdout.readline():
        worker.terminate()
        os.unlink(SOCK)
        raise SystemExit("kokoro worker never finished loading; see " + DAEMON_LOG)

    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                return
            with conn:
                # A caller that connects and then says nothing must not be able to
                # hold the only daemon hostage. Drop it and go back to accepting.
                conn.settimeout(WORKER_REPLY_TIMEOUT)
                try:
                    line = conn.makefile("r").readline()
                except (socket.timeout, OSError):
                    continue
                if not line.strip():
                    continue
                worker.stdin.write(line if line.endswith("\n") else line + "\n")
                worker.stdin.flush()
                # Bounded, because a wedged worker would otherwise block this loop
                # forever and every later turn in every session would hang on it.
                if not select.select([worker.stdout], [], [], WORKER_REPLY_TIMEOUT)[0]:
                    return  # wedged; end so the next turn gets a fresh worker
                reply = worker.stdout.readline()
                if not reply:
                    return  # worker died; the model is gone, so end with it
                try:
                    conn.sendall(reply.encode())
                except OSError:
                    # The caller was killed mid-synthesis, which is what a barge-in
                    # looks like from here. Losing one reply must not cost the
                    # loaded model, or the next turn pays the whole load again.
                    pass
    finally:
        worker.terminate()
        srv.close()
        if os.path.exists(SOCK):
            os.unlink(SOCK)


def _ask_daemon(req):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
        c.settimeout(120)
        c.connect(SOCK)
        c.sendall((json.dumps(req) + "\n").encode())
        try:
            return json.loads(c.makefile("r").readline())
        except (socket.timeout, ValueError) as e:
            # Surfaced as a RuntimeError so it prints as one `say:` line rather
            # than a traceback, the same as every other failure here.
            raise RuntimeError(f"kokoro daemon gave no usable reply: {e}")


def _start_daemon():
    os.makedirs(STATE, exist_ok=True)
    log = open(DAEMON_LOG, "ab")
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                     start_new_session=True)
    deadline = time.time() + MODEL_LOAD_TIMEOUT
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
                c.connect(SOCK)
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"kokoro daemon did not come up in {MODEL_LOAD_TIMEOUT}s; see {DAEMON_LOG}")


def _kokoro(text, out, voice, speed):
    req = {"text": text, "out": out, "voice": voice, "speed": speed}
    try:
        r = _ask_daemon(req)
    except (FileNotFoundError, ConnectionError):
        # ConnectionError rather than just refused: a socket left behind by a
        # daemon that died mid-request answers the connect and then resets.
        _start_daemon()
        try:
            r = _ask_daemon(req)
        except (FileNotFoundError, ConnectionError) as e:
            # It answered the connect probe and then went away, so report it the
            # same way as any other failure rather than as a bare socket error.
            raise RuntimeError(f"kokoro daemon died right after starting: {e}")
    if not r.get("ok"):
        raise RuntimeError(f"kokoro: {r.get('error')}")
    return out


def shutdown_daemon():
    """Kill the warm daemon and remove its socket, freeing the loaded model.

    Signals rather than asks: the daemon spends its life blocked in accept(),
    so there is no request that would make it return.

    The socket is only removed once a daemon was actually signalled. Removing it
    otherwise would strand a live daemon: still running, still holding the lock,
    but with no socket for anyone to reach it through, so no session could speak
    and no replacement could start.
    """
    if not os.path.exists(SOCK):
        return False
    killed = subprocess.run(["pkill", "-f", f"{os.path.abspath(__file__)} --serve"],
                            check=False).returncode == 0
    if killed and os.path.exists(SOCK):
        os.unlink(SOCK)
    return killed


# --- cloud engines -------------------------------------------------------


def _provider_key(env_names, key_filename):
    """Resolve a cloud provider key from env vars (in order), then a key file."""
    for var in env_names:
        key = os.environ.get(var)
        if key:
            return key.strip()
    path = f"{BAKE}/{key_filename}"
    if os.path.exists(path):
        return open(path).read().strip()
    raise RuntimeError(f"no key: set {env_names[0]} or write {path}")


# A hung cloud call is indistinguishable from a broken feature: the turn simply
# never speaks. Bounded so a bad call fails fast enough to notice and retry.
CLOUD_TIMEOUT = 30.0


def _cartesia(text, out, voice, speed):
    from cartesia import Cartesia
    client = Cartesia(api_key=_provider_key(("CARTESIA_API_KEY",), ".cartesia_key"),
                      timeout=CLOUD_TIMEOUT)
    resp = client.tts.generate(
        model_id="sonic-3.5", transcript=text,
        voice={"mode": "id", "id": voice},
        # 16-bit, not the float encoding app.py uses: `pad_silence` reads through
        # Python's `wave`, which rejects IEEE float outright ("unknown format: 3").
        output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100},
    )
    resp.write_to_file(out)
    _fix_wav_header(out)
    return out


def _deepgram(text, out, voice, speed):
    from deepgram import DeepgramClient
    client = DeepgramClient(api_key=_provider_key(("DEEPGRAM_API_KEY",), ".deepgram_key"))
    audio = client.speak.v1.audio.generate(
        text=text, model=voice, encoding="linear16", container="wav", sample_rate=24000,
        # This SDK takes its timeout per request rather than on the client.
        request_options={"timeout_in_seconds": int(CLOUD_TIMEOUT)})
    with open(out, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)
    _fix_wav_header(out)
    return out


def _openai(text, out, voice, speed):
    from openai import OpenAI
    client = OpenAI(api_key=_provider_key(("OPENAI_API_KEY", "OPEN_AI_TTS_KEY"), ".openai_key"),
                    timeout=CLOUD_TIMEOUT)
    with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts-2025-12-15", voice=voice,
            input=text, response_format="wav") as resp:
        resp.stream_to_file(out)
    _fix_wav_header(out)
    return out


def _fix_wav_header(path):
    """Patch a streamed WAV's length fields to the real file size.

    The cloud engines emit the header before the total length is known and stamp
    the RIFF and data sizes with a placeholder. Players that trust it compute a
    bogus duration and stop early, clipping the final words. Same fix as app.py.
    """
    import struct
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        if f.read(4) != b"RIFF" or f.seek(8) and f.read(4) != b"WAVE":
            return
        f.seek(4)
        f.write(struct.pack("<I", size - 8))
        f.seek(12)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            cid, csz = struct.unpack("<4sI", hdr)
            if cid == b"data":
                data_start = f.tell()
                f.seek(data_start - 4)
                f.write(struct.pack("<I", size - data_start))
                return
            f.seek(csz + (csz & 1), 1)


ENGINES = {
    "kokoro": (_kokoro, DEFAULT_VOICE),
    "cartesia": (_cartesia, "6f84f4b8-58a2-430c-8c79-688dad597532"),
    "deepgram": (_deepgram, "aura-2-thalia-en"),
    "openai": (_openai, "marin"),
}


# --- playback ------------------------------------------------------------

POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
_PLAYER_ARGV0 = re.compile(r"^(paplay|\S*powershell\.exe)\b")


def use_windows_player():
    """True when this is WSL and Windows' own audio stack is reachable.

    WSLg exposes a PulseAudio server that forwards to Windows over an RDP
    transport, and that transport stalls badly on sustained streams: measured
    2026-07-31, an 80.4s utterance took 171.6s to play through `paplay`, arriving
    in stutters and long gaps, and left the sound server unresponsive to even
    `pactl info` for half a minute afterwards. Handing the same file to Windows
    played it in 80.8s and left the sound server alone. Short utterances fit
    inside the sink's ~2s buffer, which is why this only ever showed up on long
    ones.
    """
    try:
        wsl = "microsoft" in open("/proc/sys/kernel/osrelease").read().lower()
    except OSError:
        return False
    return wsl and os.path.exists(POWERSHELL)


def _is_speaking_group(pgid):
    """True if pgid still names an utterance of ours rather than a recycled id.

    A run killed outright leaves its pid file behind, and this machine's pid_max
    is only 99999, so that id can come back around attached to something else
    entirely. Signalling it blind would SIGTERM an innocent process group, so
    confirm the group still holds the parts of an utterance: this script, or a
    player orphaned by a run that died without taking it along.
    """
    ps = subprocess.run(["ps", "-eo", "pgid=,args="], capture_output=True, text=True)
    for line in ps.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] == str(pgid):
            if "say.py" in parts[1] or _PLAYER_ARGV0.search(parts[1]):
                return True
    return False


def stop_speaking():
    """Cut off the current utterance, whatever stage it has reached.

    Signals the whole process group rather than just `paplay`, because a turn
    spends its first second or two synthesizing with nothing playing yet. A
    barge-in that only killed the player would let that pending audio start
    speaking a moment later.
    """
    try:
        pgid = int(open(SPEAK_PID).read().strip())
    except (FileNotFoundError, ValueError):
        return False
    killed = False
    if pgid != os.getpgrp() and _is_speaking_group(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            pass
    try:
        os.unlink(SPEAK_PID)
    except FileNotFoundError:
        pass  # a concurrent barge-in cleared the claim first
    return killed


def _stopped_since_start():
    """True if someone asked for silence after this run began.

    A turn's speech is launched detached, so there is a moment between launch and
    claiming the speaker where this run owns nothing and a barge-in has nothing
    to kill. Without this check that barge-in is ignored and the run starts
    talking immediately afterwards, which is the interruption failing outright.
    """
    try:
        return os.path.getmtime(STOPPED) > STARTED
    except OSError:
        return False


def claim_speaker():
    """Become the one run allowed to make noise, ending any run already going.

    Raises SystemExit if a barge-in landed while this run was starting up.
    """
    os.setpgrp()  # the player inherits this group, so one signal reaches both
    stop_speaking()
    os.makedirs(STATE, exist_ok=True)
    with open(SPEAK_PID, "w") as f:
        f.write(str(os.getpgrp()))
    # Claim first, then look. A stop landing between the two orderings would
    # otherwise find no claim to kill and pass unnoticed, leaving this run to
    # speak straight through an interruption that had already been made.
    if _stopped_since_start():
        release_speaker()
        raise SystemExit(0)


def release_speaker():
    try:
        if int(open(SPEAK_PID).read().strip()) == os.getpgrp():
            os.unlink(SPEAK_PID)
    except (FileNotFoundError, ValueError):
        pass


# Kokoro renders roughly 2.6x faster than realtime, so rendering a whole
# utterance before playing any of it cost 13.7s of silence on a 123-word one
# (measured 2026-07-31). Rendering in chunks and playing the first while the rest
# is still being made turns that into the cost of the first chunk alone. The
# first is kept short for exactly that reason; later ones are larger because by
# then the renderer is comfortably ahead of the player.
FIRST_CHUNK_WORDS = 15
CHUNK_WORDS = 50

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, first=FIRST_CHUNK_WORDS, rest=CHUNK_WORDS):
    """Split into sentence-aligned chunks, the first one deliberately small.

    Splitting mid-sentence would be audible, since the engine's prosody runs to
    the end of a sentence, so chunk edges only ever land where a sentence does.
    """
    chunks, current, limit = [], [], first
    for sentence in _SENTENCE_END.split(text):
        if not sentence.strip():
            continue
        current.append(sentence)
        if sum(len(s.split()) for s in current) >= limit:
            chunks.append(" ".join(current))
            current, limit = [], rest
    if current:
        chunks.append(" ".join(current))
    return chunks


def _render(chunks, work, synth, voice, speed, pad):
    """Render each chunk in order, flagging each one done as it lands.

    A chunk is only announced once it is complete on disk, so the player can
    never pick up a half-written file.
    """
    for i, chunk in enumerate(chunks):
        out = os.path.join(work, "chunk-%d.wav" % i)
        try:
            synth(chunk, out, voice, speed)
            if i == 0 and pad > 0:
                pad_silence(out, pad)
        except Exception:
            # Say why. This runs in a thread whose exception would otherwise go
            # nowhere, leaving a turn that is simply silent with no explanation.
            print(f"say: chunk {i} failed to render", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            open(os.path.join(work, "abort"), "w").close()
            return
        open(os.path.join(work, "chunk-%d.ok" % i), "w").close()


def _play_chunks_windows(work, count):
    """Play the chunks through one PowerShell that outlives all of them.

    Starting PowerShell costs ~0.75s, which as a per-chunk cost would be an
    audible gap between every sentence. So it is started once and waits for each
    chunk in turn. The count is known up front, and an `abort` file releases it
    if rendering dies, so it cannot wait forever.
    """
    win = subprocess.run(["wslpath", "-w", work],
                         capture_output=True, text=True).stdout.strip()
    if not win:
        raise RuntimeError(f"wslpath could not convert {work}")
    script = (
        "$d = '%s'\n"
        "for ($i = 0; $i -lt %d; $i++) {\n"
        "  $wav = Join-Path $d ('chunk-' + $i + '.wav')\n"
        "  $ok = Join-Path $d ('chunk-' + $i + '.ok')\n"
        "  while (-not (Test-Path $ok)) {\n"
        "    if (Test-Path (Join-Path $d 'abort')) { exit }\n"
        "    Start-Sleep -Milliseconds 20\n"
        "  }\n"
        "  if (Test-Path $wav) { (New-Object Media.SoundPlayer $wav).PlaySync() }\n"
        "}\n"
    ) % (win.replace("'", "''"), count)
    subprocess.run([POWERSHELL, "-NoProfile", "-Command", script],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _play_chunks_paplay(work, count):
    if not shutil.which("paplay"):
        raise RuntimeError("no paplay on PATH, and this is not WSL, so nothing can play audio")
    for i in range(count):
        ok = os.path.join(work, "chunk-%d.ok" % i)
        while not os.path.exists(ok):
            if os.path.exists(os.path.join(work, "abort")):
                return
            time.sleep(0.02)
        wav = os.path.join(work, "chunk-%d.wav" % i)
        if os.path.exists(wav):
            subprocess.run(["paplay", wav],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def speak(text, synth, voice, speed, pad):
    """Say the text, starting playback before the whole thing has been rendered.

    Rendering runs in a thread so it can stay ahead of the player. Both die with
    the process, which is what a barge-in signals, so neither needs its own
    shutdown path.
    """
    chunks = chunk_text(text)
    if not chunks:
        return
    # Per process, not one shared directory. A barge-in signals the previous run
    # but does not wait for it to die, so for a moment two runs are alive at once
    # and a shared directory would let the new one delete chunks the old one is
    # still playing, or splice the two utterances together.
    work = os.path.join(STATE, "stream-%d" % os.getpid())
    _clear_dead_streams()
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    renderer = threading.Thread(target=_render,
                                args=(chunks, work, synth, voice, speed, pad),
                                daemon=True)
    renderer.start()
    try:
        if use_windows_player():
            _play_chunks_windows(work, len(chunks))
        else:
            _play_chunks_paplay(work, len(chunks))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _clear_dead_streams():
    """Remove stream directories whose run is gone.

    A run killed mid-utterance never reaches its own cleanup, so the tidying has
    to be done by whoever comes next.
    """
    for name in os.listdir(STATE) if os.path.isdir(STATE) else []:
        if not name.startswith("stream-"):
            continue
        try:
            pid = int(name.split("-", 1)[1])
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(os.path.join(STATE, name), ignore_errors=True)
        except OSError:
            pass  # alive but not ours to signal; leave it alone


# --- entry point ---------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Speak text through a bake-off TTS engine.")
    ap.add_argument("text", nargs="*", help="text to speak; omit to read stdin")
    ap.add_argument("--engine", default="kokoro", choices=sorted(ENGINES))
    ap.add_argument("--voice", default=None, help="engine-specific voice or model id")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Kokoro only")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="where to write the WAV; only used with --no-play, since "
                         "playing renders in chunks rather than one file")
    ap.add_argument("--no-play", action="store_true",
                    help="render the whole utterance to --out and stay silent")
    ap.add_argument("--raw", action="store_true", help="skip the markdown sanitizer")
    ap.add_argument("--pad", type=float, default=0.7,
                    help="seconds of leading silence; raise if a first word goes missing")
    ap.add_argument("--stop", action="store_true", help="cut off playback and exit")
    ap.add_argument("--shutdown", action="store_true", help="also drop the warm model")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.serve:
        _serve()
        return 0

    if args.stop or args.shutdown:
        os.makedirs(STATE, exist_ok=True)
        # Recorded even when there was nothing to kill, so a run still starting
        # up can see that silence was asked for and give up before it speaks.
        with open(STOPPED, "w"):
            pass
        stop_speaking()
        if args.shutdown:
            shutdown_daemon()
        return 0

    text = " ".join(args.text) if args.text else sys.stdin.read()
    if not args.raw:
        text = sanitize(text)
    if not text.strip():
        return 0

    # Claimed before synthesis, not before playback: this also cuts off a
    # predecessor still rendering into the same file we are about to overwrite.
    claim_speaker()
    try:
        synth, default_voice = ENGINES[args.engine]
        voice = args.voice or default_voice
        if args.no_play:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            out = synth(text, args.out, voice, args.speed)
            if args.pad > 0:
                pad_silence(out, args.pad)
        else:
            speak(text, synth, voice, args.speed, args.pad)
    finally:
        release_speaker()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"say: {e}", file=sys.stderr)
        sys.exit(1)
