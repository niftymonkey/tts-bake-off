"""PROTOTYPE. Hear four candidate spoken registers back to back.

Question: when Claude Code's Stop hook speaks the end of a turn, what should it
sound like? Four registers, five real turn shapes, rendered through Kokoro so
the choice is made by ear instead of by argument.

Throwaway TUI. The registers and the sanitizer in `registers.py` are the part
that lifts into the real hook.

Run:  just register     (or)  python3 prototype/voice_register.py
Keys: [1-5] turn  [a-d] register  [p] speak  [x] speak all four
      [v] voice   [+/-] speed     [s] source  [q] quit
"""
import json
import os
import subprocess
import sys
import termios
import tty
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import FIXTURES, FIXTURES_BY_KEY          # noqa: E402
from registers import (REGISTERS, REGISTERS_BY_KEY,      # noqa: E402
                       sanitize, spoken_seconds)

BAKE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{BAKE}/out/proto-register.wav"

# A spread across Kokoro's US/British and male/female voices, not the whole 54.
VOICES = ["am_michael", "am_onyx", "am_fenrir", "bm_george",
          "af_heart", "af_bella", "bf_emma"]

BOLD, DIM, RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"
CYAN, GREEN, YELLOW = "\x1b[36m", "\x1b[32m", "\x1b[33m"


class Kokoro:
    """Warm Kokoro worker. Loading the model costs ~3.4s, so pay it once."""

    def __init__(self):
        self.proc = None

    def _start(self):
        self.proc = subprocess.Popen(
            [f"{BAKE}/venv-kokoro/bin/python", f"{BAKE}/engines/kokoro_worker.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        if not self.proc.stdout.readline():
            raise RuntimeError("kokoro worker exited during load")

    def close(self):
        """Take the worker down with the TUI, rather than leaving it warm."""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()
        self.proc = None

    def synth(self, text, voice, speed):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        req = {"text": text, "out": OUT, "voice": voice, "speed": speed}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())


def pad_silence(path, seconds=0.7):
    """Prepend silence so a suspended sink cannot swallow the first word.

    WSLg's RDPSink suspends when idle and loses the opening fraction of a second
    while it resumes. Padding costs a little dead air and makes what gets lost
    silence instead of speech. Lifts straight into the real read-aloud path.
    """
    with wave.open(path, "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    quiet = b"\x00" * int(params.framerate * seconds) * params.sampwidth * params.nchannels
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(quiet + frames)


class State:
    def __init__(self):
        self.fixture = FIXTURES[0]
        self.register = REGISTERS[0]
        self.voice_ix = 0
        self.speed = 1.2
        self.show_source = False
        self.status = "ready"

    @property
    def voice(self):
        return VOICES[self.voice_ix]

    def spoken(self):
        return self.register.spoken(self.fixture)


def wrap(text, width=76, indent=""):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(indent + line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(indent + line)
    return lines


def render(state):
    print("\033[2J\033[H", end="")
    spoken = state.spoken()
    words = len(spoken.split())

    print(f"{BOLD}Spoken register prototype{RESET}  "
          f"{DIM}which voice-mode register should the Stop hook speak?{RESET}\n")

    print(f"{BOLD}Turn{RESET}     " + "  ".join(
        f"{CYAN}{BOLD}[{f.key}]{f.name}{RESET}" if f is state.fixture
        else f"{DIM}[{f.key}]{f.name}{RESET}" for f in FIXTURES))
    print(f"{DIM}         shape: {state.fixture.shape}{RESET}\n")

    print(f"{BOLD}Register{RESET}")
    for r in REGISTERS:
        secs = spoken_seconds(r.spoken(state.fixture))
        mark = f"{CYAN}{BOLD}>{RESET}" if r is state.register else " "
        style = f"{CYAN}{BOLD}" if r is state.register else DIM
        print(f"  {mark} {style}[{r.key}] {r.name:<34}{RESET}"
              f"{DIM}{secs:5.1f}s   budget {r.budget}{RESET}")
    for line in wrap(state.register.idea, 68, "      "):
        print(f"{DIM}{line}{RESET}")
    print()

    screen_words = len(sanitize(state.fixture.markdown).split())
    ratio = f"{screen_words / words:.1f}x shorter" if words else "n/a"
    print(f"{BOLD}Would speak{RESET} {DIM}({words} words, "
          f"{spoken_seconds(spoken):.1f}s spoken; "
          f"{screen_words} words on screen, {ratio}){RESET}")
    if spoken:
        for line in wrap(spoken, 74, "  "):
            print(f"{GREEN}{line}{RESET}")
    else:
        print(f"  {YELLOW}(nothing: this register produces no speech here){RESET}")
    print()

    if state.show_source:
        print(f"{BOLD}On screen{RESET} {DIM}(what Claude actually wrote){RESET}")
        for line in state.fixture.markdown.splitlines():
            print(f"{DIM}  {line}{RESET}")
        print()

    print(f"{DIM}voice {RESET}{state.voice}{DIM}   speed {RESET}{state.speed:.2f}"
          f"{DIM}   status {RESET}{state.status}")
    print(f"{DIM}[1-5] turn  [a-d] register  [p] speak  [x] speak all four  "
          f"[v] voice  [+/-] speed  [s] source  [q] quit{RESET}")


def speak(kokoro, state, text, label=None):
    if not text.strip():
        state.status = "nothing to speak"
        return
    state.status = f"synthesizing{'' if not label else ' ' + label}..."
    render(state)
    r = kokoro.synth(text, state.voice, state.speed)
    if not r.get("ok"):
        state.status = f"kokoro failed: {r.get('error')}"
        return
    pad_silence(OUT)
    state.status = f"speaking{'' if not label else ' ' + label}..."
    render(state)
    subprocess.run(["paplay", OUT], timeout=120)
    state.status = "ready"


def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    state = State()
    kokoro = Kokoro()
    try:
        loop(state, kokoro)
    finally:
        kokoro.close()


def loop(state, kokoro):
    render(state)
    while True:
        k = getkey()
        if k in ("q", "\x03"):
            print("\033[2J\033[H", end="")
            break
        elif k in FIXTURES_BY_KEY:
            state.fixture = FIXTURES_BY_KEY[k]
        elif k.upper() in REGISTERS_BY_KEY:
            state.register = REGISTERS_BY_KEY[k.upper()]
        elif k == "p":
            speak(kokoro, state, state.spoken())
        elif k == "x":
            chosen = state.register
            for r in REGISTERS:
                state.register = r
                speak(kokoro, state, state.spoken(), label=f"[{r.key}] {r.name}")
            state.register = chosen
        elif k == "v":
            state.voice_ix = (state.voice_ix + 1) % len(VOICES)
        elif k in ("+", "="):
            state.speed = min(2.0, state.speed + 0.05)
        elif k in ("-", "_"):
            state.speed = max(0.5, state.speed - 0.05)
        elif k == "s":
            state.show_source = not state.show_source
        render(state)


if __name__ == "__main__":
    main()
