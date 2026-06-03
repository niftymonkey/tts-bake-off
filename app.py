"""TTS Bake-off: a local Gradio playground to compare TTS engines.

Local engines (Kokoro, Chatterbox, XTTS-v2, Dia) each run as a warm worker in
their own venv, loaded once and driven over a stdin/stdout JSON pipe. Cloud
engines (OpenAI) are called in-process. Audio plays in-browser, so nothing
touches the WSLg PulseAudio server.

GPU budget: only one GPU model stays resident at a time (Chatterbox, XTTS, Dia
together would exceed 11GB). Switching GPU engines releases the previous one and
reloads the new, which costs a load. Kokoro is CPU and OpenAI is cloud, so both
are always available with no GPU cost.

Launch:  venv-ui/bin/python app.py   then open the printed URL.
"""
import os
import json
import subprocess
import threading
import itertools

import gradio as gr

BAKE = "/home/mlo/dev/niftymonkey/tts-bake-off"
OUT = f"{BAKE}/out"
LOGS = f"{BAKE}/logs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

BREW_LIB = "/home/linuxbrew/.linuxbrew/lib"

ENGINES = {
    "kokoro": {
        "python": f"{BAKE}/venv-kokoro/bin/python",
        "script": f"{BAKE}/engines/kokoro_worker.py",
        "env": {}, "gpu": False,
    },
    "chatterbox": {
        "python": f"{BAKE}/venv-chatterbox/bin/python",
        "script": f"{BAKE}/engines/chatterbox_worker.py",
        "env": {}, "gpu": True,
    },
    "xtts": {
        "python": f"{BAKE}/venv-xtts/bin/python",
        "script": f"{BAKE}/engines/xtts_worker.py",
        "env": {"LD_LIBRARY_PATH": BREW_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")},
        "gpu": True,
    },
    "dia": {
        "python": f"{BAKE}/venv-dia/bin/python",
        "script": f"{BAKE}/engines/dia_worker.py",
        "env": {}, "gpu": True,
    },
}

_counter = itertools.count(1)
GPU_LOCK = threading.Lock()  # serializes GPU use; enforces one resident GPU model


class Worker:
    """Lazily-started persistent subprocess for one local engine."""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.gpu = cfg.get("gpu", False)
        self.proc = None
        self.lock = threading.Lock()

    def _start(self):
        env = dict(os.environ)
        env.update(self.cfg["env"])
        log = open(f"{LOGS}/{self.name}.log", "ab")
        self.proc = subprocess.Popen(
            [self.cfg["python"], self.cfg["script"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            env=env, text=True, bufsize=1,
        )
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError(f"{self.name}: worker exited during load. See {LOGS}/{self.name}.log")
        try:
            msg = json.loads(ready)
        except json.JSONDecodeError:
            raise RuntimeError(f"{self.name}: bad startup line {ready!r}. See {LOGS}/{self.name}.log")
        if not msg.get("ready"):
            raise RuntimeError(f"{self.name}: not ready: {ready!r}")

    def stop(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None

    def _ensure_and_send(self, req):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{self.name}: worker died mid-request. See {LOGS}/{self.name}.log")
            return json.loads(line)

    def synth(self, req):
        if not self.gpu:
            return self._ensure_and_send(req)
        with GPU_LOCK:
            for other_name, other in WORKERS.items():
                if other_name != self.name and other.gpu:
                    other.stop()
            return self._ensure_and_send(req)


WORKERS = {name: Worker(name, cfg) for name, cfg in ENGINES.items()}


def _out_path(engine):
    return f"{OUT}/ui-{engine}-{next(_counter)}.wav"


def _run(engine, req):
    if not req["text"].strip():
        raise gr.Error("Enter some text first.")
    r = WORKERS[engine].synth(req)
    if not r.get("ok"):
        raise gr.Error(r.get("error", f"{engine} failed"))
    return r["out"]


def gen_kokoro(text, voice, speed):
    return _run("kokoro", {"text": text, "out": _out_path("kokoro"), "voice": voice, "speed": speed})


def gen_chatterbox(text, exaggeration, cfg_weight, ref):
    req = {"text": text, "out": _out_path("chatterbox"),
           "exaggeration": exaggeration, "cfg_weight": cfg_weight}
    if ref:
        req["audio_prompt"] = ref
    return _run("chatterbox", req)


def gen_xtts(text, speaker, language, ref):
    req = {"text": text, "out": _out_path("xtts"), "language": language}
    if ref:
        req["speaker_wav"] = ref
    else:
        req["speaker"] = speaker
    return _run("dia" if False else "xtts", req)


def gen_dia(text):
    return _run("dia", {"text": text, "out": _out_path("dia")})


# --- OpenAI (cloud, in-process) ---

OPENAI_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable",
                 "nova", "onyx", "sage", "shimmer", "verse"]


def _openai_key():
    for var in ("OPENAI_API_KEY", "OPEN_AI_TTS_KEY"):
        key = os.environ.get(var)
        if key:
            return key.strip()
    path = f"{BAKE}/.openai_key"
    if os.path.exists(path):
        return open(path).read().strip()
    return None


def gen_openai(text, voice, instructions):
    if not text.strip():
        raise gr.Error("Enter some text first.")
    key = _openai_key()
    if not key:
        raise gr.Error("No OpenAI key found. Put it in ~/tts-bakeoff/.openai_key (or set OPENAI_API_KEY) and restart.")
    from openai import OpenAI
    client = OpenAI(api_key=key)
    out = _out_path("openai")
    params = dict(model="gpt-4o-mini-tts", voice=voice, input=text, response_format="wav")
    instr = (instructions or "").strip()
    if instr:
        params["instructions"] = instr  # omit entirely when empty; the API rejects null
    try:
        with client.audio.speech.with_streaming_response.create(**params) as resp:
            resp.stream_to_file(out)
    except Exception as e:
        raise gr.Error(f"OpenAI TTS failed: {e}")
    return out


KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex",
    "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi",
    "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei",
    "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang",
]

XTTS_SPEAKERS = [
    "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
    "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
    "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
    "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper",
    "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka",
    "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy",
    "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah",
    "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos",
    "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando",
    "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger",
    "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick",
    "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María",
    "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman",
    "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı",
    "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski",
]

XTTS_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
              "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]

try:
    SAMPLE = open(f"{BAKE}/sample.txt").read().strip()
except OSError:
    SAMPLE = "Right then, let's hear how this one sounds."

with gr.Blocks(title="TTS Bake-off") as demo:
    gr.Markdown(
        "# TTS Bake-off\n"
        "Type once, render across engines, compare by ear. Audio plays in your browser.\n\n"
        "_GPU engines (Chatterbox / XTTS / Dia) load on first use and only one stays "
        "resident at a time, so switching between them costs a reload. Kokoro (CPU) and "
        "OpenAI (cloud) are always instant to reach._"
    )
    text = gr.Textbox(label="Text to speak", value=SAMPLE, lines=3)

    with gr.Tabs():
        with gr.Tab("Kokoro (fast, local)"):
            gr.Markdown("Fast and clean. The only local engine quick enough for real-time read-aloud.")
            k_voice = gr.Dropdown(KOKORO_VOICES, value="am_michael", label="Voice (am_=US male, af_=US female, bm_/bf_=British)")
            k_speed = gr.Slider(0.5, 2.0, value=1.2, step=0.05, label="Speed")
            k_btn = gr.Button("Generate (Kokoro)", variant="primary")
            k_out = gr.Audio(label="Kokoro output", type="filepath", autoplay=True)
            k_btn.click(gen_kokoro, [text, k_voice, k_speed], k_out)

        with gr.Tab("OpenAI (fast, cloud)"):
            gr.Markdown("Streaming cloud TTS, low latency and natural. Needs an OpenAI key in `~/tts-bakeoff/.openai_key`. Text leaves your machine.")
            o_voice = gr.Dropdown(OPENAI_VOICES, value="onyx", label="Voice")
            o_instr = gr.Textbox(label="Tone instructions (optional)", placeholder="e.g. Calm, warm, conversational. Speak a touch faster.", lines=2)
            o_btn = gr.Button("Generate (OpenAI)", variant="primary")
            o_out = gr.Audio(label="OpenAI output", type="filepath", autoplay=True)
            o_btn.click(gen_openai, [text, o_voice, o_instr], o_out)

        with gr.Tab("Chatterbox (local, slow)"):
            gr.Markdown("Expressive. Higher exaggeration = more emotion; lower CFG = looser pacing. Drop a clip to clone a voice.")
            c_exag = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Exaggeration")
            c_cfg = gr.Slider(0.2, 1.0, value=0.5, step=0.05, label="CFG weight")
            c_ref = gr.Audio(label="Optional: reference clip to clone (5-15s)", sources=["upload", "microphone"], type="filepath")
            c_btn = gr.Button("Generate (Chatterbox)", variant="primary")
            c_out = gr.Audio(label="Chatterbox output", type="filepath", autoplay=True)
            c_btn.click(gen_chatterbox, [text, c_exag, c_cfg, c_ref], c_out)

        with gr.Tab("XTTS-v2 (local, slow)"):
            gr.Markdown("58 built-in voices, or drop a clip to clone (cloning overrides the chosen speaker).")
            x_spk = gr.Dropdown(XTTS_SPEAKERS, value="Andrew Chipper", label="Built-in speaker")
            x_lang = gr.Dropdown(XTTS_LANGS, value="en", label="Language")
            x_ref = gr.Audio(label="Optional: reference clip to clone (6s+)", sources=["upload", "microphone"], type="filepath")
            x_btn = gr.Button("Generate (XTTS-v2)", variant="primary")
            x_out = gr.Audio(label="XTTS-v2 output", type="filepath", autoplay=True)
            x_btn.click(gen_xtts, [text, x_spk, x_lang, x_ref], x_out)

        with gr.Tab("Dia (local, slowest)"):
            gr.Markdown("Ultra-expressive dialogue model. Single-speaker text is auto-tagged. Heaviest render; not for real-time.")
            d_btn = gr.Button("Generate (Dia)", variant="primary")
            d_out = gr.Audio(label="Dia output", type="filepath", autoplay=True)
            d_btn.click(gen_dia, [text], d_out)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True, allowed_paths=[OUT])
