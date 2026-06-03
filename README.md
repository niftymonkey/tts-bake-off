# TTS Bake-off

A local playground for comparing text-to-speech engines side by side. Type a line once,
render it across five engines, and listen in the browser to pick a voice.

## Engines

| Tab | Where | Speed | Notes |
|-----|-------|-------|-------|
| Kokoro | local CPU | fast | 54 voices, speed slider. Real-time capable. |
| OpenAI gpt-4o-mini-tts | cloud | fast | Streaming, steerable tone. Real-time capable. Sends text to OpenAI. |
| Cartesia Sonic | cloud | fast | Lowest-latency streaming (~90ms), very natural. Sends text to Cartesia. |
| ElevenLabs Flash v2.5 | cloud | fast | Most human-sounding, ~75ms streaming, huge voice library. Sends text to ElevenLabs. |
| Chatterbox | local GPU | slow | Exaggeration / CFG sliders, voice cloning. |
| XTTS-v2 | local GPU | slow | 58 built-in speakers, voice cloning. |
| Dia | local GPU | slowest | Ultra-expressive dialogue model. |

The cloud engines (**OpenAI**, **Cartesia**, **ElevenLabs**) and **Kokoro** (local CPU)
are all fast enough for real-time read-aloud. The GPU engines are for quality comparison,
not production use. Only one GPU model stays resident at a time (11GB budget), so
switching between Chatterbox, XTTS, and Dia forces a reload; the cloud engines and Kokoro
are always available.

Each cloud tab has a curated voice dropdown plus a custom-voice-ID box. Note: on a free
ElevenLabs plan the API only allows the premade voices, not voice-library voices.

## Requirements

- Linux / WSL2 with WSLg (audio + GPU passthrough). Built against an RTX 2080 Ti
  (11GB, Turing / sm_75), 12-core CPU, 23GB RAM.
- [`uv`](https://github.com/astral-sh/uv) for virtual-environment management.
- linuxbrew `ffmpeg` (torchcodec needs it for XTTS).
- An OpenAI API key for the OpenAI tab.

## Setup

Each engine runs in its own `uv` virtual environment (`venv-*/`) because their
dependencies conflict and cannot share an interpreter. The venvs and the Kokoro model
weights are gitignored, so recreate them on a fresh clone with:

```sh
./setup.sh
```

This rebuilds each engine's venv from the committed lockfiles in `requirements/` and
downloads the Kokoro weights into `models/`. The Chatterbox, XTTS, and Dia weights
lazy-download from Hugging Face on first generate. Run `./setup.sh --help` for options
(`--with-orpheus`, `--force`, `--venvs-only`, `--models-only`).

Each cloud tab reads its API key from an environment variable, or from a key file in the
project root as a fallback:

| Tab | Env var | Key file fallback |
|-----|---------|-------------------|
| OpenAI | `OPENAI_API_KEY` (or `OPEN_AI_TTS_KEY`) | `.openai_key` |
| Cartesia | `CARTESIA_API_KEY` | `.cartesia_key` |
| ElevenLabs | `ELEVENLABS_API_KEY` | `.elevenlabs_key` |

A tab without a key still loads; it just errors when you click Generate. Restart the app
after adding a key.

## Run

```sh
venv-ui/bin/python app.py     # serves http://localhost:7860
```

Open `http://localhost:7860` using the literal `localhost` so the browser treats it as a
secure context. Otherwise the in-browser mic for voice cloning is disabled.

## Layout

- `app.py`: Gradio front-end. Talks to per-engine workers; calls OpenAI in-process.
- `engines/*_worker.py`: one warm worker per engine. Each loads its model once and serves
  one JSON request per line over a stdin/stdout pipe, run by that engine's venv.
- `sample.txt`: default comparison line.
- `venv-*/`: one uv venv per engine (gitignored; deps conflict, so they cannot share).
- `models/`: Kokoro ONNX model + voices (gitignored).
- `out/`, `logs/`: generated audio and worker logs (gitignored).

## Architecture notes

- Audio plays in-browser (Gradio's audio component), deliberately avoiding the WSLg
  PulseAudio server, which jams if a raw-stream player hangs.
- GPU workers are mutually exclusive, enforced by `GPU_LOCK` in `app.py`.
- uv venvs proved relocatable for direct `bin/python` invocation, which is how this whole
  directory was moved without reinstalling anything.

## Roadmap

- An Orpheus tab (`venv-orpheus` is installed; needs a worker plus a Turing / bf16
  shakedown). `./setup.sh --with-orpheus` already rebuilds the venv.
