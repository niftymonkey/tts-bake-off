"""OmniVoice warm worker. Run by venv-omnivoice python.

Same contract as kokoro_worker.py: load the model once, then read one JSON
request per line from stdin and write one JSON response per line, with all
library chatter pushed to stderr so stdout carries only the protocol.

`voice` is either a path to a baked voice-clone prompt or a voice-design
description ("male, american accent"). Prefer the prompt: design mode does not
pin a speaker, so each request invents a new one and a reply split across
requests comes back in several voices. `engines/omnivoice_voice.py` bakes one.

Request:  {"text": str, "out": str, "voice": str, "speed": float}
Response: {"ready": true}  on startup, then {"ok": bool, "out"/"error": str}
"""
import os
import sys
import json

# Keep stdout clean for the protocol: send everything libraries print to stderr.
_real_stdout = os.dup(1)
os.dup2(2, 1)


def emit(obj):
    os.write(_real_stdout, (json.dumps(obj) + "\n").encode())


import soundfile as sf
import torch
from omnivoice import OmniVoice

MODEL = os.environ.get("OMNIVOICE_MODEL") or "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
# The 2080 Ti is Turing, which has no native bfloat16 units, so the project's
# default precision would run emulated. float16 was checked for NaNs and clipping
# on this card before being chosen.
DTYPE = torch.float16
# Diffusion steps. The project's own default is 32; 16 is its documented "faster"
# setting and measured ~1.6x quicker on this card at the chunk sizes say.py uses.
NUM_STEP = int(os.environ.get("OMNIVOICE_STEPS", 16))

model = OmniVoice.from_pretrained(MODEL, device_map="cuda:0", dtype=DTYPE)

# Loading a prompt costs a `torch.load`, so keep each one for the daemon's life.
_prompts = {}


def _prompt(path):
    """Load a baked speaker, keyed on path AND mtime.

    Baking a new voice rewrites the same path, so a path-only cache would keep
    serving the old speaker until the daemon idled out, with nothing to say why.
    """
    key = (path, os.path.getmtime(path))
    if key not in _prompts:
        from omnivoice.models.omnivoice import VoiceClonePrompt
        _prompts.clear()  # only the current voice is ever wanted again
        _prompts[key] = VoiceClonePrompt.load(path)
    return _prompts[key]


emit({"ready": True})

while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        voice = req.get("voice") or "male, american accent"
        # A path means a baked speaker; anything else is a design description.
        # A missing prompt file is an error rather than a fallback: falling back
        # to design mode would quietly reinstate the drifting voice it exists to
        # prevent, and sound like nothing was wrong.
        if voice.endswith(".pt"):
            if not os.path.exists(voice):
                raise FileNotFoundError(f"voice-clone prompt not found: {voice}")
            style = {"voice_clone_prompt": _prompt(voice)}
        else:
            style = {"instruct": voice}
        audio = model.generate(
            text=req["text"],
            speed=float(req.get("speed", 1.0)),
            num_step=NUM_STEP,
            **style,
        )
        sf.write(req["out"], audio[0], SAMPLE_RATE)
        emit({"ok": True, "out": req["out"]})
    except Exception as e:
        emit({"ok": False, "error": repr(e)})
