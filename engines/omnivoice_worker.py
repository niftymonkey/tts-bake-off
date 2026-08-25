"""OmniVoice warm worker. Run by venv-omnivoice python.

Same contract as kokoro_worker.py: load the model once, then read one JSON
request per line from stdin and write one JSON response per line, with all
library chatter pushed to stderr so stdout carries only the protocol.

`voice` here is a voice-design description ("male, american accent") rather than
a name from a fixed list. The seed is pinned because the description alone does
not fully determine the speaker, and an utterance split across several requests
has to come back in one voice.

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
VOICE_SEED = 42
# Diffusion steps. The project's own default is 32; 16 is its documented "faster"
# setting and measured ~1.6x quicker on this card at the chunk sizes say.py uses.
NUM_STEP = int(os.environ.get("OMNIVOICE_STEPS", 16))

model = OmniVoice.from_pretrained(MODEL, device_map="cuda:0", dtype=DTYPE)
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
        audio = model.generate(
            text=req["text"],
            instruct=req.get("voice") or "male, american accent",
            speed=float(req.get("speed", 1.0)),
            seed=VOICE_SEED,
            num_step=NUM_STEP,
        )
        sf.write(req["out"], audio[0], SAMPLE_RATE)
        emit({"ok": True, "out": req["out"]})
    except Exception as e:
        emit({"ok": False, "error": repr(e)})
