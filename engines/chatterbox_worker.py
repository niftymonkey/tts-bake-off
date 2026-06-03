"""Chatterbox warm worker. Run by venv-chatterbox python (CUDA).

Loads the model once on the GPU, then serves one JSON request per line.

Request:  {"text": str, "out": str, "exaggeration": float, "cfg_weight": float,
           "audio_prompt": str|null}   (audio_prompt = reference clip to clone)
Response: {"ready": true}, then {"ok": bool, "out"/"error": str}
"""
import os
import sys
import json

_real_stdout = os.dup(1)
os.dup2(2, 1)


def emit(obj):
    os.write(_real_stdout, (json.dumps(obj) + "\n").encode())


import torch  # noqa: F401
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
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
        kwargs = {}
        if req.get("exaggeration") is not None:
            kwargs["exaggeration"] = float(req["exaggeration"])
        if req.get("cfg_weight") is not None:
            kwargs["cfg_weight"] = float(req["cfg_weight"])
        if req.get("audio_prompt"):
            kwargs["audio_prompt_path"] = req["audio_prompt"]
        wav = model.generate(req["text"], **kwargs).detach().cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(req["out"], wav, model.sr)
        emit({"ok": True, "out": req["out"]})
    except Exception as e:
        emit({"ok": False, "error": repr(e)})
