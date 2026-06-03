"""Dia (Nari Labs) warm worker. Run by venv-dia python (CUDA).

Dia is a dialogue TTS that expects speaker tags ([S1]/[S2]); single-speaker text
is prefixed with [S1]. Forced to float16 because the RTX 2080 Ti (Turing, sm_75)
has no native bfloat16. Loads once, serves one JSON request per line.

Request:  {"text": str, "out": str}
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
from dia.model import Dia

model = None
last_err = None
for repo in ("nari-labs/Dia-1.6B-0626", "nari-labs/Dia-1.6B"):
    try:
        model = Dia.from_pretrained(repo, compute_dtype="float16")
        break
    except Exception as e:
        last_err = e

if model is None:
    emit({"ready": False, "error": repr(last_err)})
    sys.exit(1)

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
        text = req["text"]
        if not text.lstrip().startswith("[S"):
            text = "[S1] " + text
        audio = model.generate(text, use_torch_compile=False, verbose=False)
        model.save_audio(req["out"], audio)
        emit({"ok": True, "out": req["out"]})
    except Exception as e:
        emit({"ok": False, "error": repr(e)})
