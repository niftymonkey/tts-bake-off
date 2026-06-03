"""Kokoro warm worker. Run by venv-kokoro python.

Loads the model once, then reads one JSON request per line from stdin and
writes one JSON response per line. All library chatter is redirected to
stderr so stdout carries only the protocol.

Request:  {"text": str, "out": str, "voice": str, "speed": float, "lang": str}
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


from kokoro_onnx import Kokoro
import soundfile as sf

MODELS = "/home/mlo/dev/niftymonkey/tts-bake-off/models"
kokoro = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
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
        samples, sample_rate = kokoro.create(
            req["text"],
            voice=req.get("voice", "am_michael"),
            speed=float(req.get("speed", 1.0)),
            lang=req.get("lang", "en-us"),
        )
        sf.write(req["out"], samples, sample_rate)
        emit({"ok": True, "out": req["out"]})
    except Exception as e:
        emit({"ok": False, "error": repr(e)})
