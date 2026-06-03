"""XTTS-v2 warm worker. Run by venv-xtts python (CUDA).

Caller must set LD_LIBRARY_PATH to the linuxbrew lib dir so torchcodec can find
FFmpeg. Loads the model once, then serves one JSON request per line.

Request:  {"text": str, "out": str, "speaker": str, "language": str,
           "speaker_wav": str|null}   (speaker_wav = reference clip to clone;
                                        overrides the built-in speaker)
Response: {"ready": true}, then {"ok": bool, "out"/"error": str}
"""
import os
import sys
import json

os.environ["COQUI_TOS_AGREED"] = "1"

_real_stdout = os.dup(1)
os.dup2(2, 1)


def emit(obj):
    os.write(_real_stdout, (json.dumps(obj) + "\n").encode())


from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
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
        language = req.get("language", "en")
        if req.get("speaker_wav"):
            tts.tts_to_file(
                text=req["text"], file_path=req["out"],
                speaker_wav=req["speaker_wav"], language=language,
            )
        else:
            tts.tts_to_file(
                text=req["text"], file_path=req["out"],
                speaker=req.get("speaker", "Andrew Chipper"), language=language,
            )
        emit({"ok": True, "out": req["out"]})
    except Exception as e:
        emit({"ok": False, "error": repr(e)})
