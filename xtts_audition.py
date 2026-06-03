import os
os.environ["COQUI_TOS_AGREED"] = "1"
import re
from TTS.api import TTS

AUDITION = "Here's how I sound reading your code reviews and walking through a tricky bug."
CANDIDATES = ["Aaron Dreschner", "Andrew Chipper", "Craig Gutsy", "Viktor Eka"]
OUT = "/home/mlo/dev/niftymonkey/tts-bake-off/out"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
spks = list(tts.speakers or [])
print(f"=== all {len(spks)} XTTS speakers ===")
for i, s in enumerate(spks):
    print(f"{i:2d}. {s}")

def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

print("=== rendering candidates ===")
for name in CANDIDATES:
    if name not in spks:
        print("skip (not in model):", name)
        continue
    out = f"{OUT}/xtts-{slug(name)}.wav"
    tts.tts_to_file(text=AUDITION, file_path=out, speaker=name, language="en")
    print("rendered:", name, "->", out)
