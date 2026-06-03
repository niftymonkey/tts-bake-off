import sys
import soundfile as sf
from kokoro_onnx import Kokoro

MODELS = "/home/mlo/dev/niftymonkey/tts-bake-off/models"
SAMPLE = "/home/mlo/dev/niftymonkey/tts-bake-off/sample.txt"

voice = sys.argv[1] if len(sys.argv) > 1 else "am_michael"
out = sys.argv[2] if len(sys.argv) > 2 else "/home/mlo/dev/niftymonkey/tts-bake-off/out/kokoro.wav"
speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

with open(SAMPLE) as f:
    text = f.read().strip()

kokoro = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin")
samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
sf.write(out, samples, sample_rate)
print(f"wrote {out}: {len(samples)} samples @ {sample_rate} Hz, voice={voice}, speed={speed}")
