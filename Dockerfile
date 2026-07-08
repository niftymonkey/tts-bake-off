# CPU-only TTS Bake-off: Kokoro (local) + OpenAI / Cartesia / ElevenLabs / Deepgram (cloud).
# The GPU engines (Chatterbox, XTTS, Dia) are intentionally excluded; they need CUDA
# venvs and multi-GB weights and are not the always-on, real-time use case.
FROM python:3.12-slim

# uv drives the per-engine venvs, exactly as setup.sh does on the host.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# libsndfile1: soundfile (Kokoro output). libgomp1: onnxruntime. curl: Kokoro weights.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kokoro and the UI keep separate venvs because their deps conflict, same as on the
# host. Build them first (from the lockfiles alone) so this layer caches across app edits.
COPY requirements/kokoro.lock requirements/ui.lock requirements/
RUN uv venv venv-kokoro --python 3.12 \
    && uv pip install --python venv-kokoro/bin/python -r requirements/kokoro.lock \
    && uv venv venv-ui --python 3.12 \
    && uv pip install --python venv-ui/bin/python -r requirements/ui.lock

# Bake the Kokoro weights into the image so the container is self-contained on boot.
# Sizes match setup.sh; a truncated download fails the build rather than shipping silently.
RUN mkdir -p models \
    && curl -fL --retry 3 -o models/kokoro-v1.0.onnx \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" \
    && curl -fL --retry 3 -o models/voices-v1.0.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
    && [ "$(stat -c %s models/kokoro-v1.0.onnx)" = "325532387" ] \
    && [ "$(stat -c %s models/voices-v1.0.bin)" = "28214398" ]

COPY app.py sample.txt ./
COPY engines/kokoro_worker.py engines/

ENV BAKE_DIR=/app \
    MODELS_DIR=/app/models \
    GPU_ENGINES=0

EXPOSE 7860
CMD ["venv-ui/bin/python", "app.py"]
