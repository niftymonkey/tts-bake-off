# TTS Bake-off task runner. Run `just` with no args to see this list.

# Show available recipes
default:
    @just --list

# Run the app the normal way (no reload). Serves http://localhost:7860
run:
    venv-ui/bin/python app.py

# Developer mode: Gradio hot-reload, restarts the app when you save app.py
dev:
    # API keys still load the same way (env var or key file).
    # Invoked as a module: the venv was relocated, so bin/gradio's shebang is stale.
    GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 venv-ui/bin/python -m gradio app.py

# Rebuild venvs and fetch models. Passes flags through, e.g. `just setup --force`
setup *args:
    ./setup.sh {{args}}

# Kill stuck read-aloud/audio processes and report PulseAudio health
clean:
    bash cleanup_audio.sh

# Tail a worker log. Defaults to the app log; e.g. `just logs kokoro`
logs name="app":
    tail -n 40 -f logs/{{name}}.log
