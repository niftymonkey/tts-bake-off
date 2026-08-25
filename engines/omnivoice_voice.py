"""Bake a fixed OmniVoice speaker into a file the worker reuses on every call.

Voice-design mode ("male, american accent") does not pin a speaker. `generate()`
has no seed parameter at all, so the description is all the model gets and it
invents a fresh speaker each call. One reply split across chunks therefore comes
back in several voices. Voice-clone mode does pin it: a `VoiceClonePrompt` built
once from reference audio makes every later call the same person.

Two steps, because a designed voice has to be auditioned before it is worth
keeping:

    audition   synthesize N candidates from a description, one WAV each
    bake       turn the WAV you liked into the prompt file the worker loads

A recording of a real voice can be baked directly, skipping the audition.

Run with the OmniVoice venv, from the repo root:

    venv-omnivoice/bin/python engines/omnivoice_voice.py audition \
        "male, american accent, warm and measured" --n 5
    venv-omnivoice/bin/python engines/omnivoice_voice.py bake out/voice-3.wav
"""
import argparse
import os
import sys

import soundfile as sf
import torch

MODEL = os.environ.get("OMNIVOICE_MODEL") or "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
# float16 for the same reason the worker uses it: Turing has no native bfloat16.
DTYPE = torch.float16
NUM_STEP = int(os.environ.get("OMNIVOICE_STEPS", 32))

STATE = os.environ.get("VOICE_DIR") or os.path.expanduser("~/.claude/voice")
DEFAULT_PROMPT = os.environ.get("OMNIVOICE_VOICE_PROMPT") or f"{STATE}/omnivoice-voice.pt"

# The audition line is long enough to give the cloner ~12s of varied prosody and
# short enough to stay under the 20s the library trims at. It mixes a statement,
# a question and a list so the baked voice is not shaped by one flat sentence.
REF_TEXT = (
    "Right, let me walk you through what actually changed here. "
    "The build is green, the tests pass, and nothing else moved. "
    "Do you want me to push it now, or wait until the review lands? "
    "There are three things left: the config, the docs, and one flaky test."
)


def load_model():
    from omnivoice import OmniVoice
    return OmniVoice.from_pretrained(MODEL, device_map="cuda:0", dtype=DTYPE)


def audition(args):
    model = load_model()
    os.makedirs(args.dir, exist_ok=True)
    for i in range(1, args.n + 1):
        wav = f"{args.dir}/voice-{i}.wav"
        audio = model.generate(text=REF_TEXT, instruct=args.description,
                               num_step=NUM_STEP)
        sf.write(wav, audio[0], SAMPLE_RATE)
        # The transcript travels with the audio so `bake` never has to be told it.
        with open(f"{wav}.txt", "w") as f:
            f.write(REF_TEXT)
        print(wav)
    print(f"\nListen to all {args.n}, then bake the one you want:\n"
          f"  {sys.argv[0]} bake {args.dir}/voice-N.wav")


def bake(args):
    text = args.text
    if text is None:
        side = f"{args.wav}.txt"
        if os.path.exists(side):
            text = open(side).read().strip()
    if text is None:
        raise SystemExit(f"No transcript. Pass --text, or put it in {args.wav}.txt")

    model = load_model()
    prompt = model.create_voice_clone_prompt(ref_audio=args.wav, ref_text=text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    prompt.save(args.out)
    print(f"baked {args.out}")

    # Prove the prompt reproduces: two separate calls, the way two chunks of one
    # reply are two separate calls. They should be the same person.
    for i, line in enumerate(
        ["This is the first chunk of a reply.",
         "And this is a second chunk, generated separately."], 1):
        wav = f"{args.out}.check-{i}.wav"
        audio = model.generate(text=line, voice_clone_prompt=prompt, num_step=NUM_STEP)
        sf.write(wav, audio[0], SAMPLE_RATE)
        print(f"check {wav}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audition", help="synthesize N candidate voices to pick from")
    a.add_argument("description",
                   help='comma-separated terms from OmniVoice\'s fixed list, e.g. '
                        '"male, american accent, middle-aged, moderate pitch". '
                        'Free text is rejected; a bad term prints the whole list.')
    a.add_argument("--n", type=int, default=5)
    a.add_argument("--dir", default="out", help="where to write the candidate WAVs")
    a.set_defaults(func=audition)

    b = sub.add_parser("bake", help="turn a WAV into the prompt file the worker loads")
    b.add_argument("wav", help="reference audio: an audition candidate or a recording")
    b.add_argument("--text", default=None,
                   help="transcript; defaults to <wav>.txt when that exists")
    b.add_argument("--out", default=DEFAULT_PROMPT)
    b.set_defaults(func=bake)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
