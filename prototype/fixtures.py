"""PROTOTYPE. Real turn shapes to hear each register against.

Lifted from an actual Claude Code session rather than invented, because the
registers only diverge meaningfully on real material. The five shapes cover the
cases that pull the design in different directions: a throwaway one-liner, a
long assessment, a question aimed back at the user, a tool-heavy turn whose
value is a table, and an explanation whose subject is code.

`markdown` is what Claude actually put on screen, kept so the TUI can show the
size gap between what you read and what you hear. `authored` holds the spoken
version for each register, written to that register's word budget.
"""


class Fixture:
    def __init__(self, key, name, shape, markdown, authored):
        self.key = key
        self.name = name
        self.shape = shape
        self.markdown = markdown.strip()
        self.authored = authored


FIXTURES = [
    Fixture(
        "1", "Speaker test", "trivial work, nothing below it",
        """
That was me. I played three test clips through your speakers to check whether the
WSLg audio path still works, since your old read-aloud hook was disabled over a
PulseAudio jamming bug. It's cut from an old Kokoro render sitting in `out/`. No
built-in feature, and nothing is listening or recording.

Useful result though: the audio path is healthy, playback adds about a second of
startup overhead per clip, and it didn't hang.
""",
        {
            "A": "That was me, just testing your speakers. They work.",
            "B": (
                "That was me testing your audio. It works, and it didn't jam the way "
                "your old hook used to."
            ),
            "C": "That was me testing your speakers, and they're fine.",
            "D": "That was me testing your audio, it works. Nothing for you to do.",
        },
    ),
    Fixture(
        "2", "Feasibility assessment", "long analysis with numbers and a table",
        """
**The good news is this is much easier than it looks, and you already built most of it.**

The hook side turns out to be nearly free. Claude Code fires a `Stop` hook when I
finish a turn, and its stdin payload includes `last_assistant_message` outright, the
complete response text. Your old Piper hook does a bunch of `jq` gymnastics to dig
that text out of the transcript JSONL, and none of that is necessary anymore.

| Engine | Time to WAV |
|--------|-------------|
| Cartesia | 1.87s |
| Deepgram | 2.09s |
| OpenAI | 4.11s |
| Kokoro | 4.71s cold |

Total build is roughly half a day.
""",
        {
            "A": "Half a day. Easier than expected, the hook hands me the response text outright.",
            "B": (
                "About half a day, and easier than I thought, since the hook gets my full "
                "response text handed to it. The catch is Kokoro's cold start, three and a "
                "half seconds just to load the model."
            ),
            "C": (
                "Call it half a day. The nice surprise is that the hook gets my whole response "
                "text handed to it, so all that transcript parsing your old one does is dead "
                "weight now. The catch is latency. Cartesia's about two seconds, Kokoro's nearly "
                "five cold because the model load eats three and a half, and playback adds "
                "another second on top of whichever you pick."
            ),
            "D": (
                "About half a day, and the plumbing's easier than expected. Latency numbers per "
                "engine are on screen, and you'll want to pick one before I start."
            ),
        },
    ),
    Fixture(
        "3", "Question back to you", "turn that ends by asking, not telling",
        """
Before we get to length and delimiters, I want to pin down the thing everything
else hangs on: what is your attention doing while the audio plays?

In the first, the audio is **primary**. You've pushed back from the desk, and the
spoken response is how you actually receive the answer.

In the second, the audio is **ambient**. You're still looking at the screen, and the
spoken track is a low-attention signal running alongside.

Which of those is closest to what you were picturing?
""",
        {
            "A": "Before I go further, are you listening to this or reading it?",
            "B": (
                "One thing before I go further. Whether this is how you actually receive the "
                "answer, or just a signal to look up, changes the whole design."
            ),
            "C": (
                "Quick question before I go further. If you've pushed back from the desk and "
                "this is how you actually get the answer, it needs to stand on its own. If "
                "you're still reading the screen, it should be much shorter. Which is it?"
            ),
            "D": (
                "One question before I go further, it's on screen. Two options, and the answer "
                "changes the whole design."
            ),
        },
    ),
    Fixture(
        "4", "Probe results", "tool-heavy turn whose value is the table",
        """
Ran the latency probe across all four fast engines against the same sentence.

| Engine | Cold | Warm |
|--------|------|------|
| Cartesia | 1.87s | n/a |
| Deepgram | 2.09s | n/a |
| OpenAI | 4.11s | n/a |
| Kokoro | 4.71s | 1.3-2.7s |
""",
        {
            "A": "Cartesia's fastest at just under two seconds. Deepgram's basically tied.",
            "B": (
                "Cartesia won at just under two seconds with Deepgram right behind. Kokoro's "
                "the outlier at nearly five cold, though most of that is loading the model once."
            ),
            "C": (
                "Ran all four. Cartesia's fastest at just under two seconds, Deepgram basically "
                "tied, OpenAI noticeably slower at four. Kokoro's nearly five cold, but that's "
                "mostly the model load, so warm it drops to one or two."
            ),
            "D": (
                "Cartesia's fastest, Deepgram's tied with it. Numbers are on screen, and this is "
                "the call you need to make before I wire anything."
            ),
        },
    ),
    Fixture(
        "5", "Header fix", "explanation whose subject is code",
        """
That's fixed. The WAV header was lying about its length, so players computed a
bogus duration and cut off the last few words.

```python
f.seek(4)
f.write(struct.pack("<I", size - 8))
```

See `app.py:130` for the full patch.
""",
        {
            "A": "Fixed. The WAV header was lying about its length.",
            "B": (
                "Fixed, and it was a lying WAV header. Players trusted the placeholder length "
                "and cut off the last few words."
            ),
            "C": (
                "That's fixed. These engines write the WAV header before they know how long the "
                "audio is, so they stamp in a placeholder, players trust it, and the last few "
                "words get cut. I rewrite the size fields once it's on disk."
            ),
            "D": (
                "Fixed, it was a lying WAV header. Patch is four lines on screen if you want to "
                "check it."
            ),
        },
    ),
]

FIXTURES_BY_KEY = {f.key: f for f in FIXTURES}
