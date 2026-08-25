"""PROTOTYPE. Candidate spoken registers for Claude Code voice mode.

Question being answered: when Claude finishes a turn and the Stop hook speaks
something, what should that something sound like?

Second cut. The first cut lost on length: two of the four candidates ran nearly
as long as the text on screen, which defeats the point. All four now start from
the work-recap rule already in the user's CLAUDE.md, "tell me about it like a
fellow engineer at my desk, flowing spoken prose, cut detail not wording, and
trivial work gets one sentence." They differ in how they spend a small budget,
not in whether they sound conversational.

`sanitize` is the part worth keeping; it lifts straight into the real hook.
"""
import re


# Measured 2026-07-31 over all 20 register/fixture combinations rendered by
# Kokoro am_michael at speed 1.2: 475 words in 149.9s. Per-utterance rate ranged
# 2.24 to 3.77, so the TUI estimate is indicative, not exact. Used only to put a
# visible time cost next to each candidate.
WORDS_PER_SECOND = 3.17


# --- spoken-text cleanup -------------------------------------------------

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]+`")
# Before _URL, which would take the closing paren with the address and strand
# the label: "[docs](http://x)" became "docs(" rather than "docs".
_MD_LINK = re.compile(r"\[([^\]]+)\]\(https?://[^)\s]+\)")
_URL = re.compile(r"https?://\S+")
_PATH_LINE = re.compile(r"\b[\w./-]+\.(py|ts|tsx|js|json|sh|md|yaml|yml):\d+\b")
_MD_PUNCT = re.compile(r"[*_#>|\[\]]+")
_WS = re.compile(r"\s+")


def sanitize(text):
    """Strip everything unlistenable from markdown, leaving speakable prose.

    Fenced code, inline code, URLs, and file:line references all read as noise
    through a TTS engine, so they come out entirely rather than being narrated.
    """
    text = _FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _PATH_LINE.sub(" ", text)
    text = _MD_PUNCT.sub("", text)
    text = _WS.sub(" ", text)
    return text.strip()


def spoken_seconds(text):
    return len(text.split()) / WORDS_PER_SECOND


# --- register registry ---------------------------------------------------


class Register:
    """One candidate spoken register.

    `budget` is the rough word ceiling the authored text was held to, shown in
    the TUI so the time cost of each candidate is visible without playing it.
    """

    def __init__(self, key, name, idea, budget):
        self.key = key
        self.name = name
        self.idea = idea
        self.budget = budget

    def spoken(self, fixture):
        return sanitize(fixture.authored[self.key])


REGISTERS = [
    Register(
        "A", "One breath",
        "The single most load-bearing fact, said the way you would say it standing "
        "at someone's desk. Everything else is on screen. Never more than one breath.",
        "~15 words",
    ),
    Register(
        "B", "What changed, and what fought back",
        "Two sentences, the two atoms of the CLAUDE.md recap rule: what is now "
        "different, then the one thing that surprised you. All support detail dropped.",
        "~35 words",
    ),
    Register(
        "C", "Scaled to the substance",
        "No fixed length. Obeys 'trivial work gets one sentence' literally, so a "
        "throwaway answer is a clause and a real finding earns three or four sentences. "
        "Spends words only where there are words worth spending.",
        "1 sentence to ~60",
    ),
    Register(
        "D", "Colleague, then the ask",
        "What happened in the desk-colleague voice, ending on the decision or action "
        "that is now yours. Optimised for turns that hand something back to you.",
        "~30 words",
    ),
]

REGISTERS_BY_KEY = {r.key: r for r in REGISTERS}
