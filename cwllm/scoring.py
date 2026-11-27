"""Assistant-only sudden death, solo health isolation and mirrored aggregation."""
from dataclasses import dataclass
from fractions import Fraction
import math
import re


@dataclass(frozen=True)
class KeywordPair:
    word_1: str
    word_2: str
    forms_1: tuple = ()
    forms_2: tuple = ()

    def mirrored(self):
        return KeywordPair(self.word_2, self.word_1, self.forms_2, self.forms_1)


def inflections(word, extra=()):
    word = word.lower()
    if not re.fullmatch(r"[a-z]+", word):
        raise ValueError("keyword must be an English noun lemma")
    forms = {word, word + "s", *(x.lower() for x in extra)}
    if word.endswith(("s", "x", "z", "ch", "sh", "o")):
        forms.add(word + "es")
    if len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        forms.add(word[:-1] + "ies")
    if word.endswith("fe"):
        forms.add(word[:-2] + "ves")
    elif word.endswith("f"):
        forms.add(word[:-1] + "ves")
    return forms


def keyword_pattern(word, extra=()):
    surfaces = sorted(inflections(word, extra), key=lambda x: (-len(x), x))
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(x) for x in surfaces) +
                      r")(?:['’]s|['’])?(?![\w])", re.IGNORECASE)


def contains_keyword(text, word, extra=()):
    return keyword_pattern(word, extra).search(text) is not None


def score_dialogue(messages, pair):
    patterns = (keyword_pattern(pair.word_1, pair.forms_1),
                keyword_pattern(pair.word_2, pair.forms_2))
    if inflections(pair.word_1, pair.forms_1) & inflections(pair.word_2, pair.forms_2):
        raise ValueError("taboo surface sets must be disjoint")
    for turn, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        found = [(m.start(), who, m.group()) for who, p in enumerate(patterns)
                 if (m := p.search(message["content"])) is not None]
        if found:
            offset, loser, surface = min(found)
            return {"score_A": float(loser), "score_B": float(1 - loser),
                    "winner": "B" if loser == 0 else "A", "turn": turn,
                    "offset": offset, "surface": surface}
    return {"score_A": 0.5, "score_B": 0.5, "winner": "draw",
            "turn": None, "offset": None, "surface": None}


@dataclass(frozen=True)
class HealthThresholds:
    tau_ppl: float = 1.0
    tau_rep: float = 0.20
    tau_must: float = 0.875


def adjudicate_solo_health(patch, metrics, *, valid=True, thresholds=None):
    thresholds = thresholds or HealthThresholds()
    failures = [] if valid else ["validation"]
    for name, threshold, lower in (("fluency_ppl_delta", thresholds.tau_ppl, False),
                                   ("repetition_rate", thresholds.tau_rep, False),
                                   ("must_say_recall", thresholds.tau_must, True)):
        value = metrics.get(name)
        if (type(value) not in (int, float) or not math.isfinite(value) or
                (value < threshold if lower else value > threshold)):
            failures.append(name)
    return {"patch": {"edits": []} if failures else patch,
            "fault_isolated": bool(failures), "failures": failures}


def aggregate_mirrored_worlds(world_1, world_2):
    if not world_1 or len(world_1) != len(world_2):
        raise ValueError("mirrored worlds require equal nonzero dialogue counts")
    rows = list(world_1) + list(world_2)
    if any(r["score_A"] not in (0, 0.5, 1) or r["score_A"] + r["score_B"] != 1 for r in rows):
        raise ValueError("invalid dialogue scores")
    a = sum((Fraction(r["score_A"]) for r in rows), Fraction()) / len(rows)
    return {"score_A": float(a), "score_B": float(1 - a), "dialogues": len(rows),
            "wins_A": sum(r["winner"] == "A" for r in rows),
            "draws": sum(r["winner"] == "draw" for r in rows),
            "wins_B": sum(r["winner"] == "B" for r in rows)}
