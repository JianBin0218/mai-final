"""Frozen question-only taxonomy for A-line evaluation."""

import csv
import re
import unicodedata
from collections import defaultdict


TYPES = ("source_or_event", "count_or_relation", "time_pattern", "scene_context", "acoustic_property")


def canonical(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().replace("’", "'").split())


RULES = tuple((category, re.compile(pattern)) for category, pattern in (
    ("count_or_relation", r"\b(how many|more than (one|two|three)|more th[ae]t? a dozen|at least (one|two|three)|"
     r"multiple|several|only (one|a single|males|females)|one single|alone|isolated|"
     r"outnumber\w*|a (group|crowd|variety) of|a lot of|many|twice|once|"
     r"(two|three|four|five|six|different) (separate times|times|different voices|voices|people|"
     r"birds|cars|animals|sources|sounds|instruments|speakers|dogs|vehicles|species|events|types)|"
     r"both male and female|only thing that can be heard)\b"),
    ("time_pattern", r"\b(constant\w*|continuous\w*|steady|consisten\w* (rhythm|speed)|rhythm\w*|sequence|"
     r"repetiti\w*|repeats?|beginning|at the (start|end)|in the middle|toward\w* the end|"
     r"first half|second half|as time goes on|each time|each break|in between|pause\w*|"
     r"stop\w*|start\w* (and|then|again|before|after)|before the|after the|"
     r"quickly|slowly|speeding up|slow\w* down|becom\w* (noticeably )?(louder|quieter)|"
     r"get\w* (any )?(louder|quieter)|grow\w* louder|change\w* (speed|tempo|pitch)|"
     r"speed .* change|stay\w* at the same)\b"),
    ("count_or_relation", r"\b(background|foreground|closer|further|farther|nearby|"
     r"far away|close by|distant|in front of|behind|left (side|channel)|right (side|channel)|"
     r"side by side|singing together|louder than|quieter than)\b"),
    ("acoustic_property", r"\b(loud\w*|quiet\w*|high[- ]pitch\w*|low[- ]pitch\w*|high toned|low toned|"
     r"treble|bass|static|electronic (noise|sound)|sound electronic|distortion|distorted|"
     r"annoying|noisy|chaotic|soft and pleasant|shrill|muffled|tinny|resonant|"
     r"high[- ]frequen\w*|low[- ]frequen\w*)\b"),
    ("scene_context", r"\b(indoors|outdoors|inside|outside|in nature|city|"
     r"roadway|crowded|dry sunny day|wet day|good day to fly a kite|outdoor environment|"
     r"stage show|set up correctly|wd-40|in (a |the )?(kitchen|house|bathroom|ocean|forest|"
     r"building|car|room|park|city)|at (a |the )?(concert|beach|station|school|market)|"
     r"near the ocean)\b"),
))


def classify(question, exemplars):
    key = canonical(question)
    if key in exemplars:
        return exemplars[key], "exemplar"
    for category, pattern in RULES:
        if pattern.search(key):
            return category, "rule"
    return "source_or_event", "rule"


def load_exemplars(path):
    labels = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["type"] not in TYPES:
                raise ValueError(f"Unexpected question type: {row['type']}")
            labels[canonical(row["question"])].add(row["type"])
    return {key: next(iter(types)) for key, types in labels.items() if len(types) == 1}
