"""Transcript normalization and error rates.

WER = word edits (insert, delete, substitute) / words in the reference; CER the same on characters.
Both sides are normalized first (lowercase, punctuation removed, Unicode NFC) so that only what can be
heard is scored. FLEURS' own normalized transcription column is the reference.
"""

import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    kept = (c if unicodedata.category(c)[0] in "LMN" else " " for c in text)  # letters, marks (matras), digits
    return " ".join("".join(kept).split())


def edit_distance(ref: list, hyp: list) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (r != h))
    return row[-1]


def error_rate(refs: list[str], hyps: list[str], unit: str = "word") -> float:
    split = str.split if unit == "word" else list
    edits = sum(edit_distance(split(normalize(r)), split(normalize(h))) for r, h in zip(refs, hyps, strict=True))
    return edits / max(1, sum(len(split(normalize(r))) for r in refs))
