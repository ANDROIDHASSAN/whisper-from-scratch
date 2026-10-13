"""Byte-pair encoding (Sennrich, Haddow & Birch 2016, arXiv 1508.07909), byte-level variant.

The Transformer paper (section 5.1) encodes English-German with "byte-pair encoding, which has a shared
source-target vocabulary of about 37000 tokens". We do the same for English-Hindi: ONE vocabulary learned on
both languages, so the encoder embedding, the decoder embedding and the output layer can share one matrix
(section 3.4).

Sennrich's BPE starts from characters. We start from the 256 bytes instead (as GPT-2 does): Devanagari,
Latin, digits and anything else in the corpus can always be encoded, and there is never an unknown token.
One Devanagari letter is 3 bytes, so the first merges rebuild letters and matras before they reach syllables.

    1. vocabulary = the 256 bytes
    2. glue the most frequent neighbouring pair into a new token
    3. repeat until the vocabulary has the size you asked for
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>"]

# Cut text into words (with their leading space) and never merge across a word border. Training then only
# has to count each distinct word once.
CHUNK = re.compile(r" ?\S+|\s")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self, merges: list[tuple[int, int]]) -> None:
        self.merges = merges  # merge i creates token id 256 + i
        self.rank = {pair: 256 + i for i, pair in enumerate(merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.rank.items():
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        first = 256 + len(merges)
        self.pad, self.bos, self.eos = first, first + 1, first + 2
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(SPECIALS)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPE":
        freq = Counter(CHUNK.findall(text))
        words = [list(w.encode("utf-8")) for w in freq]
        counts = list(freq.values())

        # pair_count: how often each pair occurs; where: which words hold it, so a merge only touches those.
        pair_count: Counter = Counter()
        where: defaultdict = defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += counts[i]
                where[p].add(i)

        merges: list[tuple[int, int]] = []
        while len(merges) < vocab_size - 256 - len(SPECIALS):
            best = max(pair_count, key=lambda p: (pair_count[p], p))  # ties broken by the pair: deterministic
            if pair_count[best] < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for i in where.pop(best):
                w = words[i]
                for p in zip(w, w[1:]):
                    pair_count[p] -= counts[i]
                w = words[i] = merge(w, best, new_id)
                for p in zip(w, w[1:]):
                    pair_count[p] += counts[i]
                    where[p].add(i)
            del pair_count[best]
        return cls(merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))  # earliest merge first
            if pair not in self.rank:
                break
            ids = merge(ids, pair, self.rank[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in CHUNK.findall(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            ids.extend(self._cache[chunk])
        return ids

    def decode(self, ids: list[int]) -> str:
        # Join the bytes first, decode once: one letter can be split across two tokens.
        return b"".join(self.vocab[i] for i in ids if i < self.pad).decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"merges": self.merges}))

    @classmethod
    def load(cls, path: Path) -> "BPE":
        return cls([tuple(p) for p in json.loads(path.read_text())["merges"]])
