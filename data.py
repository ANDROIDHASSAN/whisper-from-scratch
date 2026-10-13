"""FLEURS (Conneau et al. 2022, CC-BY 4.0) Hindi, Marathi and English -> Whisper input features on disk.

    python data.py                       # all three languages, all splits
    python data.py --langs hi            # one language

Every clip becomes a 30 s window (80 x 3000). Most clips are under 15 s, so most of each window is the
silence padding, which after Whisper's normalization is one constant per clip (log10 of zero power is floored
at max - 8). We store only the frames that touch real audio plus that constant, and rebuild the full window
when batching; check.py shows the rebuilt window equals the directly computed one exactly.

    data/<split>.mel     float16 frames, clip after clip, (frames, 80)
    data/<split>.json    per clip: start, frames, floor, language, raw transcript
    data/bpe.json        the Indic tokenizer, learned on the training transcripts
"""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from audio import FRAMES, HOP, N_FFT, N_MELS, SR, features, read_wav
from tokenizer import IndicTokenizer

DATA = Path(__file__).parent / "data"
FLEURS = {"hi": "hi_in", "mr": "mr_in", "en": "en_us"}


def compact(feat: torch.Tensor, samples: int) -> tuple[np.ndarray, float]:
    """(80, 3000) -> (the frames whose window overlaps audio, as (n, 80) float16; the padding value)."""
    n = min(FRAMES, (samples + N_FFT // 2) // HOP + 2)
    floor = float(feat[0, -1]) if n < FRAMES else 0.0
    return feat[:, :n].T.numpy().astype(np.float16), floor


def expand(frames: np.ndarray, floor: float) -> torch.Tensor:
    out = torch.full((N_MELS, FRAMES), floor)
    out[:, : len(frames)] = torch.from_numpy(frames.astype(np.float32)).T
    return out


def build(split: str, langs: list[str]) -> list[dict]:
    index, offset = [], 0
    with open(DATA / f"{split}.mel", "wb") as out:
        for lang in langs:
            get = lambda f: hf_hub_download("google/fleurs", f"data/{FLEURS[lang]}/{f}", repo_type="dataset",
                                            local_dir=DATA / "fleurs")
            rows = {r[1]: r for r in (line.rstrip("\n").split("\t") for line in open(get(f"{split}.tsv"), encoding="utf-8"))}
            count = 0
            with tarfile.open(get(f"audio/{split}.tar.gz"), "r|gz") as archive:
                for member in archive:
                    row = rows.get(Path(member.name).name)
                    if row is None or not member.isfile():
                        continue
                    wave, rate = read_wav(archive.extractfile(member).read())
                    if rate != SR or len(wave) > 30 * SR or not row[2].strip():
                        continue
                    frames, floor = compact(features(wave), len(wave))
                    out.write(frames.tobytes())
                    index.append({"lang": lang, "text": row[2].strip(), "start": offset, "frames": len(frames), "floor": floor})
                    offset += len(frames)
                    count += 1
            print(f"  {split:5s} {lang}: {count} clips", flush=True)
    (DATA / f"{split}.json").write_text(json.dumps(index, ensure_ascii=False))
    return index


def open_split(split: str) -> tuple[np.ndarray, list[dict]]:
    index = json.loads((DATA / f"{split}.json").read_text())
    return np.memmap(DATA / f"{split}.mel", dtype=np.float16, mode="r").reshape(-1, N_MELS), index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=list(FLEURS), choices=list(FLEURS))
    ap.add_argument("--vocab", type=int, default=4000)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for split in ("dev", "test", "train"):
        index = build(split, args.langs)
    tok = IndicTokenizer.train([e["text"] for e in index], args.vocab)
    tok.save(DATA / "bpe.json")
    print(f"Indic tokenizer: {tok.vocab_size} tokens including Whisper's special tokens")


if __name__ == "__main__":
    main()
