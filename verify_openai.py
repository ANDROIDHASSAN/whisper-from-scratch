"""Load OpenAI's released Whisper weights into this from-scratch implementation and transcribe FLEURS test
clips. If the architecture, the audio front-end, the tokenizer or the decoder differed from the original in
any detail, the transcripts would be garbage; a normal WER is the end-to-end proof that they match.

    python verify_openai.py --model openai/whisper-tiny --lang en --clips 50
"""

import argparse
import tarfile
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from audio import SR, features, read_wav
from decode import beam_search, detect_language, greedy
from model import load_pretrained
from text import error_rate
from tokenizer import OpenAITokenizer

FLEURS = {"en": "en_us", "hi": "hi_in", "mr": "mr_in"}


def fleurs_test(lang: str, n: int, cache: Path) -> list[tuple]:
    folder = f"data/{FLEURS[lang]}"
    get = lambda f: hf_hub_download("google/fleurs", f"{folder}/{f}", repo_type="dataset", local_dir=cache)
    rows = {r[1]: r for r in (line.rstrip("\n").split("\t") for line in open(get("test.tsv"), encoding="utf-8"))}
    clips = []
    with tarfile.open(get("audio/test.tar.gz"), "r|gz") as archive:
        for member in archive:
            row = rows.get(Path(member.name).name)
            if row and member.isfile():
                wave, rate = read_wav(archive.extractfile(member).read())
                if rate == SR:
                    clips.append((wave, row[3]))  # column 3: FLEURS' normalized transcription
            if len(clips) == n:
                break
    return clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--lang", default="en", choices=list(FLEURS))
    ap.add_argument("--clips", type=int, default=50)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--fleurs-cache", type=Path, default=Path("data/fleurs"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_pretrained(args.model).to(args.device).eval()
    tok = OpenAITokenizer(args.model)
    clips = fleurs_test(args.lang, args.clips, args.fleurs_cache)
    t0, refs, hyps, detected = time.time(), [], [], 0
    for wave, ref in clips:
        mel = features(wave)[None].to(args.device)
        probs = detect_language(model, mel, tok)[0]
        detected += max(probs, key=probs.get) == args.lang
        ids = greedy(model, mel, tok, args.lang)[0] if args.beam == 1 else beam_search(model, mel, tok, args.lang, args.beam)
        refs.append(ref)
        hyps.append(tok.decode(ids))
    print(f"{args.model} on FLEURS {args.lang} test, first {len(clips)} clips, beam {args.beam}, "
          f"{time.time() - t0:.0f}s on {args.device}")
    print(f"  WER {error_rate(refs, hyps):.3f}  CER {error_rate(refs, hyps, 'char'):.3f}  "
          f"language identified correctly: {detected}/{len(clips)}")
    for r, h in list(zip(refs, hyps))[:3]:
        print(f"  ref: {r[:100]}\n  hyp: {h[:100]}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
