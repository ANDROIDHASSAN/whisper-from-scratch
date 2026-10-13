"""Test-set WER / CER per language, or transcribe one file (with language identification).

    python evaluate.py runs/tiny-hi.pt --langs hi              # greedy
    python evaluate.py runs/tiny-hi.pt --langs hi --beam 5
    python evaluate.py runs/tiny-hi.pt --wav clip.wav
"""

import argparse
import time

import torch

from audio import SR, features, read_wav
from data import open_split
from decode import beam_search, detect_language, greedy
from model import load
from text import error_rate
from tokenizer import get_tokenizer
from train import batch_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--langs", nargs="+", default=["hi", "mr", "en"])
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--wav")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, ckpt = load(args.ckpt, args.device)
    tok = get_tokenizer(ckpt["tokenizer"])
    with torch.no_grad():
        if args.wav:
            wave, rate = read_wav(args.wav)
            assert rate == SR, f"expected {SR} Hz audio, got {rate}"
            mel = features(wave)[None].to(args.device)
            probs = detect_language(model, mel, tok)[0]
            lang = max(probs, key=probs.get)
            ids = greedy(model, mel, tok, lang)[0] if args.beam == 1 else beam_search(model, mel, tok, lang, args.beam)
            print(f"[{lang} {probs[lang]:.2f}] {tok.decode(ids)}")
            return
        mel, index = open_split("test")
        t0 = time.time()
        for lang in args.langs:
            ids = [i for i, e in enumerate(index) if e["lang"] == lang]
            refs, hyps = [], []
            for k in range(0, len(ids), 16 if args.beam == 1 else 1):
                chunk = ids[k : k + (16 if args.beam == 1 else 1)]
                x = batch_features(mel, index, chunk, args.device)
                out = greedy(model, x, tok, lang) if args.beam == 1 else [beam_search(model, x, tok, lang, args.beam)]
                refs += [index[i]["text"] for i in chunk]
                hyps += [tok.decode(o) for o in out]
            print(f"{lang}: WER {error_rate(refs, hyps):.3f}  CER {error_rate(refs, hyps, 'char'):.3f}  "
                  f"({len(refs)} clips, beam {args.beam})\n  ref: {refs[0][:90]}\n  hyp: {hyps[0][:90]}", flush=True)
        print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
