"""Train Whisper from scratch, or fine-tune OpenAI's weights, on FLEURS Hindi / Marathi / English.

    python train.py --init scratch --name tiny-indic                    # paper's tiny shape, our Indic tokenizer
    python train.py --init openai/whisper-tiny --langs hi --name tiny-hi --lr 1e-5 --steps 4000   # fine-tune

Optimization as in section 2.4 / Appendix F: AdamW, linear warmup then linear decay to zero, gradient-norm
clipping at 1.0, no data augmentation. Targets are the multitask format; the loss covers the language token,
the text and <|endoftext|> (task and timestamp tokens are fixed here, so there is nothing to learn).
"""

import argparse
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import expand, open_split
from decode import greedy
from model import SIZES, Whisper, load_pretrained, pick_device, save
from text import error_rate
from tokenizer import get_tokenizer, prefix

RUNS = Path(__file__).parent / "runs"


def batch_features(mel: np.ndarray, index: list[dict], ids: list[int], device: str) -> torch.Tensor:
    return torch.stack([expand(mel[index[i]["start"] : index[i]["start"] + index[i]["frames"]], index[i]["floor"])
                        for i in ids]).to(device)


def batch_tokens(index: list[dict], ids: list[int], tok, n_ctx: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder input (teacher forcing) and targets. Target -100 = no loss (padding, task, notimestamps)."""
    seqs = [(prefix(tok, index[i]["lang"]) + tok.encode(index[i]["text"]))[: n_ctx - 1] + [tok.eot] for i in ids]
    T = max(map(len, seqs)) - 1
    inp = torch.full((len(seqs), T), tok.eot)
    tgt = torch.full((len(seqs), T), -100)
    for b, s in enumerate(seqs):
        inp[b, : len(s) - 1] = torch.tensor(s[:-1])
        tgt[b, : len(s) - 1] = torch.tensor(s[1:])
        tgt[b, 1:3] = -100  # positions predicting <|transcribe|> and <|notimestamps|>
    return inp.to(device), tgt.to(device)


@torch.no_grad()
def evaluate(model: Whisper, tok, mel, index, ids, device: str, batch: int = 16) -> dict[str, tuple[float, float]]:
    was = model.training
    model.eval()
    refs, hyps = {}, {}
    by_lang: dict[str, list[int]] = {}
    for i in ids:
        by_lang.setdefault(index[i]["lang"], []).append(i)
    for lang, group in by_lang.items():
        for k in range(0, len(group), batch):
            chunk = group[k : k + batch]
            out = greedy(model, batch_features(mel, index, chunk, device), tok, lang)
            refs.setdefault(lang, []).extend(index[i]["text"] for i in chunk)
            hyps.setdefault(lang, []).extend(tok.decode(o) for o in out)
    model.train(was)
    return {lang: (error_rate(refs[lang], hyps[lang]), error_rate(refs[lang], hyps[lang], "char")) for lang in refs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="scratch", help="scratch, or openai/whisper-{tiny,base,small} to fine-tune")
    ap.add_argument("--size", default="tiny", choices=list(SIZES), help="shape when training from scratch")
    ap.add_argument("--name", default="tiny-indic")
    ap.add_argument("--langs", nargs="+", default=["hi", "mr", "en"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1.5e-3, help="paper: 1.5e-3 for tiny from scratch; ~1e-5 to fine-tune")
    ap.add_argument("--warmup", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1, help="from scratch on a few hours of audio; 0 to fine-tune")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-clips", type=int, default=100, help="dev clips per language")
    ap.add_argument("--device", default=pick_device())
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    RUNS.mkdir(exist_ok=True)
    if args.init == "scratch":
        tok_spec = "data/bpe.json"
        tok = get_tokenizer(tok_spec)
        model = Whisper(replace(SIZES[args.size], n_vocab=tok.vocab_size, dropout=args.dropout))
    else:
        tok_spec = args.init
        tok = get_tokenizer(tok_spec)
        model = load_pretrained(args.init)
        model.dims.dropout = 0.0
    model.to(args.device).train()

    train_mel, train_index = open_split("train")
    dev_mel, dev_index = open_split("dev")
    train_ids = [i for i, e in enumerate(train_index) if e["lang"] in args.langs]
    dev_ids = []
    for lang in args.langs:
        dev_ids += [i for i, e in enumerate(dev_index) if e["lang"] == lang][: args.eval_clips]

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, max(0.0, (args.steps - s) / max(1, args.steps - args.warmup))))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args}\n{n_params / 1e6:.1f}M parameters, vocabulary {tok.vocab_size}, {len(train_ids)} training clips", flush=True)

    order: list[int] = []
    best, t0, running = float("inf"), time.time(), 0.0
    for step in range(1, args.steps + 1):
        for _ in range(args.accum):
            if len(order) < args.batch:
                order = train_ids[:]
                random.shuffle(order)
            ids, order = order[: args.batch], order[args.batch :]
            mel = batch_features(train_mel, train_index, ids, args.device)
            inp, tgt = batch_tokens(train_index, ids, tok, model.dims.n_text_ctx, args.device)
            logits = model(mel, inp)
            loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-100)
            (loss / args.accum).backward()
            running += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"step {step:6d}  loss {running / 50:.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            running = 0.0
            if args.device == "mps":
                torch.mps.empty_cache()
        if step % args.eval_every == 0 or step == args.steps:
            scores = evaluate(model, tok, dev_mel, dev_index, dev_ids, args.device)
            wer = sum(s[0] for s in scores.values()) / len(scores)
            print(f"eval {step:6d}  dev WER {wer:.3f}  " + "  ".join(
                f"{lang} WER {w:.3f} CER {c:.3f}" for lang, (w, c) in sorted(scores.items())), flush=True)
            if wer < best:
                best = wer
                save(model, RUNS / f"{args.name}.pt", tokenizer=tok_spec, step=step, dev_wer=wer)


if __name__ == "__main__":
    main()
