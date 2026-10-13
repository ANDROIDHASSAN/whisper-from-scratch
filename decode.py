"""Decoding (section 4.5 of the paper): language identification, greedy and beam search, with a KV cache.

The decoder is autoregressive: one token per step. Without a cache, step n re-reads all n earlier tokens and
re-projects the 1500 audio positions in every cross-attention layer. With it, each step feeds one new token:
self-attention appends its key/value, cross-attention keys/values are computed once per utterance.
"""

import torch

from tokenizer import prefix


def text_mask(tok, vocab: int, device) -> torch.Tensor:
    """-inf for every special token except <|endoftext|>: no timestamps, no switching language mid-text."""
    mask = torch.zeros(vocab, device=device)
    mask[tok.n_text :] = float("-inf")
    mask[tok.eot] = 0.0
    return mask


@torch.no_grad()
def detect_language(model, mel: torch.Tensor, tok) -> list[dict[str, float]]:
    """Feed only <|startoftranscript|> and read the distribution over language tokens (section 2.3)."""
    audio = model.encoder(mel)
    logits = model.decoder(torch.full((mel.shape[0], 1), tok.sot, device=mel.device), audio)[:, -1]
    codes, ids = zip(*tok.languages.items())
    probs = logits[:, list(ids)].softmax(-1)
    return [dict(zip(codes, p.tolist())) for p in probs]


@torch.no_grad()
def greedy(model, mel: torch.Tensor, tok, lang: str, max_new: int = 224) -> list[list[int]]:
    """mel: (B, 80, 3000). Returns the text token ids of each utterance (prefix and <|endoftext|> removed)."""
    B, device = mel.shape[0], mel.device
    audio, cache = model.encoder(mel), {}
    mask = text_mask(tok, model.dims.n_vocab, device)
    logits = model.decoder(torch.tensor([prefix(tok, lang)] * B, device=device), audio, cache)[:, -1]
    out, done = [[] for _ in range(B)], torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new):
        nxt = (logits + mask).argmax(-1)
        for b in range(B):
            if not done[b] and nxt[b] != tok.eot:
                out[b].append(nxt[b].item())
        done |= nxt == tok.eot
        if done.all():
            break
        logits = model.decoder(nxt[:, None], audio, cache)[:, -1]
    return out


@torch.no_grad()
def beam_search(model, mel: torch.Tensor, tok, lang: str, beam: int = 5, max_new: int = 224) -> list[int]:
    """One utterance (mel: (1, 80, 3000)). Keeps the `beam` best partial transcripts by summed log-probability;
    finished ones are ranked by log-probability / length, Whisper's default."""
    device = mel.device
    audio, cache = model.encoder(mel), {}
    mask = text_mask(tok, model.dims.n_vocab, device)
    logits = model.decoder(torch.tensor([prefix(tok, lang)], device=device), audio, cache)[:, -1]
    seqs, scores = [[]], torch.zeros(1, device=device)
    finished: list[tuple[list[int], float]] = []
    for _ in range(max_new):
        total = (scores[:, None] + (logits + mask).log_softmax(-1)).view(-1)
        top_scores, top_ids = total.topk(2 * beam)  # 2x: enough left after some candidates end here
        keep, new_seqs, new_scores = [], [], []
        for s, i in zip(top_scores.tolist(), top_ids.tolist()):
            src, token = divmod(i, logits.shape[-1])
            if token == tok.eot:
                finished.append((seqs[src], s / max(len(seqs[src]), 1)))
            else:
                keep.append(src)
                new_seqs.append(seqs[src] + [token])
                new_scores.append(s)
            if len(new_seqs) == beam:
                break
        if len(finished) >= beam:
            break
        index = torch.tensor(keep, device=device)
        for key, (k, v) in cache.items():  # every hypothesis carries the cache of the one it extends
            cache[key] = (k.index_select(0, index), v.index_select(0, index))
        audio = audio[:1].expand(len(keep), -1, -1)  # the same audio for every hypothesis
        seqs, scores = new_seqs, torch.tensor(new_scores, device=device)
        logits = model.decoder(torch.tensor([[s[-1]] for s in seqs], device=device), audio, cache)[:, -1]
    if not finished:
        finished = [(s, sc / max(len(s), 1)) for s, sc in zip(seqs, scores.tolist())]
    return max(finished, key=lambda f: f[1])[0]
