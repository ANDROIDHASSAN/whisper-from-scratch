"""Checks that fail if a core piece is wrong. CPU, under a minute (the first needs the internet once).

    python check.py
"""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audio import features, mel_filterbank
from data import compact, expand
from decode import beam_search, greedy
from model import Dims, Whisper, read_safetensors, sinusoids
from tokenizer import IndicTokenizer, prefix

torch.manual_seed(0)
TINY = Dims(n_mels=80, n_vocab=0, n_audio_ctx=1500, n_text_ctx=64, d_model=64, n_heads=4, n_layers=2)


def tiny_tok() -> IndicTokenizer:
    return IndicTokenizer.train(["नमस्ते दुनिया hello world मी मराठी बोलतो"] * 20, 300)


def tiny_model(tok, **kw) -> Whisper:
    return Whisper(Dims(**{**TINY.__dict__, "n_vocab": tok.vocab_size, **kw})).eval()


def check_mel_filterbank_matches_openai() -> None:
    from huggingface_hub import hf_hub_download

    ref = json.loads(Path(hf_hub_download("openai/whisper-tiny", "preprocessor_config.json")).read_text())["mel_filters"]
    assert torch.allclose(mel_filterbank(), torch.tensor(ref), atol=1e-6)


def check_stored_features_rebuild_exactly() -> None:
    """data.py stores only the frames that touch audio; the rebuilt 30 s window must equal the real one."""
    for seconds in (1.3, 7.0, 29.99):
        wave = np.random.default_rng(0).standard_normal(int(seconds * 16000)).astype(np.float32) * 0.1
        full = features(wave)
        frames, floor = compact(full, len(wave))
        want = full.clone()
        want[:, : len(frames)] = full[:, : len(frames)].half().float()  # stored frames are float16
        assert torch.equal(expand(frames, floor), want), seconds  # the padding is exactly one value per clip


def check_encoder_positions() -> None:
    pe = sinusoids(1500, 384)
    assert pe.shape == (1500, 384) and torch.allclose(pe[0, :192], torch.zeros(192)) and torch.allclose(pe[0, 192:], torch.ones(192))


def check_decoder_is_causal() -> None:
    tok = tiny_tok()
    m = tiny_model(tok)
    mel, tokens = torch.randn(1, 80, 200), torch.randint(0, tok.n_text, (1, 10))
    a = m(mel, tokens)
    tokens[0, 6] = (tokens[0, 6] + 1) % tok.n_text
    b = m(mel, tokens)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5) and not torch.allclose(a[:, 6:], b[:, 6:])


def check_kv_cache_equals_full_pass() -> None:
    tok = tiny_tok()
    m = tiny_model(tok)
    mel, tokens = torch.randn(2, 80, 200), torch.randint(0, tok.n_text, (2, 9))
    audio = m.encoder(mel)
    full = m.decoder(tokens, audio)
    cache = {}
    steps = [m.decoder(tokens[:, :4], audio, cache)]  # a prefix at once, then one token at a time
    steps += [m.decoder(tokens[:, i : i + 1], audio, cache) for i in range(4, 9)]
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)


def check_beam_1_is_greedy() -> None:
    tok = tiny_tok()
    m = tiny_model(tok)
    for _ in range(3):
        mel = torch.randn(1, 80, 200)
        assert beam_search(m, mel, tok, "hi", beam=1, max_new=20) == greedy(m, mel, tok, "hi", max_new=20)[0]


def check_tokenizer() -> None:
    tok = tiny_tok()
    text = "नमस्ते bhai, hello दुनिया!"
    assert tok.decode(tok.encode(text)) == text
    p = prefix(tok, "mr")
    assert p == [tok.sot, tok.languages["mr"], tok.transcribe, tok.notimestamps] and min(p) >= tok.n_text


def check_safetensors_reader() -> None:
    tensors = {"a": torch.randn(3, 4), "b": torch.arange(5, dtype=torch.float32)}
    header, blob = {}, b""
    for name, t in tensors.items():
        raw = t.numpy().tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    head = json.dumps(header).encode()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.safetensors"
        path.write_bytes(struct.pack("<Q", len(head)) + head + blob)
        back = read_safetensors(path)
    assert all(torch.equal(back[k], v) for k, v in tensors.items())


def check_can_overfit() -> None:
    """The whole loop (teacher forcing, shift by one, prefix, decoding) learns 4 utterances and reads them back."""
    tok = tiny_tok()
    m = tiny_model(tok).train()
    mel = torch.randn(4, 80, 200)
    texts = ["नमस्ते दुनिया", "hello world", "मी मराठी बोलतो", "hello दुनिया"]
    seqs = [prefix(tok, "hi") + tok.encode(t) + [tok.eot] for t in texts]
    T = max(map(len, seqs)) - 1
    inp = torch.tensor([s[:-1] + [tok.eot] * (T - len(s) + 1) for s in seqs])
    tgt = torch.tensor([s[1:] + [-100] * (T - len(s) + 1) for s in seqs])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(300):
        loss = F.cross_entropy(m(mel, inp).flatten(0, 1), tgt.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    assert [tok.decode(o) for o in greedy(m, mel, tok, "hi")] == texts, f"loss {loss.item():.4f}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("check_"):
            fn()
            print(f"ok  {name[6:]}")
