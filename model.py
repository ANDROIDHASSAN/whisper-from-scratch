"""Whisper (Radford et al. 2022, arXiv 2212.04356), section 2.2: an encoder-decoder transformer.

    log-mel (80, 3000)
      -> conv1d(k3) GELU -> conv1d(k3, stride 2) GELU        (1500, d)   20 ms per position
      -> + sinusoidal positions -> N x pre-norm block -> LayerNorm         audio features
    tokens [sot, language, task, notimestamps, text...]
      -> embedding + learned positions -> N x [masked self-attn, cross-attn over audio, MLP] -> LayerNorm
      -> @ embedding^T (tied)                                              next-token logits

Module and parameter names follow the Hugging Face checkpoint layout, so `load_pretrained` can read OpenAI's
released weights straight into this code (verify_openai.py) and fine-tune them.
"""

import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Dims:
    n_mels: int = 80
    n_vocab: int = 51865
    n_audio_ctx: int = 1500
    n_text_ctx: int = 448
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4  # encoder and decoder each
    dropout: float = 0.0  # Whisper trained without dropout; we allow some for small-data training from scratch


SIZES = {  # the paper's Table 1
    "tiny": Dims(d_model=384, n_heads=6, n_layers=4),
    "base": Dims(d_model=512, n_heads=8, n_layers=6),
    "small": Dims(d_model=768, n_heads=12, n_layers=12),
}


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
    """Whisper's encoder positions: sines in the first half of the channels, cosines in the second."""
    inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-inc * torch.arange(channels // 2))
    t = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([t.sin(), t.cos()], dim=1)


class Attention(nn.Module):
    """Multi-head attention. The key projection has no bias (as in the released model: a bias on k would add
    the same amount to every score of a query and cancel in the softmax)."""

    def __init__(self, d: int, h: int, dropout: float) -> None:
        super().__init__()
        self.h = h
        self.q_proj, self.v_proj, self.out_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, source: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                cache: dict | None = None) -> torch.Tensor:
        """x: (B, T, d) queries. source: (B, S, d) for cross-attention, None for self-attention.
        cache: KV cache for incremental decoding. Self-attention appends the new keys/values; cross-attention
        computes them from the audio once and reuses them for every token."""
        q = self.q_proj(x)
        if source is not None:  # cross-attention
            if cache is not None and self in cache:
                k, v = cache[self]
            else:
                k, v = self.k_proj(source), self.v_proj(source)
                if cache is not None:
                    cache[self] = (k, v)
        else:
            k, v = self.k_proj(x), self.v_proj(x)
            if cache is not None:
                if self in cache:
                    k, v = torch.cat([cache[self][0], k], 1), torch.cat([cache[self][1], v], 1)
                cache[self] = (k, v)
        B, T, d = q.shape
        split = lambda t: t.view(B, t.shape[1], self.h, d // self.h).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.h)
        if mask is not None:
            scores = scores + mask
        y = self.drop(scores.softmax(-1)) @ v
        return self.out_proj(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d: int, h: int, dropout: float, cross: bool) -> None:
        super().__init__()
        self.self_attn, self.self_attn_layer_norm = Attention(d, h, dropout), nn.LayerNorm(d)
        self.encoder_attn = Attention(d, h, dropout) if cross else None
        self.encoder_attn_layer_norm = nn.LayerNorm(d) if cross else None
        self.fc1, self.fc2, self.final_layer_norm = nn.Linear(d, 4 * d), nn.Linear(4 * d, d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, audio=None, mask=None, cache=None):
        x = x + self.drop(self.self_attn(self.self_attn_layer_norm(x), mask=mask, cache=cache))
        if self.encoder_attn is not None:
            x = x + self.drop(self.encoder_attn(self.encoder_attn_layer_norm(x), audio, cache=cache))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.final_layer_norm(x)))))


class AudioEncoder(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(dims.n_mels, dims.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(dims.d_model, dims.d_model, 3, stride=2, padding=1)
        self.register_buffer("embed_positions", sinusoids(dims.n_audio_ctx, dims.d_model))
        self.layers = nn.ModuleList(Block(dims.d_model, dims.n_heads, dims.dropout, cross=False) for _ in range(dims.n_layers))
        self.layer_norm = nn.LayerNorm(dims.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # (B, 80, 3000) -> (B, 1500, d)
        x = F.gelu(self.conv2(F.gelu(self.conv1(mel)))).transpose(1, 2)
        x = x + self.embed_positions[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class TextDecoder(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(dims.n_vocab, dims.d_model)
        self.embed_positions = nn.Parameter(torch.empty(dims.n_text_ctx, dims.d_model).normal_(std=0.01))
        self.layers = nn.ModuleList(Block(dims.d_model, dims.n_heads, dims.dropout, cross=True) for _ in range(dims.n_layers))
        self.layer_norm = nn.LayerNorm(dims.d_model)

    def forward(self, tokens: torch.Tensor, audio: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """tokens: (B, T) new tokens; with a cache, the earlier ones are already in it."""
        offset = next(iter(cache.values()))[0].shape[1] if cache else 0  # tokens already decoded
        T = tokens.shape[1]
        x = self.embed_tokens(tokens) + self.embed_positions[offset : offset + T]
        # Causal mask over (new tokens) x (cached + new tokens): token i sees everything up to offset + i.
        mask = torch.full((T, offset + T), float("-inf"), device=tokens.device).triu(offset + 1)
        for layer in self.layers:
            x = layer(x, audio, mask, cache)
        return self.layer_norm(x) @ self.embed_tokens.weight.T


class Whisper(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(dims)
        self.decoder = TextDecoder(dims)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens, self.encoder(mel))


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """The safetensors format: 8-byte little-endian header length, a JSON header (name -> dtype, shape,
    byte offsets), then the raw tensor bytes."""
    raw = bytearray(Path(path).read_bytes())
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = dtypes[info["dtype"]]
        count = (end - start) // torch.tensor([], dtype=dtype).element_size()
        out[name] = torch.frombuffer(raw, dtype=dtype, count=count, offset=8 + n + start).reshape(info["shape"]).clone()
    return out


def load_pretrained(name: str = "openai/whisper-tiny") -> Whisper:
    """OpenAI's released weights (Hugging Face layout) loaded into this implementation."""
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(name, "config.json")).read_text())
    dims = Dims(n_mels=cfg["num_mel_bins"], n_vocab=cfg["vocab_size"], n_audio_ctx=cfg["max_source_positions"],
                n_text_ctx=cfg["max_target_positions"], d_model=cfg["d_model"], n_heads=cfg["encoder_attention_heads"],
                n_layers=cfg["encoder_layers"])
    weights = read_safetensors(hf_hub_download(name, "model.safetensors"))
    state = {k.removeprefix("model."): v.float() for k, v in weights.items() if k.startswith("model.")}
    state["encoder.embed_positions"] = state.pop("encoder.embed_positions.weight")
    state["decoder.embed_positions"] = state.pop("decoder.embed_positions.weight")
    model = Whisper(dims)
    model.load_state_dict(state)  # strict: every tensor must match by name and shape
    return model


def save(model: Whisper, path, **extra) -> None:
    torch.save({"dims": asdict(model.dims), "model": model.state_dict(), **extra}, path)


def load(path, device: str = "cpu") -> tuple[Whisper, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Whisper(Dims(**ckpt["dims"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
