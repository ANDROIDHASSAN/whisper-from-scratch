"""Whisper's audio front-end (Radford et al. 2022, section 2.2), reproduced exactly so OpenAI's weights work.

    16 kHz waveform, padded or cut to 30 s (480,000 samples)
      -> STFT: 25 ms Hann window, 10 ms hop                                 3000 frames
      -> power -> 80 mel bands (librosa's Slaney filterbank)
      -> log10, floor at (max - 8), then (x + 4) / 4                         roughly -1 .. 1

Every input is a 30 s window: short clips are padded with silence. The model never has to handle a length it
has not seen, at the cost of computing on padding.
"""

import math
import struct
from pathlib import Path

import numpy as np
import torch

SR = 16000
N_FFT = 400  # 25 ms
HOP = 160  # 10 ms
N_MELS = 80
CHUNK = 30 * SR  # samples in one 30 s window
FRAMES = CHUNK // HOP  # 3000 mel frames per window


def read_wav(src: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Parse a WAV file (16/32-bit PCM or 32-bit float). Returns (mono float32 in -1..1, sampling rate)."""
    raw = src if isinstance(src, bytes) else Path(src).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    fmt, data, pos = None, None, 12
    while pos + 8 <= len(raw):  # chunks: 4-byte name, 4-byte size, body
        name, size = raw[pos : pos + 4], struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + size]
        if name == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: real tag inside
                fmt = (struct.unpack("<H", body[24:26])[0], *fmt[1:])
        elif name == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("WAV file has no fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(data, dtype="<f4")
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data, dtype="<i2") / 32768.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data, dtype="<i4") / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample format: tag {tag}, {bits} bits")
    x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32), rate


def slaney_mel(hz: torch.Tensor) -> torch.Tensor:
    """librosa's default ("Slaney") mel scale: linear below 1 kHz (200/3 Hz per mel), logarithmic above."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return torch.where(hz < min_log_hz, hz / f_sp, min_log_mel + torch.log(hz.clamp(min=1e-10) / min_log_hz) / logstep)


def slaney_hz(mel: torch.Tensor) -> torch.Tensor:
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return torch.where(mel < min_log_mel, mel * f_sp, min_log_hz * torch.exp(logstep * (mel - min_log_mel)))


def mel_filterbank(n_mels: int = N_MELS) -> torch.Tensor:
    """(n_mels, 201): triangles evenly spaced on the Slaney mel scale, each scaled to unit area
    ("slaney" normalization, 2 / bandwidth), as librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)."""
    fft_hz = torch.linspace(0, SR / 2, N_FFT // 2 + 1, dtype=torch.float64)
    hz = slaney_hz(torch.linspace(0.0, float(slaney_mel(torch.tensor(SR / 2.0))), n_mels + 2, dtype=torch.float64))
    width = hz[1:] - hz[:-1]
    ramps = hz[:, None] - fft_hz[None, :]
    lower = -ramps[:-2] / width[:-1, None]
    upper = ramps[2:] / width[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
    return (weights * (2.0 / (hz[2:] - hz[:-2]))[:, None]).float()


def pad_or_trim(wave: torch.Tensor, length: int = CHUNK) -> torch.Tensor:
    return wave[:length] if len(wave) >= length else torch.nn.functional.pad(wave, (0, length - len(wave)))


def log_mel(wave: torch.Tensor | np.ndarray) -> torch.Tensor:
    """wave: (samples,) at 16 kHz, already padded to 30 s -> (80, 3000) normalized log-mel."""
    wave = torch.as_tensor(wave, dtype=torch.float32)
    stft = torch.stft(wave, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)  # centred, reflect-padded
    power = stft[:, :-1].abs() ** 2  # Whisper drops the last frame: 480,000 / 160 = 3000 exactly
    mel = mel_filterbank() @ power
    log = torch.clamp(mel, min=1e-10).log10()
    log = torch.maximum(log, log.max() - 8.0)  # 80 dB of dynamic range below the loudest bin
    return (log + 4.0) / 4.0


def features(wave: np.ndarray) -> torch.Tensor:
    """Waveform of any length up to 30 s -> (80, 3000), what the encoder reads."""
    return log_mel(pad_or_trim(torch.as_tensor(wave, dtype=torch.float32)))
