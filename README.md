# Whisper from scratch — and OpenAI's weights running on it

A from-scratch PyTorch reimplementation of **Whisper** ([Radford et al., 2022](https://arxiv.org/abs/2212.04356)):
the exact audio front-end, the encoder-decoder transformer, the multitask token format, language
identification, greedy and beam-search decoding with a KV cache. It trains from scratch on FLEURS Hindi,
Marathi and English with a small Indic tokenizer, and it can load **OpenAI's released checkpoints** and
fine-tune them.

Loading the real weights is the test that matters: if the mel filterbank, the normalization, any layer, the
attention scaling, the positions or the token format differed from the original in any detail, the
transcripts would be garbage.

## OpenAI's weights on this code

`python verify_openai.py` loads `openai/whisper-tiny` from its `model.safetensors` (read by a short parser in
[model.py](model.py), no extra library) and transcribes the first 50 FLEURS test clips, greedy decoding, on CPU:

| Language | WER | CER | Language identified correctly |
|---|---|---|---|
| English | **14.2%** | 6.4% | 50 / 50 |
| Hindi | 159.8% | 140.2% | 45 / 50 |

English works as a released model should. Hindi shows why fine-tuning matters: tiny writes Hindi speech as
romanized or English words (`कुछ अणुओं में अस्थिर केंद्रक` → "Kochi Arangu mein Aarastir Kendra"). That
number is the baseline the Hindi fine-tune is measured against.

The mel filterbank written here matches the one OpenAI ships (`preprocessor_config.json`) to within 5 × 10⁻⁸.

## Fine-tuning tiny on Hindi

`openai/whisper-tiny` fine-tuned on FLEURS Hindi train (2,115 clips): 4,000 steps, batch 4 × 2 accumulated,
lr 1e-5 with 400 warmup steps, no dropout, 83 min on an M1 Pro. Scored on the full FLEURS Hindi test set,
greedy decoding:

| Model | WER | CER |
|---|---|---|
| `openai/whisper-tiny` as released (418 clips) | 138.9% | 122.7% |
| fine-tuned here (417 clips) | **49.2%** | **21.2%** |

Dev WER (100 clips) by step: 58.9% at 500, 51.4% at 1,000, 48.6% at 1,500, then flat between 48% and 50% to
4,000: with 2k clips the model has learned what this data can teach it. It now writes Devanagari
(`कुछ अणुओं में अस्थिर केंद्रक` → `कुछ अडगों मे आज स्थिक केंद्रा`) where the released model wrote romanized
or English words.

> **Status:** architecture, front-end, tokenizers, decoding, data pipeline and the Hindi fine-tune are complete
> and measured. Training from scratch is next; results will be added here from the logs.

## Paper → code

| Paper | What | Code |
|---|---|---|
| §2.2 | 16 kHz, 30 s windows, 25 ms / 10 ms log-mel (80 bands), log10, (max − 8) floor, (x + 4) / 4 | [audio.py](audio.py) |
| §2.2 | Two conv1d + GELU stem (second with stride 2), sinusoidal encoder positions, pre-norm blocks | `AudioEncoder` in [model.py](model.py) |
| §2.2 | Decoder with learned positions, cross-attention, tied input/output embedding | `TextDecoder` |
| §2.3 | Multitask format: `<\|startoftranscript\|>` language task `<\|notimestamps\|>` text `<\|endoftext\|>` | `prefix` in [tokenizer.py](tokenizer.py) |
| §2.3 | Language identification from the token after `<\|startoftranscript\|>` | `detect_language` in [decode.py](decode.py) |
| §2.4, App. F | AdamW, linear warmup (2048) then linear decay to 0, grad-norm clip 1.0, tiny max lr 1.5e-3 | [train.py](train.py) |
| §4.5 | Greedy and beam-search decoding, length-normalized scores | `greedy`, `beam_search` |
| Table 1 | Model shapes tiny / base / small | `SIZES` |

## Tokenizers

Tokens needed for the FLEURS dev transcripts (not seen when learning the Indic BPE):

| | Vocabulary | `" नमस्ते दुनिया"` | Hindi dev | Marathi dev | English dev |
|---|---|---|---|---|---|
| OpenAI's multilingual BPE | 51,865 | 15 | 28,821 | 61,749 | 10,287 |
| Indic BPE, learned on FLEURS train transcripts | 4,008 | 5 | 9,446 (3.1× fewer) | 17,908 (3.4× fewer) | 17,374 (1.7× more) |

The trade-off is visible: 4k merges spent on three languages beat a 51k English-heavy vocabulary on Devanagari
and lose on English.

A from-scratch model on a few hours of audio cannot use a 51k-row embedding it mostly never sees, so
`train.py --init scratch` uses the small tokenizer with the same special tokens; fine-tuning keeps OpenAI's.

## Correctness checks

`python check.py` (CPU, ~10 s; the first check downloads one small file once):

| Check | What it proves |
|---|---|
| `mel_filterbank_matches_openai` | Our Slaney filterbank equals the one OpenAI ships |
| `stored_features_rebuild_exactly` | data.py stores only frames that touch audio; the rebuilt 30 s window equals the real one |
| `encoder_positions` | Sinusoid layout (sines then cosines) |
| `decoder_is_causal` | Changing token *j* changes no prediction before *j* |
| `kv_cache_equals_full_pass` | Incremental decoding (prefix, then one token at a time) gives the same logits as one full pass |
| `beam_1_is_greedy` | Beam search with one beam is greedy decoding |
| `tokenizer` | Lossless round trip of mixed Hindi-English text; correct multitask prefix |
| `safetensors_reader` | The parser reads back a hand-written safetensors file |
| `can_overfit` | A tiny model memorizes 4 utterances and greedy decoding reads them back |

Removing the causal mask or breaking the KV-cache position offset makes the matching check fail.

## Run it

```bash
pip install -r requirements.txt
python check.py
python verify_openai.py --lang en --clips 50          # OpenAI's weights on this code
python data.py                                        # FLEURS hi + mr + en -> features, Indic tokenizer
python train.py --init scratch --name tiny-indic      # from scratch, paper's tiny shape
python train.py --init openai/whisper-tiny --langs hi --name tiny-hi --lr 1e-5 --warmup 400 --dropout 0 --steps 4000 --batch 4 --accum 2
python evaluate.py runs/tiny-hi.pt --langs hi --beam 5
python evaluate.py runs/tiny-hi.pt --wav clip.wav     # language ID + transcript
```

## Differences from the paper

- Trained on FLEURS (~23 h, three languages) instead of 680,000 h of web audio, so from-scratch quality is
  bounded by data, not by the code.
- No timestamp tokens or long-form (sliding 30 s window) transcription; the `<|notimestamps|>` path only.
- Dropout is available for small-data training from scratch; Whisper used none.

## References

- Radford et al., 2022. *Robust Speech Recognition via Large-Scale Weak Supervision.* [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- Vaswani et al., 2017. *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Sennrich et al., 2016. *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Conneau et al., 2022. *FLEURS.* [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)
