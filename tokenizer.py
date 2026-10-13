"""Two tokenizers with one interface.

OpenAITokenizer   Whisper's own multilingual byte-level BPE (51,865 tokens), needed for OpenAI's weights.
IndicTokenizer    a small BPE learned on our transcripts, for training from scratch.

Why the second one: Whisper's vocabulary was built mostly from English, so Devanagari falls apart into
bytes. " नमस्ते दुनिया" (13 characters) is 15 Whisper tokens. A small model trained on a few hours of audio
cannot afford a 51k-row embedding that it will mostly never see, nor sequences longer than the text.

Both put the paper's multitask format (section 2.3) around the text:

    <|startoftranscript|> <|hi|> <|transcribe|> <|notimestamps|>  text tokens ...  <|endoftext|>
"""

from pathlib import Path

from bpe import BPE

LANGS = ["en", "hi", "mr"]


class OpenAITokenizer:
    def __init__(self, name: str = "openai/whisper-tiny") -> None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))
        tid = self.tok.token_to_id
        self.eot, self.sot = tid("<|endoftext|>"), tid("<|startoftranscript|>")
        self.translate, self.transcribe = tid("<|translate|>"), tid("<|transcribe|>")
        self.notimestamps = tid("<|notimestamps|>")
        self.languages = {}  # every <|xx|> between <|startoftranscript|> and <|translate|>
        for i in range(self.sot + 1, self.translate):
            self.languages[self.tok.id_to_token(i)[2:-2]] = i
        self.vocab_size = self.tok.get_vocab_size()
        self.n_text = self.eot  # ids below this are text

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(" " + text.strip(), add_special_tokens=False).ids  # Whisper text starts with a space

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i < self.eot]).strip()


class IndicTokenizer:
    def __init__(self, bpe: BPE) -> None:
        self.bpe = bpe
        base = bpe.vocab_size
        names = ["endoftext", "startoftranscript", *LANGS, "translate", "transcribe", "notimestamps"]
        ids = {n: base + i for i, n in enumerate(names)}
        self.eot, self.sot = ids["endoftext"], ids["startoftranscript"]
        self.translate, self.transcribe, self.notimestamps = ids["translate"], ids["transcribe"], ids["notimestamps"]
        self.languages = {lang: ids[lang] for lang in LANGS}
        self.vocab_size = base + len(names)
        self.n_text = bpe.pad  # ids below this are text (the BPE's own pad/bos/eos are never used)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 4000) -> "IndicTokenizer":
        return cls(BPE.train("\n".join(texts), vocab_size))

    def encode(self, text: str) -> list[int]:
        return self.bpe.encode(" " + text.strip())

    def decode(self, ids: list[int]) -> str:
        return self.bpe.decode([i for i in ids if i < self.bpe.pad]).strip()

    def save(self, path: Path) -> None:
        self.bpe.save(path)

    @classmethod
    def load(cls, path: Path) -> "IndicTokenizer":
        return cls(BPE.load(path))


def prefix(tok, lang: str, task: str = "transcribe") -> list[int]:
    return [tok.sot, tok.languages[lang], tok.transcribe if task == "transcribe" else tok.translate, tok.notimestamps]


def get_tokenizer(spec: str):
    """'openai/whisper-tiny' (or any openai/whisper-*) -> OpenAITokenizer; a path to bpe.json -> IndicTokenizer."""
    if spec.startswith("openai/"):
        return OpenAITokenizer(spec)
    return IndicTokenizer.load(Path(__file__).parent / spec)  # relative to the repo, e.g. data/bpe.json
