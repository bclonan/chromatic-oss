"""Chromatic symbols, numerical tokenization, and compact residual sequences."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..codecs import (HierarchicalCodec, LearnedChromaCodebook, LearnedCodec, RGBDCodec,
                      ScalarCodec, codec_from_config)
from ..codecs.base import INTEGER_DTYPES, id_dtype, validate_codes


class ChromaCodebook(LearnedChromaCodebook):
    """A numerical centroid table paired with unique RGB vocabulary symbols."""


class ChromaVocabulary:
    """Canonical RGB symbols with efficient integer IDs.

    An unfitted vocabulary is a deterministic uniform scalar quantizer.
    `fit` changes its centroid table through seeded Lloyd updates.
    """

    def __init__(self, size: int = 256, minimum: float = -1, maximum: float = 1,
                 learned: bool = False, codebook: LearnedChromaCodebook | None = None):
        self.codebook = codebook if codebook is not None else ChromaCodebook(size, minimum, maximum)
        self.size = self.codebook.size
        self.minimum, self.maximum = self.codebook.minimum, self.codebook.maximum
        self.learned = bool(learned)
        self.codec = LearnedCodec(codebook=self.codebook)

    def __len__(self) -> int:
        return self.size

    @property
    def palette(self) -> torch.Tensor:
        return self.codebook.palette

    @property
    def values(self) -> torch.Tensor:
        return self.codebook.values.detach()

    @property
    def bits_per_token(self) -> int:
        return math.ceil(math.log2(self.size))

    def symbol(self, token_id: int) -> str:
        if not 0 <= token_id < self.size:
            raise IndexError("Token ID is absent from this vocabulary")
        r, g, b = self.palette[token_id].tolist()
        return f"COLOR_{r:02X}{g:02X}{b:02X}"

    def token_id(self, symbol: str) -> int:
        if not symbol.startswith("COLOR_") or len(symbol) != 12:
            raise ValueError("Chromatic symbols use COLOR_RRGGBB")
        try:
            packed = int(symbol[6:], 16)
        except ValueError as error:
            raise ValueError("Chromatic symbol contains invalid hexadecimal RGB") from error
        color = torch.tensor([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], dtype=torch.uint8)
        return int(self.codebook.ids_from_colors(color))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(x)[..., 0]

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.codebook.decode(ids)

    def colors(self, ids: torch.Tensor) -> torch.Tensor:
        return self.codebook.colors(ids)

    def fit(self, samples: torch.Tensor, iterations: int = 20, seed: int = 0) -> "ChromaVocabulary":
        self.codebook.fit(samples, iterations=iterations, seed=seed)
        self.learned = True
        return self

    def to_config(self) -> dict:
        return {"learned": self.learned, "codebook": self.codebook.to_config()}

    @classmethod
    def from_config(cls, config: dict) -> "ChromaVocabulary":
        return cls(learned=config["learned"], codebook=LearnedChromaCodebook.from_config(config["codebook"]))


@dataclass
class TokenSequence:
    """RGBD byte rows or explicit palette IDs, with numerical shape metadata.

    RGBD stores one [R, G, B, D] row per value. D is the FP32 exponent.
    Explicit palette and residual modes retain their compact integer IDs.
    No original numerical tensor is retained.
    """

    tokens: torch.Tensor
    shape: tuple[int, ...]
    codec_config: dict
    lengths: torch.Tensor | None = None
    decoded_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.shape = tuple(self.shape)
        if any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError("Token shape dimensions must be nonnegative integers")
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError("TokenSequence requires torch.Tensor tokens")
        if not isinstance(self.decoded_dtype, torch.dtype) or self.decoded_dtype not in (
                torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("TokenSequence requires a supported floating decoded dtype")
        codec = codec_from_config(self.codec_config)
        count = math.prod(self.shape)
        if isinstance(codec, RGBDCodec):
            if self.tokens.dtype != torch.uint8:
                raise TypeError("RGBD tokens must contain uint8 color bytes")
            if self.tokens.ndim != 2 or tuple(self.tokens.shape) != (count, 4):
                raise ValueError("RGBD tokens require one [R, G, B, D] row per value")
            if self.lengths is not None:
                raise ValueError("RGBD tokens have no residual lengths")
            codec.validate(self.tokens)
            return
        if codec.channels != 1 and not isinstance(codec, HierarchicalCodec):
            raise ValueError("TokenSequence requires RGBD bytes or a scalar/residual ID codec")
        if self.tokens.ndim != 1 or self.tokens.dtype not in INTEGER_DTYPES:
            raise TypeError("TokenSequence tokens must be a one-dimensional integer tensor")
        if self.lengths is None:
            if self.tokens.numel() != count:
                raise ValueError("Token count differs from the value shape")
        else:
            if self.lengths.dtype not in INTEGER_DTYPES or tuple(self.lengths.shape) != self.shape:
                raise ValueError("Residual lengths must be integer values with the numerical tensor shape")
            if self.lengths.device != self.tokens.device or (self.lengths.long() < 1).any():
                raise ValueError("Residual lengths must be positive and share the token device")
            if int(self.lengths.long().sum()) != self.tokens.numel():
                raise ValueError("Residual lengths do not account for every token")

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[0]

    @property
    def rgbd(self) -> torch.Tensor:
        if self.codec_config.get("name") != "rgbd":
            raise ValueError("This sequence stores palette IDs, not RGBD byte tokens")
        return self.tokens.reshape((*self.shape, 4))

    @property
    def rgb(self) -> torch.Tensor:
        """RGB display view. Decoding still needs all four RGBD channels."""
        return self.rgbd[..., :3]

    @property
    def tokens_per_value(self) -> float:
        count = math.prod(self.shape)
        return self.num_tokens / count if count else 0.0

    @property
    def storage_nbytes(self) -> int:
        result = self.tokens.numel() * self.tokens.element_size()
        if self.lengths is not None:
            result += self.lengths.numel() * self.lengths.element_size()
        return result

    def to(self, device) -> "TokenSequence":
        return TokenSequence(self.tokens.to(device), self.shape, self.codec_config,
                             None if self.lengths is None else self.lengths.to(device), self.decoded_dtype)

    def save(self, path, *, compress: bool = False) -> int:
        """Save actual RGBD bytes through the checksummed ctensor format."""
        from ..tensor import ChromaTensor
        from ..serialization import save_ctensor
        self.__post_init__()
        return save_ctensor(ChromaTensor(self.rgbd, RGBDCodec(), dtype=self.decoded_dtype),
                            path, compress=compress)

    @classmethod
    def load(cls, path, **budgets) -> "TokenSequence":
        """Load saved RGBD bytes without an encoder or the original values."""
        from ..serialization import load_ctensor
        tensor = load_ctensor(path, **budgets)
        if not isinstance(tensor.codec, RGBDCodec):
            raise ValueError("TokenSequence.load requires an RGBD ctensor file")
        return cls(tensor.codes.reshape(-1, 4), tuple(tensor.shape), tensor.codec.to_config(),
                   decoded_dtype=tensor.decoded_dtype)


class ChromaTokenizer:
    """Default to real RGBD byte tokens. Explicit vocabularies quantize to IDs."""

    def __init__(self, vocabulary: ChromaVocabulary | None = None, vocabulary_size: int | None = None,
                 minimum: float | None = None, maximum: float | None = None, levels: int = 1,
                 tolerance: float | None = None, *, codec: RGBDCodec | None = None):
        if type(levels) is not int or levels < 1:
            raise ValueError("levels must be a positive integer")
        bounded = (vocabulary is not None or vocabulary_size is not None or minimum is not None
                   or maximum is not None or levels != 1 or tolerance is not None)
        if codec is not None and (not isinstance(codec, RGBDCodec) or bounded):
            raise ValueError("An explicit RGBD codec cannot be combined with vocabulary or range settings")
        if not bounded:
            self.vocabulary = None
            self.codec: ScalarCodec = codec if codec is not None else RGBDCodec()
            return
        self.vocabulary = vocabulary if vocabulary is not None else ChromaVocabulary(
            256 if vocabulary_size is None else vocabulary_size,
            -1 if minimum is None else minimum, 1 if maximum is None else maximum)
        if levels == 1:
            self.codec = self.vocabulary.codec
        else:
            if self.vocabulary.learned:
                raise ValueError("Residual tokenizer uses a deterministic bounded vocabulary")
            self.codec = HierarchicalCodec(levels, self.vocabulary.size, self.vocabulary.minimum,
                                           self.vocabulary.maximum, tolerance)

    def encode(self, x: torch.Tensor) -> TokenSequence:
        x = torch.as_tensor(x)
        codes = self.codec.encode(x)
        dtype = x.dtype if x.is_floating_point() else torch.float32
        if isinstance(self.codec, RGBDCodec):
            return TokenSequence(codes.reshape(-1, 4), tuple(x.shape), self.codec.to_config(),
                                 decoded_dtype=dtype)
        if isinstance(self.codec, HierarchicalCodec):
            lengths = codes[..., -1].clone()
            slots = codes[..., :-1].reshape(-1, self.codec.levels)
            active = torch.arange(self.codec.levels, device=x.device)[None, :] < lengths.reshape(-1, 1)
            tokens = slots[active]
            return TokenSequence(tokens, tuple(x.shape), self.codec.to_config(), lengths, dtype)
        return TokenSequence(codes[..., 0].reshape(-1), tuple(x.shape), self.codec.to_config(), decoded_dtype=dtype)

    def tokenize(self, x: torch.Tensor) -> TokenSequence:
        return self.encode(x)

    def decode(self, sequence: TokenSequence | torch.Tensor) -> torch.Tensor:
        return ChromaDecoder(self.vocabulary).decode(sequence)


class ChromaDecoder:
    """Reconstruct numerical tensors from tokens and their saved codec metadata."""

    def __init__(self, vocabulary: ChromaVocabulary | None = None):
        self.vocabulary = vocabulary

    def decode(self, sequence: TokenSequence | torch.Tensor) -> torch.Tensor:
        if isinstance(sequence, torch.Tensor):
            if self.vocabulary is None:
                if sequence.dtype == torch.uint8 and sequence.ndim >= 1 and sequence.shape[-1] == 4:
                    return RGBDCodec().decode(sequence)
                raise ValueError("Raw integer tokens require a vocabulary")
            return self.vocabulary.decode(sequence)
        codec = codec_from_config(sequence.codec_config)
        sequence.__post_init__()
        if isinstance(codec, RGBDCodec):
            return codec.decode(sequence.rgbd).to(dtype=sequence.decoded_dtype)
        if isinstance(codec, HierarchicalCodec):
            if sequence.lengths is None or (sequence.lengths.long() > codec.levels).any():
                raise ValueError("Residual token sequence has invalid precision lengths")
            rows = math.prod(sequence.shape)
            codes = torch.zeros((rows, codec.channels), dtype=torch.uint8, device=sequence.tokens.device)
            active = torch.arange(codec.levels, device=sequence.tokens.device)[None, :] < sequence.lengths.reshape(-1, 1)
            validate_codes(sequence.tokens.unsqueeze(-1), 1, codec.vocabulary_size - 1)
            codes[:, :-1][active] = sequence.tokens.to(torch.uint8)
            codes[:, -1] = sequence.lengths.reshape(-1).to(torch.uint8)
            result = codec.decode(codes.reshape(*sequence.shape, codec.channels))
        else:
            if sequence.lengths is not None:
                raise ValueError("A scalar vocabulary has no residual lengths")
            result = codec.decode(sequence.tokens.reshape(*sequence.shape, 1))
        return result.to(dtype=sequence.decoded_dtype)

    def __call__(self, sequence: TokenSequence | torch.Tensor) -> torch.Tensor:
        return self.decode(sequence)


__all__ = ["ChromaCodebook", "ChromaVocabulary", "ChromaTokenizer", "ChromaDecoder", "TokenSequence"]
