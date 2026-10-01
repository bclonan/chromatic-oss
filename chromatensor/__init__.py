"""Native hue tensors, with explicit legacy numerical comparison APIs."""
from importlib import import_module

from .hue import HueTensor, ColorTensor, Tensor, HueEncoder, HueDecoder, PHASE_MODULUS, native_info

_LEGACY = {
    "ChromaTensor": ".tensor", "ChromaCube": ".cube", "HolographicCube": ".holographic",
    "LearnedNeuralDecoder": ".decoders",
    **{name: ".codecs" for name in ["RGBCodec", "RGBDCodec", "HSVCodec", "HueCodec", "UniformCodec",
        "LearnedCodec", "HierarchicalCodec", "LearnedRGBCodec", "LearnedRGBDCodec",
        "LearnedChromaEncoder", "LearnedChromaDecoder", "LearnedChromaCodebook"]},
    **{name: ".tokenizer" for name in ["ChromaTokenizer", "ChromaDecoder", "ChromaVocabulary",
        "ChromaCodebook", "TokenSequence"]},
    **{name: ".serialization" for name in ["save_ctensor", "load_ctensor",
        "save_chroma_checkpoint", "load_chroma_checkpoint"]},
}


def __getattr__(name):
    if name not in _LEGACY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LEGACY[name], __name__), name)
    globals()[name] = value
    return value


__all__ = ["HueTensor", "ColorTensor", "Tensor", "HueEncoder", "HueDecoder", "PHASE_MODULUS", "native_info", *_LEGACY]
