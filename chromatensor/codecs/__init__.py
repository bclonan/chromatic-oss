"""Interchangeable scalar-to-chromatic codecs."""
from .base import ScalarCodec
from .rgb import RGBCodec, UniformCodec
from .rgbd import RGBDCodec
from .hsv import HSVCodec, HueCodec, hsv_to_rgb
from .learned import (LearnedCodec, LearnedRGBDCodec, LearnedChromaCodebook,
                      LearnedChromaEncoder, LearnedChromaDecoder)
from .hierarchical import HierarchicalCodec
from .neural_rgb import LearnedRGBCodec


def codec_from_config(config: dict) -> ScalarCodec:
    """Reconstruct a codec without pickle or executable metadata."""
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError("Codec configuration must contain a name")
    name = config["name"]
    arguments = {key: value for key, value in config.items() if key != "name"}
    classes = {"rgb": RGBCodec, "uniform": UniformCodec, "hsv": HSVCodec,
               "hue": HueCodec, "hierarchical": HierarchicalCodec}
    if name in classes:
        return classes[name](**arguments)
    if name == "rgbd":
        if set(arguments) != {"format"} or arguments.get("format") != "finite-ieee754-fp32-sign-fraction-exponent":
            raise ValueError("Unknown RGBD format")
        return RGBDCodec()
    if name == "learned":
        return LearnedCodec(codebook=LearnedChromaCodebook.from_config(arguments["codebook"]))
    if name == "learned_rgbd":
        book = LearnedChromaCodebook.from_config(arguments["codebook"])
        modifier = LearnedChromaCodebook.from_config(arguments["modifier"])
        return LearnedRGBDCodec(size=book.size, modifier_size=modifier.size, codebook=book, modifier=modifier)
    if name == "learned_rgb":
        return LearnedRGBCodec.from_config(config)
    raise ValueError(f"Unknown codec name: {name}")


__all__ = ["ScalarCodec", "RGBCodec", "RGBDCodec", "UniformCodec", "HSVCodec", "HueCodec",
           "LearnedCodec", "LearnedRGBDCodec", "LearnedChromaCodebook", "LearnedChromaEncoder",
           "LearnedChromaDecoder", "LearnedRGBCodec", "HierarchicalCodec", "codec_from_config", "hsv_to_rgb"]

