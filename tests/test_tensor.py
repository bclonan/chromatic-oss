import pytest
import numpy as np
import torch

from chromatensor.tensor import ChromaTensor, cat, stack
from chromatensor.codecs import RGBCodec, RGBDCodec, UniformCodec, LearnedCodec, HierarchicalCodec


def test_default_tensor_roundtrip_arbitrary_shape_and_real_storage():
    x = torch.randn(4, 7, 9)
    chroma = ChromaTensor.from_tensor(x)
    assert isinstance(chroma.codec, RGBDCodec)
    assert torch.equal(chroma.decode(), x)
    assert chroma.shape == x.shape
    assert chroma.dtype == torch.uint8
    assert chroma.decoded_dtype == torch.float32
    assert chroma.device == x.device
    assert chroma.rgb.shape == (*x.shape, 3)
    assert chroma.storage_nbytes == x.numel() * 4
    assert chroma.numel() == x.numel()
    assert set(chroma.__dict__) == {"codes", "codec", "decoded_dtype"}


@pytest.mark.parametrize("codec", [RGBCodec(), UniformCodec(16), LearnedCodec(1024), HierarchicalCodec()])
def test_tensor_structure_matches_torch_and_keeps_code_channels(codec):
    x = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
    chroma = ChromaTensor.from_tensor(x, codec)
    decoded = chroma.decode()
    assert torch.equal(chroma.reshape(6, 4).decode(), decoded.reshape(6, 4))
    assert torch.equal(chroma.reshape((4, -1)).decode(), decoded.reshape(4, -1))
    assert torch.equal(chroma.transpose(0, -1).decode(), decoded.transpose(0, -1))
    assert torch.equal(chroma.T.decode(), decoded.permute(2, 1, 0))
    assert torch.equal(chroma.flatten(1).decode(), decoded.flatten(1))
    for index in [(1,), (slice(None), 1), (Ellipsis, 2), (None, Ellipsis, slice(1, 3)),
                  (torch.tensor([0, 1]), slice(None), 2), (x > 0,)]:
        assert torch.equal(chroma[index].decode(), decoded[index])
    with pytest.raises(IndexError):
        chroma[:, :, :, 0]
    with pytest.raises(IndexError):
        chroma.transpose(0, 3)
    with pytest.raises(ValueError):
        chroma.reshape(-1, -1)


def test_shape_operations_never_decode_values_again():
    codec = RGBDCodec()
    chroma = ChromaTensor.from_tensor(torch.randn(4, 4), codec)
    def rejected_decode(codes):
        raise AssertionError("Shape operation decoded the integer representation")
    codec.decode = rejected_decode
    assert chroma.reshape(16).shape == (16,)
    assert chroma[:, 1].shape == (4,)
    assert chroma.transpose(0, 1).shape == (4, 4)
    assert chroma.to("cpu").shape == (4, 4)
    assert chroma.clone().shape == (4, 4)


@pytest.mark.parametrize("codec", [RGBDCodec(), UniformCodec(16), LearnedCodec(1024)])
@pytest.mark.parametrize("flag", [True, False])
def test_boolean_scalar_indexing_preserves_logical_axes(codec, flag):
    chroma = ChromaTensor.from_tensor(torch.arange(6.).reshape(2, 3), codec)
    numerical = chroma.decode()
    for index in [(flag, Ellipsis, 0), (flag, slice(None), 0), (None, flag, Ellipsis, 1)]:
        assert torch.equal(chroma[index].decode(), numerical[index])
        numpy_index = tuple(np.bool_(value) if isinstance(value, bool) else value for value in index)
        assert torch.equal(chroma[numpy_index].decode(), numerical[index])


@pytest.mark.parametrize("codec", [RGBDCodec(), UniformCodec(16)])
def test_boolean_lists_and_numpy_masks_keep_code_channels(codec):
    chroma = ChromaTensor.from_tensor(torch.arange(6.).reshape(2, 3), codec)
    numerical = chroma.decode()
    for mask in [[True, False], [[True, False, True], [False, True, False]]]:
        tensor_mask = torch.tensor(mask, dtype=torch.bool)
        for index in [mask, np.array(mask, dtype=bool), tensor_mask]:
            result = chroma[index]
            assert torch.equal(result.decode(), numerical[tensor_mask])
            assert result.codes.shape[-1] == codec.channels
    index = ([True, False], Ellipsis, 1)
    assert torch.equal(chroma[index].decode(), numerical[index])


def test_cat_stack_scalar_and_incompatible_codecs():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    chroma = ChromaTensor.from_tensor(x)
    assert torch.equal(cat([chroma, chroma], -1).decode(), torch.cat([x, x], -1))
    assert torch.equal(stack([chroma, chroma], -1).decode(), torch.stack([x, x], -1))
    scalar = ChromaTensor.from_tensor(torch.tensor(2.5))
    assert scalar.shape == ()
    assert torch.equal(stack([scalar, scalar]).decode(), torch.tensor([2.5, 2.5]))
    assert scalar.reshape(1).shape == (1,)
    with pytest.raises(TypeError):
        len(scalar)
    with pytest.raises(ValueError):
        cat([chroma, ChromaTensor.from_tensor(x, RGBCodec())])
    with pytest.raises(ValueError):
        stack([])


def test_dtype_interop_and_storage_validation():
    x = torch.linspace(-1, 1, 17, dtype=torch.float64)
    chroma = ChromaTensor.from_tensor(x, RGBCodec())
    assert chroma.decode().dtype == torch.float64
    assert chroma.to_torch(torch.float32).dtype == torch.float32
    assert chroma.to("cpu", decoded_dtype=torch.float16).decode().dtype == torch.float16
    with pytest.raises(TypeError):
        chroma.to(torch.float32)
    with pytest.raises(TypeError):
        ChromaTensor(torch.zeros(3, 3), RGBCodec())
    with pytest.raises(ValueError):
        ChromaTensor(torch.zeros(3, 3, dtype=torch.uint8), RGBCodec(), shape=(4,))
    clone = chroma.clone()
    clone.codes[0] = 255
    assert not torch.equal(clone.codes, chroma.codes)


@pytest.mark.parametrize("dtype", [torch.uint32, torch.uint64])
def test_unsigned_32_bit_code_indexing_preserves_high_values(dtype):
    values = [0, 2**31, 2**32 - 1]
    chroma = ChromaTensor(torch.tensor(values, dtype=dtype).unsqueeze(-1), UniformCodec(32))
    decoded = chroma.decode()
    for index in [slice(1, None), torch.tensor([2, 1]), torch.tensor([False, True, True])]:
        result = chroma[index]
        assert result.codes.dtype == dtype
        assert result.codes.tolist() == [[values[i]] for i in torch.arange(3)[index].tolist()]
        assert torch.equal(result.decode(), decoded[index])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_tensor_cpu_cuda_interoperation():
    x = torch.randn(4, 16)
    chroma = ChromaTensor.from_tensor(x)
    device_tensor = chroma.cuda()
    assert device_tensor.device.type == "cuda"
    assert torch.equal(device_tensor.decode().cpu(), x)
    assert torch.equal(device_tensor[:, 2].cpu().decode(), x[:, 2])

