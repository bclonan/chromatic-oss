import pytest
import torch

from chromatensor import ChromaTensor, RGBDCodec, UniformCodec
from chromatensor.operators import chroma_add, chroma_sub, chroma_mul, chroma_dot, chroma_matmul, chroma_similarity, chroma_distance, chroma_activation, LookupOperator, direct_grid_add, LearnedOperator


def test_decoded_operator_correctness():
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    ca, cb = ChromaTensor.from_tensor(a), ChromaTensor.from_tensor(b)
    for operator, reference in [(chroma_add, a + b), (chroma_sub, a - b), (chroma_mul, a * b), (chroma_matmul, a @ b), (chroma_dot, (a * b).sum())]:
        assert torch.allclose(operator(ca, cb).decode(), reference, atol=2e-6)
    assert torch.allclose(chroma_distance(ca, cb), torch.linalg.vector_norm(a - b))
    assert abs(float(chroma_similarity(ca, ca)) - 1) < 1e-6
    assert torch.equal(chroma_activation(ca).decode(), a.relu())


def test_lookup_and_grid_add_all_pairs_are_integer_runtime(monkeypatch):
    codec = UniformCodec(bits=4, minimum=0, maximum=1)
    values = torch.arange(16).float() / 15
    ca = ChromaTensor.from_tensor(values[:, None], codec)
    cb = ChromaTensor.from_tensor(values[None, :], codec)
    lookup = LookupOperator(codec)
    expected = lookup(ca, cb).codes.clone()
    def forbidden(*args):
        raise AssertionError("Runtime must not decode numerical scalar values")
    monkeypatch.setattr(codec, "decode", forbidden)
    assert torch.equal(lookup(ca, cb).codes, expected)
    assert torch.equal(direct_grid_add(ca, cb).codes, expected)


def test_lookup_rejects_mismatched_domains():
    codec = UniformCodec(bits=4, minimum=0, maximum=1)
    lookup = LookupOperator(codec)
    with pytest.raises(ValueError):
        lookup(ChromaTensor.from_tensor(torch.ones(2), codec), ChromaTensor.from_tensor(torch.ones(2)))


@pytest.mark.parametrize("bits,dtype,codes,expected", [
    (8, torch.int8, [100, 127], [200, 254]),
    (16, torch.int16, [20000, 32767], [40000, 65534]),
    (8, torch.int64, [100, 200], [200, 255]),
])
def test_direct_grid_add_uses_a_container_that_holds_the_full_vocabulary(bits, dtype, codes, expected):
    codec = UniformCodec(bits=bits, minimum=0, maximum=1)
    values = ChromaTensor(torch.tensor(codes, dtype=dtype).unsqueeze(-1), codec)
    result = direct_grid_add(values, values)
    assert torch.equal(result.codes.long().flatten(), torch.tensor(expected))
    reference = codec.decode(torch.tensor(expected, dtype=torch.int64).unsqueeze(-1))
    assert torch.equal(result.decode(dtype=torch.float64), reference)
    assert result.codes.dtype == codec.encode(torch.tensor([0.])).dtype


def test_learned_operator_fits_real_finite_function():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    codec = UniformCodec(bits=4, minimum=0, maximum=1)
    table = LookupOperator(codec, operation="mul")
    model = LearnedOperator(16)
    losses = model.fit(table, steps=150, seed=9)
    ids = torch.arange(16)
    a, b = torch.meshgrid(ids, ids, indexing="ij")
    accuracy = (model(a, b) == table.table[..., 0]).float().mean()
    assert losses[-1] < losses[0] / 5
    assert accuracy > 0.9
