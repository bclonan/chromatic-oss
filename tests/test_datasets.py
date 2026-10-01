import pytest
import torch

from experiments.datasets import stratified_indices


def test_full_stratified_selection_keeps_imbalanced_official_style_counts():
    counts = torch.tensor([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    labels = torch.repeat_interleave(torch.arange(10), counts)
    selected = stratified_indices(labels, 60000, 7)
    assert selected.numel() == 60000
    assert torch.equal(selected.sort().values, torch.arange(60000))
    assert torch.equal(selected, stratified_indices(labels, 60000, 7))


def test_stratified_shortages_fill_requested_subset_without_replacement():
    labels = torch.tensor([0] + [1] * 10 + [2] * 20)
    selected = stratified_indices(labels, 18, 7)
    assert selected.numel() == selected.unique().numel() == 18
    assert (labels[selected] == 0).sum() == 1
    assert set(labels[selected].tolist()) == {0, 1, 2}
    assert torch.equal(selected, stratified_indices(labels, 18, 7))
    assert not torch.equal(selected, stratified_indices(labels, 18, 23))


def test_selection_preserves_balanced_quotas_and_zero_count():
    labels = torch.arange(3).repeat_interleave(10)
    selected = stratified_indices(labels, 8, 7)
    assert torch.bincount(labels[selected]).tolist() == [3, 3, 2]
    assert stratified_indices(labels, 0, 7).numel() == 0
    assert stratified_indices(labels, 40, 7).numel() == len(labels)


@pytest.mark.parametrize("labels,count", [([], 1), ([[1, 2]], 1), ([0, 1], -1), ([0, 1], True)])
def test_selection_rejects_invalid_requests(labels, count):
    with pytest.raises(ValueError):
        stratified_indices(labels, count, 7)
