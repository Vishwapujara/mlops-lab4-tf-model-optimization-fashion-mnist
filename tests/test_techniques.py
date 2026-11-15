import numpy as np
import pytest

from optimize import techniques as T


@pytest.mark.parametrize("schedule", ["polynomial", "constant"])
def test_pruning_reaches_target_sparsity(trained, toy_data, cfg, schedule):
    pruned = T.prune(trained, *toy_data, cfg, target=0.75, schedule=schedule)
    assert T.weight_stats(pruned)["sparsity"] == pytest.approx(0.75, abs=0.02)


def test_pruning_does_not_touch_the_baseline(trained, toy_data, cfg):
    before = [w.copy() for w in trained.get_weights()]
    T.prune(trained, *toy_data, cfg, target=0.5, schedule="constant")
    assert all(np.array_equal(a, b) for a, b in zip(before, trained.get_weights()))


def test_two_by_four_pattern(trained, toy_data, cfg):
    pruned = T.prune_2_by_4(trained, *toy_data, cfg)
    assert T.weight_stats(pruned)["sparsity"] == pytest.approx(0.5, abs=0.01)
    dense = pruned.layers[-2].kernel.numpy()  # Dense(128): input dim 1600 = 400 blocks of 4
    nonzero_per_block = (dense.reshape(-1, 4, dense.shape[1]) != 0).sum(axis=1)
    assert nonzero_per_block.max() <= 2


def test_clustering_limits_distinct_values(trained, toy_data, cfg):
    clustered = T.cluster(trained, *toy_data, cfg, n_clusters=8)
    assert T.weight_stats(clustered)["max_unique_values"] <= 8


def test_collaborative_chains_keep_sparsity(trained, toy_data, cfg):
    pruned = T.prune(trained, *toy_data, cfg, target=0.75, schedule="polynomial")
    assert T.weight_stats(T.prune_preserving_qat(pruned, *toy_data, cfg))["sparsity"] >= 0.73
    assert T.weight_stats(T.prune_cluster_preserving_qat(pruned, *toy_data, cfg, 8))["sparsity"] >= 0.73


def test_unknown_schedule(trained, toy_data, cfg):
    with pytest.raises(ValueError, match="schedule"):
        T.prune(trained, *toy_data, cfg, target=0.5, schedule="cubic")
