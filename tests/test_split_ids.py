"""Manifest hard-coded IDs and train_eval reproducibility."""
from dataset.split_ids import (HIDDEN_VAL_IDS, TEST_IDS, TRAIN_IDS, VAL_IDS,
                               build_train_eval_ids)


def test_counts():
    assert len(TRAIN_IDS) == 400
    assert len(VAL_IDS) == 34
    assert len(TEST_IDS) == 50
    assert len(HIDDEN_VAL_IDS) == 16


def test_disjoint_union_500():
    union = set(TRAIN_IDS) | set(VAL_IDS) | set(TEST_IDS) | set(HIDDEN_VAL_IDS)
    assert union == set(range(1, 501))
    # disjoint
    assert (len(TRAIN_IDS) + len(VAL_IDS) + len(TEST_IDS)
            + len(HIDDEN_VAL_IDS) == 500)


def test_train_eval_reproducible():
    a = build_train_eval_ids(seed=42)
    b = build_train_eval_ids(seed=42)
    assert a == b
    assert len(a) == 34
    assert set(a).issubset(set(TRAIN_IDS))
