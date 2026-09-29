"""CPU/offline smoke: generate synthetic numeric inputs in a temporary directory."""
import csv
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from bank_io import CLASSES, TRI, TYPES, load_bank, load_pair
from core import (Frame, SEEDS, batch_actions, batch_stats, candidates, exact_selector,
                  frontier, logit_gains, objective, rho, routed)
from summarize import summarize


def solver_checks():
    rng = np.random.default_rng(19)
    for n in range(1, 6):
        gains = rng.integers(-1, 5, size=(n, 3)) / 4
        for menu in ((3,), (1, 2, 3)):
            solve = exact_selector(gains, 3 * n, menu)
            for budget in range(3 * n + 1):
                actions, value = solve(budget)
                brute = max(objective(gains, np.array(a)) for a in product((0,) + menu, repeat=n)
                            if sum(a) <= budget)
                assert abs(value - brute) < 1e-10 and actions.sum() <= budget
    assert exact_selector(np.array([[0., 3., 0.], [0., 0., 4.]]), 3, (1, 2, 3))(3)[1] == 4
    assert exact_selector(np.array([[5e-13, 0., 0.]]), 1, (1, 2, 3))(1)[0].tolist() == [1]
    # Row-order tie behavior is part of frozen allocation reproducibility.
    assert exact_selector(np.ones((2, 3)), 1, (1, 2, 3))(1)[0].tolist() == [1, 0]
    for counts in product(range(3), repeat=3):
        depth = np.repeat([1, 2, 3], counts)
        gains = (np.arange(1, 4)[None] >= depth[:, None]).astype(float)
        for budget in range(13):
            full, prefix, _ = frontier(*counts, budget)
            assert full == exact_selector(gains, budget, (3,))(budget)[1]
            assert prefix == exact_selector(gains, budget, (1, 2, 3))(budget)[1]


def charging_checks():
    calls = []

    def reveal(i, action, spent, cap):
        assert spent + action <= cap
        calls.append(i)
        return 1

    choices = [(1., "duplicate", 3, 0), (1., "duplicate", 3, 1)]
    for cap in (0, 2):
        calls.clear()
        assert not routed(choices, 2, cap, True, reveal)[0].any() and not calls
    calls.clear()
    actions, charges = routed(choices, 2, 4, True, reveal)
    assert calls == [0, 1] and actions.tolist() == [3, 3] and charges.sum() == 2
    assert routed([(1., "x", 1, 0), (.9, "x", 3, 0)], 1, 5, True, reveal)[0].tolist() == [1]
    g = np.ones((1, 3))
    assert candidates(["x"], g, np.array([.2]), np.array([.3]), (3,), True, True)[0][0] == 1 / 1.4
    assert candidates(["x"], g, np.array([.2]), np.array([.3]), (3,), True, False)[0][0] == 1 / 1.26


def resampling_checks():
    ids, groups = np.array(["z", "a", "b"]), np.array(["g2", "g1", "g1"])
    frame = Frame(ids, dict(zip(ids, groups)), reps=10)
    inverse = np.searchsorted(frame.groups, groups)
    assert inverse.tolist() == [1, 0, 0]
    assert np.repeat(np.arange(3), np.array([2, 0])[inverse]).tolist() == [1, 1, 2, 2]
    sorted_values = np.array([1., 2., 3.])
    np.testing.assert_array_equal(frame.sample_sums(sorted_values), frame.weights @ np.array([3., 3.]))
    cell = dict(corpus="mold", seed=17, threshold=0., ids=ids, group=groups,
                depth=np.array([1, 2, 3]), joint=np.array([[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]),
                t1=np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]),
                continue1=np.array([.2, .5, .8]), continue2=np.array([.5, .5, .5]))
    gain = np.array([[0, 0, .9], [0, 0, .8], [0, 0, .7]])
    index = np.array([1, 2, 2])
    old = exact_selector(gain, 3, (3,))(3)[0]
    new = exact_selector(gain[index], 3, (3,))(3)[0]
    assert not old[index].any() and new.sum() == 3
    for values in (gain, np.zeros((3, 3)), np.ones((3, 3))):
        actions, charges, optima = batch_actions(cell, values, index)
        stats = batch_stats(cell, values, index, actions, charges, optima)
        assert np.isfinite(stats[..., :13]).all()
    values = np.array([[10., 1., 2.], [12., 2., 3.], [14., 3., 4.], [16., 4., 5.]])
    mean = list(summarize(values, SEEDS))[-1]
    assert mean["estimate"] == 13. and mean["lo"] == np.quantile([2.5, 3.5], .025)
    assert mean["seed_sd"] == values[:, 0].std(ddof=1)
    assert list(summarize(np.full((4, 3), np.nan), SEEDS))[-1]["empty_replicates"] == 2
    x = np.arange(3.)[:, None]
    for depth in range(4):
        joint = (np.arange(4)[None] >= depth).repeat(3, axis=0).astype(int)
        gains, metadata = logit_gains(x, joint, x)
        np.testing.assert_array_equal(gains, np.repeat([[int(0 < depth <= a) for a in (1, 2, 3)]], 3, 0))
        assert metadata["constant_class"] == depth


def synthetic_bank(corpus, seed, split, n):
    """Illustrative synthetic outcomes, never evidence for manuscript results."""
    rng = np.random.default_rng(seed + (0 if split == "dev" else 100))
    tama = corpus == "tama"
    depth = np.resize([1, 3] if tama else [1, 2, 3], n)
    gold1 = np.where(depth == 1, 0, 1)
    p1 = rng.dirichlet(np.ones(3 if tama else 2), n)
    p2 = rng.dirichlet(np.ones(12 if tama else 2), n)
    p3 = rng.dirichlet(np.ones(3), n)
    # Exact duplicate text can carry conflicting existing references.
    for probability in (p1, p2, p3):
        probability[1] = probability[0]
    first_wrong = p1.argmax(1) != gold1
    minimum = np.array([rng.integers(0, d + 1) for d in depth])
    minimum = np.maximum(minimum, first_wrong.astype(int))
    minimum[depth == 1] = first_wrong[depth == 1]
    joint = (np.arange(4)[None] >= minimum[:, None]).astype(np.int8)
    t1 = np.column_stack([~first_wrong, np.ones((n, 3), bool)]).astype(np.int8)
    groups = np.array([f"{split}-group-{i:03}" for i in range(n)])
    groups[1] = groups[0]
    bank = dict(ids=np.array([f"{split}-{i:03}" for i in reversed(range(n))]), group=groups,
                depth=depth, joint_success=joint, t1_correct=t1, p1=p1, p2=p2,
                split=np.array(split), seed=np.array(seed), model=np.array("synthetic"),
                dataset=np.array("mold2" if corpus == "mold" else corpus),
                checkpoint_sha256=np.array("synthetic-checkpoint-identity"), adapter_sha256=np.array("synthetic-adapter"),
                action_ids=np.arange(4), costs=np.arange(4))
    if tama:
        bank.update(joint_success=np.repeat(joint[:, :, None], 3, axis=2), thresholds=np.array([.5, .8, 1.]),
                    gold_t1=gold1, tri_classes=np.array(TRI), types=np.array(TYPES))
    else:
        bank.update(p3=p3, characters=np.full(n, 40))
        stages = np.full((n, 4, 3), -1, np.int8)
        stages[:, :, 0] = t1
        for i, d in enumerate(depth):
            if d >= 2:
                stages[i, :, 1:d] = 1
                stages[i, :, d - 1] = joint[i]
        bank["stage_correct"] = stages
        del bank["t1_correct"]
        bank.update({f"classes{s}": np.array(classes) for s, classes in enumerate(CLASSES, 1)})
    return bank


def cli_checks(directory):
    root = Path(__file__).resolve().parent
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for corpus in ("tama", "olid_br", "mold"):
        seeds = SEEDS if corpus != "mold" else (17,)
        paths = {split: [] for split in ("dev", "test")}
        for seed in seeds:
            for split, n in (("dev", 48), ("test", 12)):
                path = directory / f"{corpus}_{seed}_{split}.npz"
                np.savez_compressed(path, **synthetic_bank(corpus, seed, split, n))
                paths[split].append(path)
        out = directory / f"out_{corpus}"
        command = [sys.executable, str(root / "run.py"), "--corpus", corpus, "--seeds", *map(str, seeds),
                   "--dev", *map(str, paths["dev"]), "--test", *map(str, paths["test"]),
                   "--out", str(out), "--reps", "6", "--workers", "2" if corpus == "tama" else "1"]
        subprocess.run(command, check=True, env=environment, cwd=directory)
        report = json.loads((out / "run.json").read_text())
        assert report["status"] == "completed" and report["main_protocol_settings"] is False
        with (out / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert any(r["comparison"] == "menu_by_refund" for r in rows)
        assert any(r["comparison"] == "menu_by_reranking" for r in rows)
        assert any(r["estimator"] == "logit_minus_rf" for r in rows)
        assert any(r["seed"] == "four_seed_mean" for r in rows) == (len(seeds) == 4)
        assert {r["valid_replicates"] for r in rows if r["metric"] == "joint"} == {"6"}
        bad = synthetic_bank(corpus, seeds[0], "dev", 48)
        bad["ids"][1] = bad["ids"][0]
        path = directory / "invalid.npz"
        np.savez_compressed(path, **bad)
        try:
            load_bank(path, corpus, seeds[0], "dev", .5 if corpus == "tama" else 0.)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted duplicate IDs")
        try:
            load_pair(paths["test"][0], paths["test"][0], corpus, seeds[0], .5 if corpus == "tama" else 0.)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted incorrect dev/test split")
        if corpus == "tama":
            dev, test = load_pair(paths["dev"][0], paths["test"][0], corpus, seeds[0], .8)
            actions, charges, values = batch_actions(test, np.ones((12, 3)), np.arange(12))
            assert actions.shape == (2, 2, 1, 12)
            assert rho(test["joint"]).shape == (12,)
            np.testing.assert_array_equal(test["ids"], synthetic_bank(corpus, seeds[0], "test", 12)["ids"])
            bank = synthetic_bank(corpus, seeds[0], "test", 12)
            bank["tri_classes"] = bank["tri_classes"][::-1]
            np.savez_compressed(path, **bank)
            try:
                load_bank(path, corpus, seeds[0], "test", .5)
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted incorrect class order")


def main():
    if not __debug__:
        raise RuntimeError("Smoke assertions require Python without -O")
    solver_checks()
    charging_checks()
    resampling_checks()
    with tempfile.TemporaryDirectory(prefix="c-budget-smoke-") as path:
        cli_checks(Path(path))
    print("PASS: exhaustive allocation/frontier, reservation/refund, duplicates, LR classes, paired summaries, all three CLI connectors")


if __name__ == "__main__":
    main()
