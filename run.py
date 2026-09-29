"""Fit development gain estimators and reroute paired exact-text-group batches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from bank_io import digest, load_pair, require, write_json
from core import (ARMS, DRAW_SEED, ESTIMATORS, LR_PARAMETERS, RF_PARAMETERS, SEEDS,
                  Frame, batch_actions, batch_stats, fit_gains, metrics_for)
from summarize import analyze_group, write_summary


def run_cell(job):
    cell, weights, inverse, budgets = job
    names, strata = metrics_for(cell)
    records, fixed, original_actions, original_charges = [], [], [], []
    with threadpool_limits(limits=1):
        for estimator in ESTIMATORS:
            gains = cell[estimator]
            index0 = np.arange(len(cell["ids"]))
            a0, c0, _ = batch_actions(cell, gains, index0, budgets)
            original_actions.append(a0)
            original_charges.append(c0)
            one, unchanged = [], []
            for draw in range(len(weights) + 1):
                index = index0 if draw == 0 else np.repeat(index0, weights[draw - 1, inverse])
                actions, charges, optima = batch_actions(cell, gains, index, budgets)
                one.append(batch_stats(cell, gains, index, actions, charges, optima, budgets))
                unchanged.append(np.stack([cell[k][index, a0[..., index]].mean(-1)
                                           for k in ("joint", "t1")], axis=-1))
            records.append(one)
            fixed.append(unchanged)
    return dict(stats=np.asarray(records), fixed_quality=np.asarray(fixed), metrics=np.array(names),
                strata=np.array(strata), estimators=np.array(ESTIMATORS), arms=np.array(ARMS[:a0.shape[2]]),
                ids=cell["ids"], group=cell["group"], rf=cell["rf"], logit=cell["logit"],
                original_actions=np.array(original_actions), original_charges=np.array(original_charges))


def run(args):
    require(len(args.dev) == len(args.test) == len(args.seeds), "Pair one dev/test bank with each seed")
    require(tuple(args.seeds) == SEEDS or (len(args.seeds) == 1 and args.seeds[0] in SEEDS),
            "Use seeds 17 29 43 59 in order, or one of these seeds for an individual fit")
    require(1 <= args.reps <= 2000 and 1 <= args.workers <= 8, "Use 1..2000 draws and 1..8 workers")
    require(not args.out.exists(), "Output directory must be new")
    require(not args.out.resolve().is_relative_to(Path(__file__).resolve().parent),
            "Keep experiment outputs outside the source repository")
    threshold = (.5 if args.corpus == "tama" else 0.) if args.threshold is None else args.threshold
    budgets = (.6, 1.2)
    started, cells, fits, inputs = perf_counter(), [], [], []
    common = None
    with threadpool_limits(limits=1):
        for seed, dev_path, test_path in zip(args.seeds, args.dev, args.test):
            hashes = [digest(path) for path in (dev_path, test_path)]
            dev, test = load_pair(dev_path, test_path, args.corpus, seed, threshold)
            identity = {key: test[key] for key in ("ids", "group", "depth")}
            if common is not None:
                for key in common:
                    require(np.array_equal(common[key], identity[key]), f"Seeds disagree on test {key}/row order")
            common = identity
            estimates, metadata = fit_gains(dev, test)
            test.update(estimates)
            cells.append(test)
            fits.append(dict(seed=seed, **metadata))
            require(hashes == [digest(path) for path in (dev_path, test_path)], "Input changed during fitting")
            inputs.append(dict(seed=seed, dev=dev_path.name, test=test_path.name,
                               dev_sha256=hashes[0], test_sha256=hashes[1], dev_n=len(dev["ids"]), test_n=len(test["ids"])))
    frame = Frame(cells[0]["ids"].tolist(), dict(zip(cells[0]["ids"], cells[0]["group"])), reps=args.reps)
    # Frame IDs are sorted; map groups to original bank order for tied DP actions.
    inverse = np.searchsorted(frame.groups, cells[0]["group"])
    np.testing.assert_array_equal(frame.groups[inverse], cells[0]["group"])
    jobs = [(cell, frame.weights, inverse, budgets) for cell in cells]
    if args.workers == 1:
        results = list(map(run_cell, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
            results = list(executor.map(run_cell, jobs))
    rows = analyze_group(args.corpus, threshold, results, args.seeds, budgets)
    for entry, dev_path, test_path in zip(inputs, args.dev, args.test):
        require(entry["dev_sha256"] == digest(dev_path) and entry["test_sha256"] == digest(test_path),
                "Input changed during rerouting")
    args.out.mkdir(parents=True, exist_ok=False)
    for seed, bank in zip(args.seeds, results):
        np.savez_compressed(args.out / f"seed_{seed}.npz", **bank)
    np.savez_compressed(args.out / "draws.npz", groups=frame.groups, weights=frame.weights)
    write_summary(args.out / "summary.csv", rows)
    write_json(args.out / "fits.json", fits)
    write_json(args.out / "run.json", dict(status="completed", corpus=args.corpus, threshold=threshold,
        seeds=args.seeds, budgets=budgets, replicates=args.reps, draw_seed=DRAW_SEED,
        main_protocol_settings=tuple(args.seeds) == SEEDS and args.reps == 2000,
        point="original batch", interval="pointwise paired group resampling with reallocation; frozen fitted models",
        fixed_action_control="old decisions reweighted; not budget-feasible rerouting", inputs=inputs,
        rf_parameters=RF_PARAMETERS, logit_parameters=LR_PARAMETERS, workers=args.workers,
        seconds=perf_counter() - started, python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy", "threadpoolctl")},
        source_sha256={name: digest(Path(__file__).parent / name) for name in ("core.py", "bank_io.py", "run.py", "summarize.py")},
        outputs={path.name: digest(path) for path in sorted(args.out.iterdir())}))
    print(f"Completed {args.corpus}: {len(cells)} fits, {args.reps} paired draws, {len(rows)} summary rows -> {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("tama", "olid_br", "mold"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--threshold", type=float, help="TAMA: .5/.8/1 (default .5); external: 0 (default)")
    parser.add_argument("--dev", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not __debug__:
        parser.error("Do not use python -O: numerical invariant checks must remain enabled")
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Input/output error: {error}\n")


if __name__ == "__main__":
    main()
