"""Paired seed-wise and four-seed summaries from c_rerouting_summary.py."""
import csv

import numpy as np

from core import ESTIMATORS, bounds

QUALITY = ("joint", "t1")
COMPARABLE = ("joint", "t1", "repair", "loss", "epsilon", "spent", "reviewed", "refund")


def count_unit(name):
    return "check_units" if name in ("cap", "spent", "nominal", "unused", "refund", "repriced_spend") or name.endswith(("_spent", "_nominal")) else "records"


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)


def summarize(values, seeds):
    """Keep original point; average four seed contrasts within each paired draw."""
    assert values.ndim == 2 and values.shape[0] == len(seeds) and values.shape[1] > 1
    samples = list(zip(seeds, values))
    if len(seeds) == 4:
        samples.append(("four_seed_mean", values.mean(0)))
    for seed, sample in samples:
        yield dict(seed=seed, estimate=float(sample[0]) if np.isfinite(sample[0]) else None,
                   seed_sd=float(values[:, 0].std(ddof=1)) if seed == "four_seed_mean" and np.isfinite(values[:, 0]).all() else None,
                   **bounds(sample[1:]))


def analyze_group(corpus, threshold, banks, seeds, budgets):
    first, rows = banks[0], []
    for bank in banks[1:]:
        for key in ("metrics", "strata", "estimators", "arms", "ids", "group"):
            np.testing.assert_array_equal(bank[key], first[key])
    stats = np.stack([bank["stats"] for bank in banks])
    fixed = np.stack([bank["fixed_quality"] for bank in banks])
    # seed, estimator, original/draw, budget, menu, arm, metric
    assert stats.ndim == 7 and stats.shape[:2] == (len(seeds), 2)
    names, arms = first["metrics"].tolist(), first["arms"].tolist()
    raw = {name: stats[..., k] for k, name in enumerate(names)}
    n = raw["n"]
    assert (n > 0).all()
    np.testing.assert_array_equal(n, np.broadcast_to(n[:1, :1, :, :, :1, :1], n.shape))
    metrics = {}
    for name, values in raw.items():
        metrics[name + "_count"] = (values, count_unit(name))
        if name in ("n", "cap"):
            continue
        stratum = next((s for s in first["strata"] if name.startswith(str(s) + "_")), None)
        if stratum is None:
            metrics[name] = (ratio(values, n), "per_record")
        elif not name.endswith("_n"):
            metrics[name] = (ratio(values, raw[str(stratum) + "_n"]), "within_stratum")
            metrics[name + "_contribution"] = (ratio(values, n), "per_record")
    compare = list(COMPARABLE)
    compare += [f"{s}_{metric}" for s in first["strata"] for metric in QUALITY]
    compare += [f"{s}_{metric}_contribution" for s in first["strata"] for metric in QUALITY]

    def emit(values, **description):
        if not np.isnan(values).all():
            rows.extend(dict(**description, **r) for r in summarize(values, seeds))

    for b, budget in enumerate(budgets):
        base = dict(corpus=corpus, threshold=threshold, budget_per_record=budget)
        for name, (values, unit) in metrics.items():
            for e, estimator in enumerate(ESTIMATORS):
                for m, menu in enumerate(("full", "prefix")):
                    for a, arm in enumerate(arms):
                        emit(values[:, e, :, b, m, a], **base, estimator=estimator,
                             comparison=f"{menu}_{arm}", metric=name, unit=unit, estimand="rerouted_policy")
        for name in compare:
            values, unit = metrics[name]
            for e, estimator in enumerate(ESTIMATORS):
                def add(label, value):
                    emit(value, **base, estimator=estimator, comparison=label,
                         metric=name, unit=unit, estimand="paired_rerouting_contrast")
                menu_effect = values[:, e, :, b, 1] - values[:, e, :, b, 0]
                for a, arm in enumerate(arms):
                    add(f"prefix_minus_full_{arm}", menu_effect[:, :, a])
                if len(arms) == 4:
                    for after, before, label in ((1, 0, "R0_minus_exact"), (2, 1, "refund"), (3, 2, "reranking")):
                        for m, menu in enumerate(("full", "prefix")):
                            add(f"{menu}_{label}", values[:, e, :, b, m, after] - values[:, e, :, b, m, before])
                        add(f"menu_by_{label}", menu_effect[:, :, after] - menu_effect[:, :, before])
            difference = values[:, 1, :, b] - values[:, 0, :, b]
            for a, arm in enumerate(arms):
                for m, menu in enumerate(("full", "prefix")):
                    emit(difference[:, :, m, a], **base, estimator="logit_minus_rf", comparison=f"{menu}_{arm}",
                         metric=name, unit=unit, estimand="paired_estimator_contrast")
                emit(difference[:, :, 1, a] - difference[:, :, 0, a], **base, estimator="logit_minus_rf",
                     comparison=f"prefix_minus_full_{arm}", metric=name, unit=unit, estimand="estimator_by_menu")
        for e, estimator in enumerate(ESTIMATORS):
            for k, name in enumerate(QUALITY):
                for a, arm in enumerate(arms):
                    old_menu = fixed[:, e, :, b, 1, a, k] - fixed[:, e, :, b, 0, a, k]
                    emit(old_menu, **base, estimator=estimator, comparison=f"prefix_minus_full_{arm}",
                         metric=name, unit="per_record", estimand="fixed_action_resampling")
                    for m, menu in enumerate(("full", "prefix")):
                        difference = metrics[name][0][:, e, :, b, m, a] - fixed[:, e, :, b, m, a, k]
                        assert np.allclose(difference[:, 0], 0., atol=1e-15, rtol=0)
                        emit(difference, **base, estimator=estimator, comparison=f"{menu}_{arm}",
                             metric=name, unit="per_record", estimand="rerouted_minus_fixed_action")
    return rows


def write_summary(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
