"""Budget allocation, adapted from the project's frozen C experiment sources.

Sources: optimizer_probe.exact_selector; derive_audit.frontier;
single_post_exact_group_audit.Frame; c_budget candidates/routed;
single_post_pretrained_charging.candidates; c_rerouting.logit_gains.
No raw-text, training-checkpoint, or historical-audit dependencies.
"""
from fractions import Fraction
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEEDS = (17, 29, 43, 59)
MENUS = {"full": (3,), "prefix": (1, 2, 3)}
ARMS = ("exact", "R0", "R1", "R2")
ESTIMATORS = ("rf", "logit")
DRAW_SEED = 20260923
COSTS = np.arange(4)
RF_PARAMETERS = dict(n_estimators=200, max_depth=5, min_samples_leaf=15,
                     n_jobs=1, random_state=17)
LR_PARAMETERS = dict(C=1., solver="lbfgs", tol=1e-4, max_iter=2000,
                     class_weight=None, penalty="l2")


def objective(gains, actions):
    selected = np.flatnonzero(actions)
    return float(gains[selected, actions[selected] - 1].sum())


def exact_selector(gains, max_budget, menu, costs=COSTS):
    """Exact at-most-budget multiple-choice knapsack; preserve input row ties."""
    assert gains.ndim == 2 and gains.shape[1] == 3 and max_budget >= 0
    costs = np.asarray(costs)
    assert costs.shape == (4,) and costs[0] == 0
    assert np.all(costs[1:] > 0) and np.all(costs == costs.astype(int))
    n = len(gains)
    previous = np.zeros(max_budget + 1)
    choices = np.zeros((n, max_budget + 1), dtype=np.int8)
    for i in range(n):
        current = previous.copy()
        for action in menu:
            cost = int(costs[action])
            if cost > max_budget:
                continue
            candidate = previous[:-cost] + gains[i, action - 1]
            better = candidate > current[cost:]
            current[cost:][better] = candidate[better]
            choices[i, cost:][better] = action
        previous = current

    def trace(budget):
        assert 0 <= budget <= max_budget
        actions = np.zeros(n, dtype=np.int8)
        remaining = budget
        for i in range(n - 1, -1, -1):
            action = int(choices[i, remaining])
            actions[i] = action
            remaining -= int(costs[action])
        assert remaining >= 0
        assert abs(objective(gains, actions) - previous[budget]) < 1e-8
        return actions, float(previous[budget])
    return trace


def frontier(n1, n2, n3, budget):
    if min(n1, n2, n3, budget) < 0:
        raise ValueError("Negative frontier input")
    k1 = min(n1, budget)
    k2 = min(n2, (budget - k1) // 2)
    k3 = min(n3, (budget - k1 - 2 * k2) // 3)
    return min(n1 + n2 + n3, budget // 3), k1 + k2 + k3, (k1, k2, k3)


def rho(joint):
    assert joint.ndim == 2 and joint.shape[1] == 4 and np.isin(joint, [0, 1]).all()
    assert (np.diff(joint.astype(int), axis=1) >= 0).all() and joint[:, 3].all()
    return joint.argmax(1)


def logit_gains(dev_x, dev_joint, test_x):
    y = rho(dev_joint)
    scaler = StandardScaler().fit(dev_x)
    x = scaler.transform(dev_x)
    classes = np.unique(y)
    probability = np.zeros((len(test_x), 4))
    metadata = dict(parameters=LR_PARAMETERS, dev_class_counts=np.bincount(y, minlength=4).tolist(),
                    classes=classes.tolist(), scaler_mean=scaler.mean_.tolist(),
                    scaler_scale=scaler.scale_.tolist(), scaler_var=scaler.var_.tolist(),
                    scaler_n=int(scaler.n_samples_seen_), converged=True)
    if len(classes) == 1:
        probability[:, classes[0]] = 1
        metadata.update(constant_class=int(classes[0]), iterations=[])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(**LR_PARAMETERS).fit(x, y)
        if (model.n_iter_ >= LR_PARAMETERS["max_iter"]).any():
            raise RuntimeError("LR iteration limit reached; no test inference")
        probability[:, model.classes_] = model.predict_proba(scaler.transform(test_x))
        metadata.update(classes=model.classes_.tolist(), iterations=model.n_iter_.tolist(),
                        coef=model.coef_.tolist(), intercept=model.intercept_.tolist())
    gains = np.clip(np.cumsum(probability[:, 1:], axis=1), 0., 1.)
    assert np.isfinite(gains).all() and (np.diff(gains, axis=1) >= -1e-12).all()
    return gains, metadata


def fit_gains(dev, test):
    target = dev["joint"][:, 1:].astype(int) - dev["joint"][:, :1].astype(int)
    forest = RandomForestRegressor(**RF_PARAMETERS).fit(dev["features"], target)
    rf = forest.predict(test["features"])
    logit, metadata = logit_gains(dev["features"], dev["joint"], test["features"])
    return {"rf": rf, "logit": logit}, metadata


def candidates(ids, gains, p1, p2, menu, expected=False, tama=False):
    choices = []
    for i, rid in enumerate(ids):
        for action in menu:
            if not expected:
                cost = action
            elif tama:
                cost = 1 + (action - 1) * p1[i]
            else:
                cost = 1. + (action >= 2) * p1[i] + (action >= 3) * p1[i] * p2[i]
            assert 1 <= cost <= action + 1e-7
            if gains[i, action - 1] > 0:
                choices.append((float(gains[i, action - 1] / cost), str(rid), action, i))
    return sorted(choices, key=lambda row: (-row[0], row[1], row[2]))


def routed(choices, n, cap, refund, reveal):
    """Reserve requested depth before reveal; never upgrade an admitted instance."""
    assert isinstance(cap, int) and cap >= 0
    actions, charges = np.zeros(n, np.int8), np.zeros(n, np.int8)
    spent = 0
    for priority, _, action, i in choices:
        assert priority > 0 and action in (1, 2, 3) and 0 <= i < n
        if actions[i] or spent + action > cap:
            continue
        checked_depth = reveal(i, action, spent, cap)
        assert 1 <= checked_depth <= action
        charge = checked_depth if refund else action
        actions[i], charges[i] = action, charge
        spent += charge
        assert spent <= cap
    return actions, charges


def bounds(samples):
    good = np.isfinite(samples)
    values = samples[good]
    lo, hi = np.quantile(values, [.025, .975]) if len(values) else (None, None)
    return dict(lo=float(lo) if lo is not None else None,
                hi=float(hi) if hi is not None else None,
                valid_replicates=int(good.sum()), empty_replicates=int((~good).sum()))


class Frame:
    """Frozen exact-text-group draws; values passed to sample_sums use sorted IDs."""
    def __init__(self, ids, text_groups, reps=2000):
        self.ids = sorted(ids)
        assert len(self.ids) == len(set(self.ids)) and set(self.ids) <= set(text_groups)
        self.groups, self.inverse = np.unique([text_groups[k] for k in self.ids], return_inverse=True)
        self.weights = np.random.default_rng(DRAW_SEED).multinomial(
            len(self.groups), np.ones(len(self.groups)) / len(self.groups), size=reps)

    def sample_sums(self, values):
        values = np.asarray(values, dtype=float)
        assert values.shape[0] == len(self.ids) and np.isfinite(values).all()
        grouped = np.zeros((len(self.groups),) + values.shape[1:])
        np.add.at(grouped, self.inverse, values)
        return (self.weights @ grouped.reshape(len(self.groups), -1)).reshape(
            (len(self.weights),) + values.shape[1:])


def budget_cap(rate, n):
    return int(Fraction(str(rate)) * n)


def batch_actions(cell, gains, index, budgets=(.6, 1.2)):
    ids, g = cell["ids"][index], gains[index]
    n = len(ids)
    caps = [budget_cap(rate, n) for rate in budgets]
    arms = ARMS if cell["threshold"] in (0., .5) else ARMS[:1]
    actions = np.zeros((len(caps), 2, len(arms), n), np.int8)
    charges = np.zeros_like(actions)
    optima = np.zeros((len(caps), 2))

    def reveal(i, requested, spent, cap):
        assert spent + requested <= cap
        return int(min(requested, cell["depth"][index[i]]))

    for m, menu in enumerate(MENUS.values()):
        solve = exact_selector(g, max(caps), menu)
        lists = {}
        if len(arms) > 1:
            lists = {expected: candidates(ids, g, cell["continue1"][index], cell["continue2"][index],
                                         menu, expected, cell["corpus"] == "tama")
                     for expected in (False, True)}
        for b, cap in enumerate(caps):
            actions[b, m, 0], optima[b, m] = solve(cap)
            charges[b, m, 0] = actions[b, m, 0]
            for a, arm in enumerate(arms[1:], 1):
                actions[b, m, a], charges[b, m, a] = routed(lists[arm == "R2"], n, cap, arm != "R0", reveal)
                expected = np.minimum(actions[b, m, a], cell["depth"][index]) if arm != "R0" else actions[b, m, a]
                np.testing.assert_array_equal(charges[b, m, a], expected)
            assert (charges[b, m].sum(-1) <= cap).all()
            assert np.isin(actions[b, m], (0,) + menu).all()
            if len(arms) > 1:
                gap = optima[b, m] - objective(g, actions[b, m, 1])
                assert gap >= -1e-8 and (m != 0 or abs(gap) < 1e-8)
    assert (optima[:, 1] >= optima[:, 0] - 1e-8).all()
    return actions, charges, optima


BASE_METRICS = ("n", "cap", "baseline_joint", "baseline_t1", "rho0", "rho1", "rho2", "rho3",
                "oracle_full", "oracle_prefix", "joint", "t1", "repair", "loss", "predicted",
                "epsilon", "reviewed", "nominal", "spent", "unused", "refund", "repriced_spend",
                "action0", "action1", "action2", "action3")
STRATUM_METRICS = ("n", "baseline_joint", "baseline_t1", "joint", "t1", "reviewed", "spent", "nominal",
                   "action0", "action1", "action2", "action3")


def metrics_for(cell):
    strata = ("TA", "nonTA") if cell["corpus"] == "tama" else ("D1", "D2", "D3")
    return list(BASE_METRICS) + [f"{s}_{m}" for s in strata for m in STRATUM_METRICS], strata


def batch_stats(cell, gains, index, actions, charges, optima, budgets=(.6, 1.2)):
    """References score outcomes after selection; fixed-cost frontier excludes refunds."""
    names, strata = metrics_for(cell)
    joint, t1, depth = (cell[k][index] for k in ("joint", "t1", "depth"))
    n, rows = len(index), np.arange(len(index))
    counts = np.bincount(rho(joint), minlength=4)
    masks = {"TA": depth == 3, "nonTA": depth != 3} if cell["corpus"] == "tama" else {
        f"D{d}": depth == d for d in (1, 2, 3)}
    out = np.zeros(actions.shape[:-1] + (len(names),))
    for b, rate in enumerate(budgets):
        cap = budget_cap(rate, n)
        vf, vp, _ = frontier(*map(int, counts[1:]), cap)
        for m in range(2):
            for a in range(actions.shape[2]):
                act, cost = actions[b, m, a], charges[b, m, a]
                j, first = joint[rows, act], t1[rows, act]
                repair = int(j.sum() - joint[:, 0].sum())
                value = objective(gains[index], act)
                loss = (vf if m == 0 else vp) - repair if a < 2 else np.nan
                assert a >= 2 or loss >= 0
                record = dict(n=n, cap=cap, baseline_joint=int(joint[:, 0].sum()), baseline_t1=int(t1[:, 0].sum()),
                              **{f"rho{k}": int(counts[k]) for k in range(4)}, oracle_full=vf, oracle_prefix=vp,
                              joint=int(j.sum()), t1=int(first.sum()), repair=repair, loss=loss, predicted=value,
                              epsilon=optima[b, m] - value if a < 2 else np.nan, reviewed=int((act > 0).sum()),
                              nominal=int(act.sum()), spent=int(cost.sum()), unused=cap - int(cost.sum()),
                              refund=int((act - cost).sum()), repriced_spend=int(np.minimum(act, depth).sum()),
                              **{f"action{k}": int((act == k).sum()) for k in range(4)})
                for s in strata:
                    mask = masks[s]
                    sub = dict(n=int(mask.sum()), baseline_joint=int(joint[mask, 0].sum()), baseline_t1=int(t1[mask, 0].sum()),
                               joint=int(j[mask].sum()), t1=int(first[mask].sum()), reviewed=int((act[mask] > 0).sum()),
                               spent=int(cost[mask].sum()), nominal=int(act[mask].sum()),
                               **{f"action{k}": int((act[mask] == k).sum()) for k in range(4)})
                    record.update({f"{s}_{k}": v for k, v in sub.items()})
                out[b, m, a] = [record[k] for k in names]
        for a in range(min(2, actions.shape[2])):
            full, prefix = out[b, 0, a], out[b, 1, a]
            assert prefix[names.index("repair")] - full[names.index("repair")] == vp - vf - (
                prefix[names.index("loss")] - full[names.index("loss")])
    return out
