"""Explicit numeric NPZ connector; original prediction banks are also accepted."""
import hashlib
import json
from pathlib import Path

import numpy as np

TRI = ("non-abusive", "targeted-abusive", "unidentified-targets")
TYPES = ("death_threat", "sexual_assault", "sexual_explicit", "physical_harm", "radiation_of_threats",
         "attacks_on_credibility", "misogynistic", "homophobic", "religious", "political_sectarian", "racist", "general")
CLASSES = (("NOT", "OFF"), ("UNT", "TIN"), ("IND", "GRP", "OTH"))
FIELDS = ("ids", "group", "joint_success", "t1_correct", "stage_correct", "depth", "gold_t1", "gold",
          "p1", "p2", "p3", "characters", "features", "thresholds", "threshold", "tri_classes", "types",
          "classes1", "classes2", "classes3", "action_ids", "costs", "split", "seed", "dataset", "model",
          "checkpoint_sha256", "adapter_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def probability(bank, key, n, width):
    require(key in bank, f"Missing {key}")
    value = bank[key]
    require(value.shape == (n, width) and value.dtype.kind in "fiu", f"Invalid {key} shape/type")
    require(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), f"Invalid {key} values")
    require(np.allclose(value.sum(1), 1, atol=2e-6, rtol=0), f"{key} probabilities must sum to one")
    return value


def load_bank(path, corpus, seed, split, threshold):
    with np.load(path, allow_pickle=False) as archive:
        bank = {key: archive[key] for key in FIELDS if key in archive.files}
    require(all(value.dtype.kind != "O" for value in bank.values()), "Object arrays are unsupported")
    require(all(key in bank for key in ("ids", "group", "joint_success")), "Missing ids/group/joint_success")
    ids, groups = bank["ids"], bank["group"]
    n = len(ids)
    require(n > 0 and ids.shape == groups.shape == (n,), "Empty or invalid IDs/groups")
    require(ids.dtype.kind in "US" and groups.dtype.kind in "US", "IDs/groups must be strings")
    ids, groups = ids.astype(str), groups.astype(str)
    require(len(set(ids)) == n and (ids != "").all() and (groups != "").all(), "IDs must be unique; groups nonempty")
    for key, expected in (("split", split), ("seed", seed), ("dataset", "mold2" if corpus == "mold" else corpus)):
        if key in bank:
            require(bank[key].shape == () and bank[key].item() == expected, f"Mismatched {key}")
    for key in ("action_ids", "costs"):
        if key in bank:
            require(np.array_equal(bank[key], [0, 1, 2, 3]), f"Unexpected {key}")
    if corpus == "tama":
        require(threshold in (.5, .8, 1.), "TAMA threshold must be .5, .8, or 1")
        for key, expected in (("tri_classes", TRI), ("types", TYPES)):
            if key in bank:
                require(np.array_equal(bank[key], expected), f"Unexpected {key} order")
        p1, p2 = probability(bank, "p1", n, 3), probability(bank, "p2", n, 12)
        entropy = -(p1 * np.log(np.maximum(p1, 1e-12))).sum(1)
        proxy = p1.max(1) * np.where(p1.argmax(1) == 1, p2.max(1), 1.)
        features = np.column_stack((p1, entropy, proxy))
        joint = bank["joint_success"]
        if joint.ndim == 3:
            require(joint.shape == (n, 4, 3) and np.array_equal(bank.get("thresholds"), [.5, .8, 1.]),
                    "TAMA 3D outcomes require thresholds [.5, .8, 1]")
            joint = joint[:, :, (.5, .8, 1.).index(threshold)]
        else:
            require("threshold" in bank and bank["threshold"].shape == () and bank["threshold"].item() == threshold,
                    "TAMA 2D outcomes require matching scalar threshold")
        require("depth" in bank or "gold_t1" in bank, "TAMA requires depth or gold_t1")
        depth = bank["depth"] if "depth" in bank else np.where(bank["gold_t1"] == 1, 3, 1)
        require(np.isin(depth, [1, 3]).all(), "TAMA depth must be 1 or 3")
        if "gold_t1" in bank:
            require(bank["gold_t1"].shape == (n,) and np.isin(bank["gold_t1"], [0, 1, 2]).all(), "Invalid gold_t1")
            require(np.array_equal(depth, np.where(bank["gold_t1"] == 1, 3, 1)), "Depth disagrees with gold_t1")
        continue1, continue2 = p1[:, 1], np.ones(n)
    else:
        require(threshold == 0., "Classification corpora use threshold 0")
        probabilities = [probability(bank, f"p{s}", n, len(classes)) for s, classes in enumerate(CLASSES, 1)]
        for s, classes in enumerate(CLASSES, 1):
            if f"classes{s}" in bank:
                require(np.array_equal(bank[f"classes{s}"], classes), f"Unexpected classes{s} order")
        require("characters" in bank and bank["characters"].shape == (n,), "Missing character counts")
        characters = bank["characters"]
        require(characters.dtype.kind in "fiu" and np.isfinite(characters).all() and (characters >= 0).all()
                and (characters == np.floor(characters)).all(), "Invalid character counts")
        features = np.column_stack([*probabilities, *[-(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
                                                    for p in probabilities], np.log1p(characters)])
        require("depth" in bank, "Missing depth")
        depth, joint = bank["depth"], bank["joint_success"]
        require(np.isin(depth, [1, 2, 3]).all(), "Invalid external applicability depth")
        p1 = probabilities[0]
        continue1, continue2 = p1[:, 1], probabilities[1][:, 1]
    require(depth.shape == (n,), "Invalid depth shape")
    require(joint.shape == (n, 4) and np.isin(joint, [0, 1]).all(), "Invalid joint_success")
    require((np.diff(joint.astype(int), axis=1) >= 0).all() and joint[:, 3].all(), "Nonmonotone/incomplete full review")
    require(joint[np.arange(n), depth.astype(int)].all(), "Reviewing every applicable field must complete the output")
    if "t1_correct" in bank:
        t1 = bank["t1_correct"]
    else:
        require("stage_correct" in bank and bank["stage_correct"].shape == (n, 4, 3), "Missing t1_correct/stage_correct")
        require(np.isin(bank["stage_correct"][:, :, 0], [0, 1]).all(), "Invalid stage-one correctness")
        t1 = bank["stage_correct"][:, :, 0]
    require(t1.shape == (n, 4) and np.isin(t1, [0, 1]).all() and t1[:, 1:].all(), "Invalid T1 outcomes")
    require((joint <= t1).all() and np.array_equal(joint[depth == 1], t1[depth == 1]), "Joint/T1 applicability mismatch")
    if "stage_correct" in bank and corpus != "tama":
        stages = bank["stage_correct"]
        require(stages.shape == (n, 4, 3) and np.isin(stages, [-1, 0, 1]).all(), "Invalid stage_correct")
        require(np.array_equal(joint, (stages != 0).all(2)) and np.array_equal(t1, stages[:, :, 0]),
                "Stage outcomes disagree with joint/T1 outcomes")
    if "gold_t1" in bank:
        require(np.array_equal(t1[:, 0], p1.argmax(1) == bank["gold_t1"]), "Native T1 outcome mismatch")
    if "features" in bank:
        require(bank["features"].shape == features.shape and np.allclose(bank["features"], features, atol=1e-7, rtol=0),
                "Features violate the native-probability feature specification")
        # Retain original floating-point values for frozen-fit parity after validation.
        features = bank["features"]
    require(np.isfinite(features).all(), "Nonfinite deployment features")
    result = dict(ids=ids, group=groups, features=features, joint=joint.astype(np.int8), t1=t1.astype(np.int8),
                  depth=depth.astype(np.int8), continue1=continue1, continue2=continue2,
                  corpus=corpus, seed=seed, threshold=threshold)
    # The frozen TAMA DP uses source order; external experiments sort IDs stably.
    if corpus != "tama":
        order = np.argsort(ids, kind="stable")
        for key in ("ids", "group", "features", "joint", "t1", "depth", "continue1", "continue2"):
            result[key] = result[key][order]
    metadata = {}
    for key in ("model", "checkpoint_sha256", "adapter_sha256"):
        if key in bank:
            require(bank[key].shape == (), f"{key} must be scalar")
            metadata[key] = bank[key].item()
    return result, metadata


def load_pair(dev_path, test_path, corpus, seed, threshold):
    dev, dev_meta = load_bank(dev_path, corpus, seed, "dev", threshold)
    test, test_meta = load_bank(test_path, corpus, seed, "test", threshold)
    require(not set(dev["ids"]) & set(test["ids"]), "Development/test IDs overlap")
    require(not set(dev["group"]) & set(test["group"]), "Development/test exact-text groups overlap")
    require(dev_meta == test_meta, "Development/test model or checkpoint metadata mismatch")
    return dev, test
