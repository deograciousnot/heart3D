"""
MI location model: given an ECG with a heart attack (MI), which wall of the left ventricle is it on?

PTB-XL's MI codes name the location; they're grouped into three walls the 3D heart can shade:
    anterior   front wall and septum   AMI, ASMI, ALMI, INJAS, INJAL
    inferior   bottom (and back) wall  IMI, ILMI, IPLMI, IPMI, PMI, INJIN, INJIL
    lateral    left side wall          LMI, ALMI, ILMI, IPLMI, INJAL, INJLA, INJIL
(Posterior alone has ~100 ECGs, too few to learn, so it rides with inferior.) One MI can cover
several walls (ILMI = inferior + lateral), so it's multi-label.

Trained only on ECGs that have an MI code, with the same 81 median-beat measurements as
train_features.py (run that first: it builds the feature cache). Same splits: folds 1-8 train,
fold 9 picks thresholds, fold 10 is scored once with --test.

    python train_location.py          # train + validation
    python train_location.py --test   # also score the test fold (once)
"""

import argparse
import ast
import csv
import json
import os
import pickle

import numpy as np

from ptbxl_data import ROOT, load_table, split_masks
from train_features import CACHE, MODELS, best_threshold, sens_spec

WALLS = {
    "anterior": {"AMI", "ASMI", "ALMI", "INJAS", "INJAL"},
    "inferior": {"IMI", "ILMI", "IPLMI", "IPMI", "PMI", "INJIN", "INJIL"},
    "lateral": {"LMI", "ALMI", "ILMI", "IPLMI", "INJAL", "INJLA", "INJIL"},
}
WALL_NAMES = {"anterior": "Front wall and septum", "inferior": "Bottom wall", "lateral": "Left side wall"}


def wall_labels(ids):
    """(N, 3) 0/1 matrix in WALLS order; rows of ECGs without an MI code are all zero."""
    codes = {}
    with open(os.path.join(ROOT, "ptbxl_database.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[int(row["ecg_id"])] = set(ast.literal_eval(row["scp_codes"]))
    return np.array([[int(bool(codes[i] & w)) for w in WALLS.values()] for i in ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="also score the test fold (do this once)")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    ids, _, folds, _ = load_table(hz=100)
    cached = np.load(CACHE, allow_pickle=False)
    assert np.array_equal(cached["ids"], ids), "feature cache is stale: run train_features.py --rebuild"
    X, names = cached["X"], list(cached["names"])
    W = wall_labels(ids)
    has_mi = W.any(axis=1)
    train, val, test = (m & has_mi for m in split_masks(folds))
    print(f"ECGs with an MI location: train {train.sum()}, validation {val.sum()}, test {test.sum()}")

    results, models, thresholds = {"walls": {}}, [], []
    for k, wall in enumerate(WALLS):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, class_weight="balanced",
                                               early_stopping=False, random_state=0)
        model.fit(X[train], W[train, k])
        p_val = model.predict_proba(X[val])[:, 1]
        thr = best_threshold(W[val, k], p_val)
        r = results["walls"][wall] = {"name": WALL_NAMES[wall], "val_auc": float(roc_auc_score(W[val, k], p_val)),
                                      "threshold": thr, "train_positives": int(W[train, k].sum())}
        line = f"  {wall:9} val AUC {r['val_auc']:.3f}"
        if args.test:
            p_test = model.predict_proba(X[test])[:, 1]
            sens, spec = sens_spec(W[test, k], p_test, thr)
            r.update(test_auc=float(roc_auc_score(W[test, k], p_test)), test_sensitivity=sens, test_specificity=spec,
                     test_positives=int(W[test, k].sum()))
            line += f" | test AUC {r['test_auc']:.3f}, sensitivity {sens:.2f}, specificity {spec:.2f}"
        print(line, flush=True)
        models.append(model)
        thresholds.append(thr)

    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, "mi_location.pkl"), "wb") as f:
        pickle.dump({"names": names, "walls": list(WALLS), "models": models, "thresholds": thresholds}, f)
    if args.test:
        with open(os.path.join(MODELS, "mi_location_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Saved models/mi_location_results.json")
    else:
        print("Test fold not scored. Run with --test once the model is final.")


if __name__ == "__main__":
    main()
