"""
Baseline 1: an explainable classifier built on the heart-vector measurements.

Every ECG goes through the same pipeline as the viewers (filter -> Kors -> median beat),
then ~90 measurements of that beat become the features:
  - heart vector: QRS width, axes, QRS-T angle, mean / initial / terminal / peak QRS vectors, T vector, loop areas
  - per lead: QRS max and min, ST level 60 ms after the QRS, T amplitude
A gradient-boosted tree model per class predicts NORM / MI / STTC / CD / HYP.

    python train_features.py            # extract features (cached) + train + evaluate
    python train_features.py --rebuild  # re-extract features

Train on folds 1-8, pick decision thresholds on fold 9, score fold 10 once.
Threads are capped (--threads) to keep the laptop cool.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np

from ecg_to_vcg import LEAD_NAMES, analyse_beats, inverse_kors, load_wfdb
from ptbxl_data import CLASS_NAMES, CLASSES, ROOT, load_table, split_masks

CACHE = os.path.join(ROOT, "features_100hz.npz")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def shoelace(a, b):
    """Signed area of the loop traced by (a, b): the loop's size and its turning direction in one number."""
    return 0.5 * float(np.sum(a[:-1] * b[1:] - a[1:] * b[:-1]))


def beat_features(beat, fs):
    ms = lambda x: max(1, int(round(x * fs / 1000)))
    v, L, ph = beat["vector"], beat["leads"], beat["phase"]
    on, off, n = beat["onset"], beat["offset"], beat["phase"].size
    s = beat["stats"]
    qrs = v[:, on:off]
    t_mask = ph == 3

    f = {
        "qrs_ms": s["qrsMs"], "bpm": s["bpm"], "qrs_t_angle": s["qrsTAngleDeg"], "qrs_peak_mv": s["qrsPeakMv"],
        "qrs_axis_sin": np.sin(np.radians(s["qrsAxisDeg"])), "qrs_axis_cos": np.cos(np.radians(s["qrsAxisDeg"])),
        "t_axis_sin": np.sin(np.radians(s["tAxisDeg"])), "t_axis_cos": np.cos(np.radians(s["tAxisDeg"])),
        "area_frontal": shoelace(qrs[0], qrs[1]), "area_horizontal": shoelace(qrs[0], qrs[2]),
        "area_sagittal": shoelace(qrs[2], qrs[1]),
        "p_peak_mv": float(np.linalg.norm(v[:, ph == 1], axis=0).max()) if (ph == 1).any() else np.nan,
    }
    t_vec = v[:, t_mask] if t_mask.any() else np.full((3, 1), np.nan)
    vectors = {
        "qrs_mean": qrs.mean(axis=1),
        "qrs_init": v[:, on:on + ms(30)].mean(axis=1),          # first 30 ms: septum, lost in MI / LBBB
        "qrs_term": v[:, max(on, off - ms(30)):off].mean(axis=1),  # last 30 ms: late tail of RBBB
        "qrs_max": qrs[:, np.argmax(np.linalg.norm(qrs, axis=0))],
        "t_mean": t_vec.mean(axis=1),
        "t_max": t_vec[:, np.nanargmax(np.linalg.norm(t_vec, axis=0))] if t_mask.any() else t_vec[:, 0],
    }
    for name, vec in vectors.items():
        for axis, value in zip("xyz", vec):
            f[f"{name}_{axis}"] = float(value)

    st_index = min(n - 1, off + ms(60))
    for k, lead in enumerate(LEAD_NAMES):
        f[f"{lead}_qrs_max"] = float(L[k, on:off].max())
        f[f"{lead}_qrs_min"] = float(L[k, on:off].min())
        f[f"{lead}_st60"] = float(L[k, st_index])
        if t_mask.any():
            t = L[k, t_mask]
            f[f"{lead}_t_amp"] = float(t[np.argmax(np.abs(t))])
        else:
            f[f"{lead}_t_amp"] = np.nan
    return f


def extract_all(paths):
    """Features for every record; a record the pipeline can't handle gets NaNs (the model copes with those)."""
    rows, failed, start = [], 0, time.time()
    names = None
    for i, path in enumerate(paths):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, leads, by_name, fs = load_wfdb(path)
                beat = analyse_beats(inverse_kors(by_name), leads, fs)
                f = beat_features(beat, fs)
            names = names or list(f)
            rows.append([f[k] for k in names])
        except Exception:
            failed += 1
            rows.append(None)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(paths)} ECGs, {failed} failed, {time.time() - start:.0f} s", flush=True)
    X = np.array([r if r is not None else [np.nan] * len(names) for r in rows], dtype=np.float32)
    print(f"Features: {X.shape[0]} ECGs x {X.shape[1]} features, {failed} ECGs could not be analysed")
    return X, names


def best_threshold(y, p):
    """Threshold on the validation set that maximises sensitivity + specificity (Youden's J)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def sens_spec(y, p, threshold):
    pred = p >= threshold
    return float((pred & (y == 1)).sum() / max(1, (y == 1).sum())), float((~pred & (y == 0)).sum() / max(1, (y == 0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="re-extract features even if cached")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for training (default 2, keeps it cool)")
    parser.add_argument("--test", action="store_true",
                        help="also score the test fold. Leave off while tuning, so choices are made on validation only")
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    ids, paths, folds, Y = load_table(hz=100)
    if os.path.exists(CACHE) and not args.rebuild:
        cached = np.load(CACHE, allow_pickle=False)
        assert np.array_equal(cached["ids"], ids), "cache doesn't match the label table; use --rebuild"
        X, names = cached["X"], list(cached["names"])
        print(f"Loaded cached features: {X.shape}")
    else:
        print(f"Extracting features from {len(paths)} ECGs (one CPU thread)...")
        X, names = extract_all(paths)
        np.savez_compressed(CACHE, ids=ids, X=X, names=np.array(names))

    train, val, test = split_masks(folds)
    print(f"Train {train.sum()}, validation {val.sum()}, test {test.sum()} ECGs")

    results = {"model": "HistGradientBoosting on heart-vector + per-lead median-beat features",
               "features": len(names), "train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum()),
               "classes": {}}
    start = time.time()
    for c, cls in enumerate(CLASSES):
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, class_weight="balanced",
                                               early_stopping=False, random_state=0)
        model.fit(X[train], Y[train, c])
        p_val = model.predict_proba(X[val])[:, 1]
        thr = best_threshold(Y[val, c], p_val)
        r = results["classes"][cls] = {"name": CLASS_NAMES[cls], "val_auc": float(roc_auc_score(Y[val, c], p_val)),
                                       "threshold": thr}
        line = f"  {cls:5} val AUC {r['val_auc']:.3f}"
        if args.test:
            p_test = model.predict_proba(X[test])[:, 1]
            sens, spec = sens_spec(Y[test, c], p_test, thr)
            r.update(test_auc=float(roc_auc_score(Y[test, c], p_test)), test_sensitivity=sens,
                     test_specificity=spec, test_positives=int(Y[test, c].sum()))
            line += (f" | test AUC {r['test_auc']:.3f}, sensitivity {sens:.2f}, specificity {spec:.2f} "
                     f"({r['test_positives']} positives)")
        print(line, flush=True)
    results["macro_val_auc"] = float(np.mean([r["val_auc"] for r in results["classes"].values()]))
    summary = f"Macro AUC: validation {results['macro_val_auc']:.3f}"
    if args.test:
        results["macro_test_auc"] = float(np.mean([r["test_auc"] for r in results["classes"].values()]))
        summary += f", test {results['macro_test_auc']:.3f}"
    print(f"{summary} ({time.time() - start:.0f} s training)")
    if not args.test:
        print("Test fold not scored. Run with --test once the model is final.")
        return

    os.makedirs(MODELS, exist_ok=True)
    with open(os.path.join(MODELS, "feature_model_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {os.path.join('models', 'feature_model_results.json')}")


if __name__ == "__main__":
    main()
