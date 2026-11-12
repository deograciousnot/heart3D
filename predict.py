"""
Run both trained models on one PTB-XL ECG and explain the answer, for the viewers.

    from predict import ai_reading, ai_reading_from_raw
    ai_reading(338)                    # PTB-XL ecg_id -> dict that goes into the viewer data file as "ai"
    ai_reading_from_raw(raw, fs=500)   # any 10 s of raw 12-lead signal (live_server.py uses this)

Models always see 10 s at 100 Hz, the same as in training: PTB-XL records use their 100 Hz version,
other signals are resampled to 100 Hz first.

Explanations ("what drove it") answer: how much does the probability drop if this part of the
heartbeat looked typical instead?
  - Feature model: reset a group of measurements (P wave and PR, QRS, ST segment, T wave, rate)
    to the training median, re-predict.
  - Neural network: flatten that part of every beat in the recording (P, QRS or ST-T) to the
    baseline, re-predict.
Both are simple "occlusion" tests; they show what the model relies on, not medical cause.
"""

import csv
import functools
import os
import pickle
import warnings

import numpy as np

from ecg_to_vcg import LEAD_NAMES, analyse_beats, clean_ecg, detect_phases, inverse_kors, read_record
from ptbxl_data import CLASS_NAMES, CLASSES, ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# Feature groups, keyed by the part of the beat they describe. "phase" matches the viewer's
# phase numbers (1 = P, 2 = QRS, 3 = T) so the 3D view can light up the right part of the loop.
def _is_p(n):
    return n in ("pr_ms", "p_ms", "p_missing", "p_peak_mv")


def _is_t(n):
    return n.startswith(("t_", "qrs_t_")) or n.endswith("_t_amp")  # qrs_t_angle compares QRS with T: T group


def _is_qrs(n):
    return (n.startswith(("qrs_", "area_")) or n.endswith(("_qrs_max", "_qrs_min"))) and not _is_t(n)


GROUPS = {
    "P wave and PR": {"phase": 1, "match": _is_p},
    "QRS": {"phase": 2, "match": _is_qrs},
    "ST segment": {"phase": 3, "match": lambda n: n.endswith("_st60")},
    "T wave": {"phase": 3, "match": _is_t},
    "Heart rate": {"phase": 0, "match": lambda n: n == "bpm"},
}


def feature_label(name, value):
    """Human-readable name + value for a feature, e.g. ('V2 ST level', '-0.12 mV')."""
    fixed = {
        "pr_ms": ("PR interval", "ms"), "p_ms": ("P wave duration", "ms"), "qrs_ms": ("QRS width", "ms"),
        "bpm": ("Heart rate", "bpm"), "qrs_t_angle": ("QRS-T angle", "deg"), "qrs_peak_mv": ("Peak QRS vector", "mV"),
        "p_peak_mv": ("P wave size", "mV"), "p_missing": ("No clear P wave", ""),
    }
    if name in fixed:
        label, unit = fixed[name]
    elif name.split("_")[0] in LEAD_NAMES:
        lead, rest = name.split("_", 1)
        label, unit = f"{lead} " + {"qrs_max": "R height", "qrs_min": "S / Q depth", "st60": "ST level",
                                    "t_amp": "T-wave height"}[rest], "mV"
    else:
        part = {"qrs_axis": "QRS axis", "t_axis": "T axis", "area": "QRS loop area", "qrs_mean": "Mean QRS vector",
                "qrs_init": "First 30 ms of QRS", "qrs_term": "Last 30 ms of QRS", "qrs_max": "Peak QRS direction",
                "t_mean": "Mean T vector", "t_max": "Peak T direction"}
        key = next((k for k in part if name.startswith(k)), None)
        label, unit = (f"{part[key]} ({name[len(key) + 1:]})" if key else name), ""
    if name == "p_missing":
        return label, "yes" if value else "no"
    return label, (f"{value:.0f} {unit}".strip() if unit in ("ms", "bpm", "deg") else f"{value:.2f} {unit}".strip())


def _record_info(ecg_id):
    with open(os.path.join(ROOT, "ptbxl_database.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["ecg_id"]) == ecg_id:
                return row
    raise KeyError(ecg_id)


def _truth(row):
    import ast
    with open(os.path.join(ROOT, "scp_statements.csv"), encoding="utf-8") as f:
        scp = {r[""]: r for r in csv.DictReader(f)}
    codes = ast.literal_eval(row["scp_codes"])
    return sorted({scp[c]["diagnostic_class"] for c in codes if c in scp and scp[c]["diagnostic_class"]},
                  key=CLASSES.index)


@functools.lru_cache(maxsize=None)
def _load(name):
    """Trained model files, loaded once per process (a live stream asks every few seconds)."""
    path = os.path.join(MODELS, name)
    if name.endswith(".pt"):
        import torch
        from train_cnn import make_model
        state = torch.load(path, map_location="cpu")
        model = make_model()
        model.load_state_dict(state["model"])
        model.eval()
        return state, model
    with open(path, "rb") as f:
        return pickle.load(f)


def _ordered(raw, names):
    """Channels -> LEAD_NAMES order (I, II, III, aVR, aVL, aVF, V1-V6)."""
    names = [n.upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for n in names]
    return np.vstack([raw[names.index(n)] for n in LEAD_NAMES])


def _clean(raw, fs):
    """Band-pass like everything else -> (leads in LEAD_NAMES order, {name: lead})."""
    leads = clean_ecg(raw, fs)
    return leads, dict(zip(LEAD_NAMES, leads))


def _feature_reading(raw, fs):
    from train_features import beat_features
    fm = _load("feature_model.pkl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        leads, by_name = _clean(raw, fs)
        f = beat_features(analyse_beats(inverse_kors(by_name), leads, fs), fs)
    x = np.array([[f[n] for n in fm["names"]]], dtype=np.float32)
    prob = lambda X: np.array([m.predict_proba(X)[:, 1] for m in fm["models"]]).T  # (rows, classes)
    p = prob(x)[0]

    # Group occlusion: one row per group, that group reset to the training median.
    group_rows = []
    for g in GROUPS.values():
        xr = x.copy()
        idx = [i for i, n in enumerate(fm["names"]) if g["match"](n)]
        xr[0, idx] = fm["medians"][idx]
        group_rows.append(xr[0])
    pg = prob(np.array(group_rows))
    # Single-feature occlusion for the top features.
    singles = np.repeat(x, len(fm["names"]), axis=0)
    singles[np.arange(len(fm["names"])), np.arange(len(fm["names"]))] = fm["medians"]
    ps = prob(singles)

    out = {"probs": {}, "drivers": {}}
    for c, cls in enumerate(CLASSES):
        out["probs"][cls] = round(float(p[c]), 4)
        drops = {name: round(float(p[c] - pg[k, c]), 4) for k, name in enumerate(GROUPS)}
        top = np.argsort(-(p[c] - ps[:, c]))[:3]
        out["drivers"][cls] = {
            "groups": drops,
            "features": [dict(zip(("label", "value"), feature_label(fm["names"][i], float(x[0, i]))),
                              drop=round(float(p[c] - ps[i, c]), 4)) for i in top if p[c] - ps[i, c] > 0.005],
        }
    out["thresholds"] = {cls: round(float(t), 4) for cls, t in zip(CLASSES, fm["thresholds"])}
    out["_features"] = dict(zip(fm["names"], x[0].tolist()))  # reused by the MI location model, then dropped
    return out


def _location_reading(features):
    """Which wall an MI is on, from the same measurements. Only meaningful when there is an MI."""
    lm = _load("mi_location.pkl")
    x = np.array([[features[n] for n in lm["names"]]], dtype=np.float32)
    return {
        "probs": {w: round(float(m.predict_proba(x)[0, 1]), 4) for w, m in zip(lm["walls"], lm["models"])},
        "thresholds": {w: round(float(t), 4) for w, t in zip(lm["walls"], lm["thresholds"])},
        "testAuc": _results("mi_location_results.json", per="walls"),
    }


def _cnn_reading(raw, fs):
    import torch
    state, model = _load("cnn.pt")
    leads, by_name = _clean(raw, fs)
    x = leads.astype(np.float32)

    # Phase occlusion: flatten P, QRS or ST-T in every beat of the recording.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = inverse_kors(by_name)
        phase = detect_phases(h.shape[1], fs, analyse_beats(h, leads, fs))
    variants = [x]
    for k in (1, 2, 3):
        xv = x.copy()
        xv[:, phase == k] = 0.0
        variants.append(xv)
    xb = (torch.from_numpy(np.stack(variants)) - state["mean"]) / state["std"]
    with torch.no_grad():
        p = torch.sigmoid(model(xb)).numpy()
    names = {1: "P wave and PR", 2: "QRS", 3: "ST-T"}
    return {
        "probs": {cls: round(float(p[0, c]), 4) for c, cls in enumerate(CLASSES)},
        "drivers": {cls: {"groups": {names[k]: round(float(p[0, c] - p[k, c]), 4) for k in (1, 2, 3)}}
                    for c, cls in enumerate(CLASSES)},
        "thresholds": {cls: round(float(t), 4) for cls, t in zip(CLASSES, state["thresholds"])},
    }


def _results(name, per=None):
    """Test AUC from a saved results file: the macro value, or {item: auc} for per='walls'."""
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        return None
    import json
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    if per:
        return {k: round(v["test_auc"], 3) for k, v in r[per].items() if "test_auc" in v}
    return r.get("macro_test_auc")


def _model_readings(raw100, fs=100):
    """Both models (+ MI location) on raw 12-lead signal at 100 Hz in LEAD_NAMES order."""
    reading = {"classNames": CLASS_NAMES, "models": {}}
    if os.path.exists(os.path.join(MODELS, "feature_model.pkl")):
        fr = _feature_reading(raw100, fs)
        features = fr.pop("_features")
        reading["models"]["features"] = {"name": "Measurement model", "testAuc": _results("feature_model_results.json"),
                                         "groupPhase": {g: v["phase"] for g, v in GROUPS.items()}, **fr}
        if os.path.exists(os.path.join(MODELS, "mi_location.pkl")):
            reading["miLocation"] = _location_reading(features)
    if os.path.exists(os.path.join(MODELS, "cnn.pt")):
        reading["models"]["cnn"] = {"name": "Neural network", "testAuc": _results("cnn_results.json"),
                                    "groupPhase": {"P wave and PR": 1, "QRS": 2, "ST-T": 3},
                                    **_cnn_reading(raw100, fs)}
    return reading


def record_meta(ecg_id):
    """What PTB-XL says about a record: its fold, whether the models saw it, the cardiologist's classes."""
    row = _record_info(ecg_id)
    fold = int(row["strat_fold"])
    return {"ecgId": ecg_id, "fold": fold,
            "seen": "training" if fold <= 8 else ("validation" if fold == 9 else "test"),
            "truth": _truth(row), "filename_lr": row["filename_lr"], "filename_hr": row["filename_hr"]}


def _public(meta):
    return {k: v for k, v in meta.items() if not k.startswith("filename")}


def ai_reading(ecg_id):
    meta = record_meta(ecg_id)
    raw, fs, names = read_record(os.path.join(ROOT, meta["filename_lr"]))
    return {**_public(meta), **_model_readings(_ordered(raw, names), fs)}


def ai_reading_from_raw(raw, fs, meta=None):
    """
    raw: (12, T) unfiltered 12-lead signal in mV, LEAD_NAMES order, ideally exactly 10 s.
    Resampled to 100 Hz, then read exactly like a PTB-XL record. meta (e.g. record_meta) is merged in.
    """
    from math import gcd
    from scipy.signal import resample_poly
    fs = int(round(fs))
    raw = np.asarray(raw, dtype=float)
    if fs != 100:
        g = gcd(100, fs)
        raw = resample_poly(raw, 100 // g, fs // g, axis=-1)
    reading = _model_readings(raw, 100)
    return {**(_public(meta) if meta else {"truth": [], "seen": "live"}), **reading}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(ai_reading(int(sys.argv[1]) if len(sys.argv) > 1 else 338), indent=1))
