"""
ECG -> 3D heart vector (vectorcardiogram) pipeline.

Coordinate system (Frank convention):
    x = towards the patient's LEFT
    y = towards the FEET
    z = towards the BACK   (so the front of the chest is -z)

The core idea:
    every lead = dot(heart_vector(t), lead_direction)
    so 12 leads = L @ h,   with L a (12, 3) matrix of lead directions
    and recovering h from the leads is a least-squares problem.

Usage:
    python ecg_to_vcg.py                       # synthetic ECG (no download needed)
    python ecg_to_vcg.py --wfdb path/to/record # real 12-lead record, e.g. from PTB-XL (needs `pip install wfdb`)
"""

import argparse
import json
import os

import numpy as np

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def frontal(deg):
    """Limb lead direction in the frontal plane (0 deg = left, +90 deg = feet)."""
    r = np.radians(deg)
    return unit([np.cos(r), np.sin(r), 0.0])


def horizontal(deg):
    """Chest lead direction in the horizontal plane (0 deg = left side, +90 deg = straight to the front)."""
    r = np.radians(deg)
    return unit([np.cos(r), 0.0, -np.sin(r)])


# Idealised lead directions: each row is the "camera" direction for one lead.
LEAD_DIRECTIONS = np.array([
    frontal(0),       # I
    frontal(60),      # II
    frontal(120),     # III
    frontal(-150),    # aVR
    frontal(-30),     # aVL
    frontal(90),      # aVF
    horizontal(115),  # V1
    horizontal(90),   # V2
    horizontal(75),   # V3
    horizontal(60),   # V4
    horizontal(30),   # V5
    horizontal(0),    # V6
])

# Kors regression matrix (Kors et al., 1990): maps real measured leads
# [I, II, V1..V6] to Frank X, Y, Z. Calibrated on real torsos, so use it for real data.
KORS = np.array([
    # I      II     V1     V2     V3     V4     V5     V6
    [0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54],   # X
    [-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13],  # Y
    [0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31],  # Z
])
KORS_INPUTS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def gaussian(t, centre, width):
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def synthetic_heart_vector(fs=500, seconds=10.0, bpm=72, seed=0):
    """Build a plausible 3D heart vector h(t) for a normal heart: P, QRS and T loops."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    h = np.zeros((3, t.size))
    phase = np.zeros(t.size, dtype=int)  # 0 = baseline, 1 = P, 2 = QRS, 3 = T
    rr = 60.0 / bpm

    for beat_start in np.arange(0.1, seconds, rr):
        tb = t - beat_start

        # P wave: small, slow, pointing down-left (atria firing from the SA node).
        p = 0.15 * gaussian(tb, 0.08, 0.025)
        h += np.outer(unit([0.6, 0.8, -0.1]), p)
        phase[(tb > 0.03) & (tb < 0.13)] = 1

        # QRS: the big loop. Direction sweeps from right/front (septum) through
        # down-left (main ventricle mass) to up/back (last bits to fire).
        s = np.clip((tb - 0.16) / 0.09, 0, 1)
        in_qrs = (tb >= 0.16) & (tb <= 0.25)
        angle = np.radians(-40 + 200 * s)
        envelope = np.sin(np.pi * s) ** 2 * 1.6
        qrs = np.vstack([
            np.cos(angle),
            np.sin(angle) * 0.9,
            -0.6 * np.cos(np.pi * s),
        ]) * envelope
        h += np.where(in_qrs, qrs, 0)
        phase[in_qrs] = 2

        # T wave: ventricles resetting, roughly the same direction as the main QRS.
        tw = 0.35 * gaussian(tb, 0.45, 0.06)
        h += np.outer(unit([0.7, 0.6, -0.2]), tw)
        phase[(tb > 0.33) & (tb < 0.57) & (phase == 0)] = 3

    h += rng.normal(0, 0.008, h.shape)  # a little measurement noise
    return t, h, phase


def forward(h):
    """Heart vector (3, T) -> 12-lead ECG (12, T). Each lead is a dot product."""
    return LEAD_DIRECTIONS @ h


def inverse_lstsq(leads):
    """12-lead ECG (12, T) -> heart vector (3, T) by least squares on the ideal lead directions."""
    return np.linalg.lstsq(LEAD_DIRECTIONS, leads, rcond=None)[0]


def inverse_kors(leads_by_name):
    """Real ECG -> heart vector using the Kors matrix."""
    stacked = np.vstack([leads_by_name[name] for name in KORS_INPUTS])
    return KORS @ stacked


def clean_ecg(signals, fs, low_hz=0.5, high_hz=40.0):
    """Monitor-style band-pass: removes slow baseline drift (breathing) and fast noise (muscle, mains hum)."""
    from scipy.signal import butter, filtfilt

    b, a = butter(2, [low_hz / (fs / 2), high_hz / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, signals, axis=-1)


BEAT_BEFORE_MS, BEAT_AFTER_MS = 450, 550  # median-beat window around the R peak (450 ms: room for a long PR)


def find_r_peaks(h, fs):
    """QRS peaks = big maxima of |h|, at least 300 ms apart."""
    from scipy.signal import find_peaks

    magnitude = np.linalg.norm(h, axis=0)
    peaks, _ = find_peaks(magnitude, height=0.5 * magnitude.max(), distance=int(0.3 * fs))
    return peaks


def median_beat(h, leads, peaks, fs):
    """
    Average beat: cut a window around every R peak, line the beats up, take the median.
    The median ignores one-off noise and odd beats, which is why ECG machines do the same.

    Beats are first cut at the raw peaks, then each one is nudged (up to +-40 ms) to best
    match the first-pass template. That matters for wide or notched QRS (bundle branch
    blocks), where the tallest point can jump between two humps from beat to beat.
    """
    ms = lambda x: int(round(x * fs / 1000))
    before, after = ms(BEAT_BEFORE_MS), ms(BEAT_AFTER_MS)
    peaks = np.array([p for p in peaks if p - before - ms(40) >= 0 and p + after + ms(40) <= h.shape[1]])
    if len(peaks) == 0:
        raise ValueError("no complete beats found")

    cut = lambda x, p: x[:, p - before:p + after]
    template = np.median([cut(h, p) for p in peaks], axis=0)
    aligned = []
    for p in peaks:
        shifts = range(-ms(40), ms(40) + 1)
        best = max(shifts, key=lambda s: np.sum(cut(h, p + s) * template))
        aligned.append(p + best)
    aligned = np.array(aligned)
    vec = np.median([cut(h, p) for p in aligned], axis=0)
    beat_leads = np.median([cut(leads, p) for p in aligned], axis=0)
    return vec, beat_leads, aligned, before


def qrs_bounds(vec, r_index, fs):
    """
    QRS onset / offset from the spatial velocity |dh/dt|: the ventricles firing is by far
    the fastest-moving part of the beat. Start at the fastest point and grow outwards while
    the tip is still moving fast (>10% of top speed), tolerating dips shorter than 10 ms,
    such as the moment the loop turns around at its tip.
    """
    ms = lambda x: int(round(x * fs / 1000))
    velocity = np.linalg.norm(np.diff(vec, axis=1), axis=0)
    lo, hi = max(0, r_index - ms(120)), min(velocity.size, r_index + ms(150))
    centre = lo + int(np.argmax(velocity[lo:hi]))
    fast = velocity > 0.1 * velocity[centre]
    gap = max(1, ms(10))
    onset, offset = centre, centre
    while onset > lo and fast[max(0, onset - gap):onset].any():
        onset -= 1
    while offset < hi and fast[offset:offset + gap].any():
        offset += 1
    return onset, offset


def p_wave_bounds(vec, qrs_onset, fs, rr):
    """
    P wave onset / offset on the baseline-corrected median beat, or None when there's no clear P
    (e.g. atrial fibrillation).

    The search covers up to 400 ms before the QRS (never more than 45% of the beat-to-beat interval).
    The tail of the previous T wave can reach into that stretch, so the P wave is the strong bump in
    |h| CLOSEST to the QRS, not simply the biggest. Its edges are where |h| drops below a quarter of
    the bump, or where it starts rising again (the dip that separates P from the T before it).
    """
    from scipy.signal import find_peaks

    ms = lambda x: int(round(x * fs / 1000))
    reach = min(ms(400), int(0.45 * rr)) if np.isfinite(rr) else ms(400)
    lo, hi = max(0, qrs_onset - reach), qrs_onset - ms(40)
    if hi - lo < ms(40):
        return None
    # Smooth over 10 ms first, so high-resolution noise (500 Hz) doesn't split one P wave into several bumps.
    # At 100 Hz that is a single sample: no change.
    width = max(1, ms(10))
    magnitude = np.convolve(np.linalg.norm(vec, axis=0), np.ones(width) / width, mode="same")
    seg = magnitude[lo:hi]
    bumps, _ = find_peaks(np.concatenate([[0.0], seg, [0.0]]))
    # 0.05 mV floor: real P waves peak around 0.13 mV (10th percentile 0.08) in normal PTB-XL ECGs,
    # while what's left of fibrillation waves in an averaged AF beat peaks around 0.04.
    bumps = [b - 1 for b in bumps if seg[b - 1] >= max(0.05, 0.5 * seg.max())]
    if not bumps:
        return None
    peak = lo + bumps[-1]
    top = magnitude[peak]

    def edge(step, limit):
        """Walk away from the peak; stop below a quarter of it, or at a dip it clearly climbs out of."""
        i, best = peak, peak
        while i != limit:
            i += step
            if magnitude[i] < magnitude[best]:
                best = i
            if magnitude[i] < 0.25 * top:
                return i
            if magnitude[i] > magnitude[best] + 0.2 * top:  # rising again: that was the dip
                return best
        # Reached the edge of the search: accept the lowest point if it's clearly below the peak.
        return best if magnitude[best] < 0.5 * top else None

    on, off = edge(-1, lo), edge(+1, qrs_onset - 1)
    if on is None or off is None:
        return None
    return on, off + 1


def phase_windows(onset, offset, fs, p=None):
    """(label, start, end) for P / QRS / T relative to the measured QRS; P is measured when available."""
    ms = lambda x: int(round(x * fs / 1000))
    p_start, p_end = p if p else (onset - ms(180), onset - ms(50))
    return [
        (1, p_start, p_end),                    # P wave
        (2, onset, offset),                     # QRS
        (3, offset + ms(60), onset + ms(440)),  # T wave
    ]


def label_phases(n, fs, centres, onset, offset, p=None):
    """Phase labels for an n-sample signal; onset/offset (and p) are relative to each centre (R peak)."""
    phase = np.zeros(n, dtype=int)
    for c in centres:
        pc = (c + p[0], c + p[1]) if p else None
        for label, a, b in phase_windows(c + onset, c + offset, fs, pc):
            phase[max(0, a):max(0, b)] = label
    return phase


def angle_deg(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def analyse_beats(h, leads, fs):
    """
    Median beat + the classic vectorcardiogram measurements:
      - QRS duration (ms): how long the ventricles take to fire. Normal < ~110 ms; bundle branch block >= 120.
      - QRS / T frontal axis (deg): direction of the mean vector in the front view (0 = left, +90 = feet).
      - QRS-T angle (deg): 3D angle between the mean QRS and mean T vectors. Normally small;
        large when depolarisation or repolarisation is abnormal.
    """
    peaks = find_r_peaks(h, fs)
    vec, beat_leads, aligned, r_index = median_beat(h, leads, peaks, fs)
    onset, offset = qrs_bounds(vec, r_index, fs)

    # Baseline: the flat PR segment just before the QRS is "zero" for both the loop and the leads.
    ms = lambda x: int(round(x * fs / 1000))
    base = slice(max(0, onset - ms(30)), max(1, onset - ms(10)))
    vec = vec - vec[:, base].mean(axis=1, keepdims=True)
    beat_leads = beat_leads - beat_leads[:, base].mean(axis=1, keepdims=True)

    rr = float(np.median(np.diff(aligned))) if len(aligned) > 1 else float("nan")
    p_wave = p_wave_bounds(vec, onset, fs, rr)
    phase = label_phases(vec.shape[1], fs, [0], onset, offset, p_wave)
    mean_qrs = vec[:, phase == 2].mean(axis=1)
    mean_t = vec[:, phase == 3].mean(axis=1)
    frontal_axis = lambda v: float(np.degrees(np.arctan2(v[1], v[0])))
    stats = {
        "beats": int(len(aligned)),
        "bpm": 60 * fs / rr,
        "prMs": float((onset - p_wave[0]) * 1000 / fs) if p_wave else None,  # P start -> QRS start
        "pMs": float((p_wave[1] - p_wave[0]) * 1000 / fs) if p_wave else None,  # P wave duration
        "qrsMs": float((offset - onset) * 1000 / fs),
        "qrsAxisDeg": frontal_axis(mean_qrs),
        "tAxisDeg": frontal_axis(mean_t),
        "qrsTAngleDeg": angle_deg(mean_qrs, mean_t),
        "qrsPeakMv": float(np.linalg.norm(vec[:, phase == 2], axis=0).max()),
    }
    # Beats too close to the edges were left out of the median; keep them for labelling the recording.
    edge = [p for p in peaks if np.min(np.abs(aligned - p)) > ms(200)]
    return {"vector": vec, "leads": beat_leads, "phase": phase, "rIndex": r_index,
            "onset": onset, "offset": offset, "pWave": p_wave,
            "peaks": np.sort(np.concatenate([aligned, edge])).astype(int), "stats": stats}


def detect_phases(n, fs, beat):
    """P / QRS / T labels for the whole recording: the median beat's P and QRS timing, applied to every beat."""
    r = beat["rIndex"]
    p = (beat["pWave"][0] - r, beat["pWave"][1] - r) if beat["pWave"] else None
    return label_phases(n, fs, beat["peaks"], beat["onset"] - r, beat["offset"] - r, p)


def read_record(path):
    """
    Read a WFDB record -> (signals (channels, T) in mV, fs, channel names).

    PTB-XL stores every record the same simple way (one .dat file, 16-bit little-endian integers,
    channels interleaved), so that case is read directly: about 10x faster than the wfdb library,
    which matters when reading all 21,799 records. Anything else falls back to wfdb.
    """
    with open(path + ".hea", encoding="utf-8") as f:
        lines = [ln.split() for ln in f if ln.strip() and not ln.startswith("#")]
    n_sig, fs = int(lines[0][1]), float(lines[0][2])
    specs = lines[1:1 + n_sig]
    simple = all(s[1] == "16" and s[0] == specs[0][0] and "(" in s[2] for s in specs)
    if not simple:
        import wfdb
        record = wfdb.rdrecord(path)
        return record.p_signal.T, record.fs, record.sig_name
    gain = np.array([float(s[2].split("(")[0]) for s in specs])
    baseline = np.array([float(s[2].split("(")[1].split(")")[0]) for s in specs])
    raw = np.fromfile(os.path.join(os.path.dirname(path), specs[0][0]), dtype="<i2").reshape(-1, n_sig).T
    return (raw - baseline[:, None]) / gain[:, None], fs, [s[8] for s in specs]


def load_wfdb(path):
    raw, fs, sig_names = read_record(path)
    names = [n.upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for n in sig_names]
    signals = clean_ecg(raw, fs)
    by_name = {n: signals[i] for i, n in enumerate(names)}
    leads = np.vstack([by_name[n] for n in LEAD_NAMES])
    t = np.arange(signals.shape[1]) / fs
    return t, leads, by_name, fs


def beat_json(beat, step=1):
    """The averaged beat as the viewers read it (every `step`-th sample), R peak at rIndex."""
    return {
        "vector": np.round(beat["vector"][:, ::step], 4).tolist(),
        "leads": np.round(beat["leads"][:, ::step], 4).tolist(),
        "phase": beat["phase"][::step].tolist(),
        "rIndex": beat["rIndex"] // step,
        "stats": {k: (round(v, 3) if v is not None else None) for k, v in beat["stats"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wfdb", help="path to a WFDB record (without extension), e.g. PTB-XL records500/00000/00001_hr")
    parser.add_argument("--out", default="ecg_data.js", help="output file the viewer loads")
    parser.add_argument("--label", help="short diagnosis label shown in the viewers, e.g. 'Left bundle branch block'")
    parser.add_argument("--ai", action="store_true",
                        help="add both trained models' reading (PTB-XL records only; needs models/ and ptbxl/records100/)")
    args = parser.parse_args()

    if args.wfdb:
        t, leads, by_name, fs = load_wfdb(args.wfdb)
        h = inverse_kors(by_name)
        beat = analyse_beats(h, leads, fs)
        phase = detect_phases(h.shape[1], fs, beat)
        bpm = beat["stats"]["bpm"]
        print(f"Real ECG: {len(beat['peaks'])} beats detected, ~{bpm:.0f} bpm")
        source = f"Real ECG, record {args.wfdb} (Kors reconstruction, ~{bpm:.0f} bpm)"
    else:
        fs = 500
        t, h_true, phase = synthetic_heart_vector(fs=fs)
        leads = forward(h_true)
        h = inverse_lstsq(leads)
        beat = analyse_beats(h, leads, fs)
        error = np.abs(h - h_true).max()
        print(f"Synthetic ECG: {leads.shape[0]} leads x {leads.shape[1]} samples")
        print(f"Reconstruction max error vs true heart vector: {error:.2e} mV")
        source = "Synthetic normal heart (least-squares reconstruction)"

    s = beat["stats"]
    pr = f"{s['prMs']:.0f} ms" if s["prMs"] is not None else "no clear P wave"
    print(f"Median of {s['beats']} beats: PR {pr}, QRS {s['qrsMs']:.0f} ms, QRS axis {s['qrsAxisDeg']:.0f} deg, "
          f"T axis {s['tAxisDeg']:.0f} deg, QRS-T angle {s['qrsTAngleDeg']:.0f} deg, peak {s['qrsPeakMv']:.2f} mV")

    # Downsample to 250 Hz for the browser; plenty for drawing.
    step = max(1, int(round(fs / 250)))
    data = {
        "source": source,
        "label": args.label or ("Normal (synthetic)" if not args.wfdb else None),
        "fs": fs / step,
        "leadNames": LEAD_NAMES,
        "leads": np.round(leads[:, ::step], 4).tolist(),
        "vector": np.round(h[:, ::step], 4).tolist(),
        "phase": phase[::step].tolist(),
        "leadDirections": np.round(LEAD_DIRECTIONS, 4).tolist(),
        # One averaged heartbeat, R peak at index rIndex. Used by compare.html.
        "beat": beat_json(beat, step),
    }
    if args.ai and args.wfdb:
        import re
        from predict import ai_reading
        ecg_id = int(re.match(r"(\d+)_", os.path.basename(args.wfdb)).group(1))  # "00338_hr" -> 338
        data["ai"] = ai_reading(ecg_id)
        print("AI reading added: " + ", ".join(
            f"{m['name']} {max(m['probs'], key=m['probs'].get)}" for m in data["ai"]["models"].values()))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("window.ECG_DATA = " + json.dumps(data) + ";\n")
    print(f"Wrote {args.out} ({data['vector'][0].__len__()} frames at {data['fs']:.0f} Hz)")


if __name__ == "__main__":
    main()
