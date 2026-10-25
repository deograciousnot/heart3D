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


def detect_phases(h, fs):
    """Rough P / QRS / T labels for real data: find the QRS peaks in |h|, then use fixed windows around them."""
    from scipy.signal import find_peaks

    magnitude = np.linalg.norm(h, axis=0)
    peaks, _ = find_peaks(magnitude, height=0.5 * magnitude.max(), distance=int(0.3 * fs))
    phase = np.zeros(h.shape[1], dtype=int)
    ms = lambda x: int(x * fs / 1000)
    for p in peaks:
        phase[max(0, p + ms(-220)):max(0, p + ms(-90))] = 1   # P wave
        phase[max(0, p + ms(-50)):p + ms(60)] = 2             # QRS
        phase[p + ms(120):p + ms(420)] = 3                    # T wave
    return phase, peaks


def load_wfdb(path):
    import wfdb  # only needed for real data

    record = wfdb.rdrecord(path)
    names = [n.upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for n in record.sig_name]
    signals = clean_ecg(record.p_signal.T, record.fs)
    by_name = {n: signals[i] for i, n in enumerate(names)}
    leads = np.vstack([by_name[n] for n in LEAD_NAMES])
    t = np.arange(signals.shape[1]) / record.fs
    return t, leads, by_name, record.fs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wfdb", help="path to a WFDB record (without extension), e.g. PTB-XL records500/00000/00001_hr")
    parser.add_argument("--out", default="ecg_data.js", help="output file the viewer loads")
    args = parser.parse_args()

    if args.wfdb:
        t, leads, by_name, fs = load_wfdb(args.wfdb)
        h = inverse_kors(by_name)
        phase, peaks = detect_phases(h, fs)
        bpm = 60 * fs / np.median(np.diff(peaks)) if len(peaks) > 1 else float("nan")
        print(f"Real ECG: {len(peaks)} beats detected, ~{bpm:.0f} bpm")
        source = f"Real ECG, record {args.wfdb} (Kors reconstruction, ~{bpm:.0f} bpm)"
    else:
        fs = 500
        t, h_true, phase = synthetic_heart_vector(fs=fs)
        leads = forward(h_true)
        h = inverse_lstsq(leads)
        error = np.abs(h - h_true).max()
        print(f"Synthetic ECG: {leads.shape[0]} leads x {leads.shape[1]} samples")
        print(f"Reconstruction max error vs true heart vector: {error:.2e} mV")
        source = "Synthetic normal heart (least-squares reconstruction)"

    # Downsample to 250 Hz for the browser; plenty for drawing.
    step = max(1, int(round(fs / 250)))
    data = {
        "source": source,
        "fs": fs / step,
        "leadNames": LEAD_NAMES,
        "leads": np.round(leads[:, ::step], 4).tolist(),
        "vector": np.round(h[:, ::step], 4).tolist(),
        "phase": phase[::step].tolist(),
        "leadDirections": np.round(LEAD_DIRECTIONS, 4).tolist(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("window.ECG_DATA = " + json.dumps(data) + ";\n")
    print(f"Wrote {args.out} ({data['vector'][0].__len__()} frames at {data['fs']:.0f} Hz)")


if __name__ == "__main__":
    main()
