# 4. Code walkthrough

```
heart3D/
├── ecg_to_vcg.py        # maths: ECG <-> 3D heart vector, writes data for the viewer
├── viewer.html          # three.js 3D viewer + 12-lead monitor strips
├── compare.html         # normal vs abnormal: averaged beats side by side
├── download_ptbxl.py    # downloads the full PTB-XL dataset (resumable)
├── ptbxl_data.py        # PTB-XL labels (5 superclasses) and train / validation / test splits
├── train_features.py    # classifier baseline 1: median-beat features + gradient-boosted trees
├── models/              # saved results (feature_model_results.json)
├── ecg_data.js          # generated: synthetic normal heart
├── ecg_data_real.js     # generated: real PTB-XL record 00001 (normal)
├── data/                # generated: abnormal PTB-XL records (ptbxl_<id>.js)
├── ptbxl/               # downloaded PTB-XL record(s) (.hea header + .dat signal)
└── docs/                # these notes
```

## ecg_to_vcg.py

| Function | What it does |
|---|---|
| `LEAD_DIRECTIONS` | (12, 3) matrix of idealised lead "camera" directions |
| `synthetic_heart_vector()` | Builds a plausible normal heart vector: P, QRS and T loops for each beat |
| `forward(h)` | `L @ h`: heart vector -> 12 leads (each lead is a dot product) |
| `inverse_lstsq(leads)` | Least squares: 12 leads -> heart vector, using the ideal directions |
| `KORS` / `inverse_kors()` | Kors regression matrix for **real** data (leads I, II, V1-V6 -> X, Y, Z) |
| `clean_ecg()` | 0.5-40 Hz band-pass: removes breathing drift and muscle / mains noise |
| `find_r_peaks()` | QRS peaks = big maxima of the vector magnitude, at least 300 ms apart |
| `median_beat()` | Cuts a window around every beat, lines the beats up (cross-correlation, +-40 ms), takes the median |
| `qrs_bounds()` | QRS onset / offset from the **spatial velocity** \|dh/dt\|: grows outward from the fastest point while the loop is still moving fast |
| `analyse_beats()` | Median beat + measurements: QRS duration, QRS / T frontal axis, 3D QRS-T angle, peak amplitude, heart rate |
| `detect_phases()` | Labels P / QRS / T for the whole recording, using the QRS onset / offset measured on the median beat |
| `read_record()` | Reads a PhysioNet record directly (16-bit format, as in PTB-XL); about 5x faster than the wfdb library, identical values |
| `load_wfdb()` | Reads a record, filters it, returns leads by name |

Output is a `.js` file (`window.ECG_DATA = {...}`) so the viewer can load it with a plain `<script>` tag.
It holds the full recording plus a `beat` object: one averaged heartbeat (vector, 12 leads, phases), the R-peak
index, and the measurements. `--label` adds a diagnosis name that the viewers show.

**Sanity check:** on synthetic data the reconstruction error is ~1e-15, because the leads were made with the exact same dot products. That proves the pipeline is consistent, not that it's medically accurate.

## viewer.html

- Coordinates: Frank (x left, y feet, z back) -> three.js (x right on screen, y up, z towards viewer). You're looking at the patient from the front.
- **Arrow** = current heart vector, coloured by phase (blue P, orange QRS, green T).
- **Trail** = last 0.4 s of the arrow tip; **faint loop** = one full beat.
- **Lead lines** turn green when that lead is currently positive, orange when negative (the "camera" rule in 3D).
- **Right panel** = 12 leads in a 2-second monitor sweep, in sync with the 3D view.
- Everything auto-scales to the recording's amplitude.

## compare.html

- Loads the normal record plus four abnormal ones (list at the top of the script, each with a "what's happening"
  and "what to look for" text). Only the averaged beat of each is used.
- **3D view**: both loops as tubes (thick QRS, thin T, thinnest P), with a dot moving along each. Beats are lined up
  on the R peak, so the dots are at the same moment of the heartbeat.
- **Plane views**: front, top-down and side, the classic way a vectorcardiogram is printed. Dots every 20 ms show
  speed: bunched dots = slow conduction (bundle branch blocks).
- **12 leads** of the averaged beat, normal and abnormal overlaid, with a time cursor.
- All loops share one scale so sizes are comparable; the 12-lead traces are scaled per pair.

| Record | Condition | What shows up |
|---|---|---|
| 00001 | Normal | QRS 80 ms, narrow loop down-left, QRS-T angle 66° |
| 00492 | Anterior (anteroseptal) MI | Early QRS goes backward instead of forward; QS in V2 |
| 00290 | Inferior MI | Weaker early downward forces, axis 9° vs 29° (mild) |
| 00338 | Left bundle branch block | QRS 138 ms, big backward loop, T opposite (QRS-T 154°) |
| 00903 | Right bundle branch block | QRS 132 ms, slow late tail to the right and front |

Records were picked from PTB-XL's `ptbxl_database.csv`: the only diagnosis, 100% likelihood, sinus rhythm,
validated by a cardiologist, and no noise / electrode / extra-beat flags.

## train_features.py

- Runs every labelled PTB-XL ECG (21,388) through the viewer pipeline and summarises the median beat in 78 numbers:
  QRS width, axes, QRS-T angle, initial / terminal / peak QRS vectors, T vector, loop areas, and per lead the
  QRS max and min, ST level 60 ms after the QRS, and T amplitude. Cached in `ptbxl/features_100hz.npz`.
- One `HistGradientBoostingClassifier` per class (multi-label), class-weighted for the imbalance.
- Folds 1-8 train, fold 9 picks each class's threshold, fold 10 is scored only with `--test`.

| Class | Test AUC | Sensitivity | Specificity |
|---|---|---|---|
| Normal | 0.936 | 0.93 | 0.80 |
| ST/T change | 0.931 | 0.85 | 0.86 |
| MI | 0.928 | 0.87 | 0.83 |
| Hypertrophy | 0.906 | 0.77 | 0.86 |
| Conduction disturbance | 0.887 | 0.69 | 0.92 |

Macro AUC: validation 0.920, test 0.918. Conduction disturbance is weakest: the PR interval (AV blocks) isn't measured yet.

## Running it

```bash
pip install -r requirements.txt

python ecg_to_vcg.py                                                  # synthetic -> ecg_data.js
python ecg_to_vcg.py --wfdb ptbxl/00001_hr --out ecg_data_real.js --label "Normal"
python ecg_to_vcg.py --wfdb ptbxl/00492_hr --out data/ptbxl_00492.js --label "Anterior (anteroseptal) myocardial infarction"
python ecg_to_vcg.py --wfdb ptbxl/00290_hr --out data/ptbxl_00290.js --label "Inferior myocardial infarction"
python ecg_to_vcg.py --wfdb ptbxl/00338_hr --out data/ptbxl_00338.js --label "Left bundle branch block"
python ecg_to_vcg.py --wfdb ptbxl/00903_hr --out data/ptbxl_00903.js --label "Right bundle branch block"

python -m http.server 8765
# then open http://localhost:8765/viewer.html                         (synthetic)
#       or http://localhost:8765/viewer.html?data=ecg_data_real.js    (real)
#       or http://localhost:8765/viewer.html?data=data/ptbxl_00338.js (any record in data/)
#       or http://localhost:8765/compare.html                         (normal vs abnormal)
```

A local server is needed because browsers block ES modules on `file://` pages.

## Getting more records

PTB-XL records live at `https://physionet.org/files/ptb-xl/1.0.3/records500/<folder>/<id>_hr.{hea,dat}`, e.g. `records500/00000/00001_hr`. The diagnosis labels for every record are in `ptbxl_database.csv` in the same dataset.

## Known limitations

- P and T phase labels are placed at fixed offsets from the measured QRS, so they're approximate on fast or abnormal rhythms. The QRS itself is measured.
- QRS boundaries from spatial velocity are a simple method; expect a few ms of difference from a cardiologist's caliper.
- Frontal axis is unreliable when limb-lead voltage is low (records 00001 and 00903).
- Each condition is one patient, so some differences are between-person variation, not the disease.
- Kors is fitted to average adults; per person it's an estimate.
- The heart is a stand-in ellipsoid.
