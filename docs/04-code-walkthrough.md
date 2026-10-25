# 4. Code walkthrough

```
heart3D/
├── ecg_to_vcg.py        # maths: ECG <-> 3D heart vector, writes data for the viewer
├── viewer.html          # three.js 3D viewer + 12-lead monitor strips
├── ecg_data.js          # generated: synthetic normal heart
├── ecg_data_real.js     # generated: real PTB-XL record 00001
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
| `detect_phases()` | Finds QRS peaks in the vector magnitude, labels P / QRS / T with fixed time windows (approximate) |
| `load_wfdb()` | Reads a PhysioNet record, filters it, returns leads by name |

Output is a `.js` file (`window.ECG_DATA = {...}`) so the viewer can load it with a plain `<script>` tag.

**Sanity check:** on synthetic data the reconstruction error is ~1e-15, because the leads were made with the exact same dot products. That proves the pipeline is consistent, not that it's medically accurate.

## viewer.html

- Coordinates: Frank (x left, y feet, z back) -> three.js (x right on screen, y up, z towards viewer). You're looking at the patient from the front.
- **Arrow** = current heart vector, coloured by phase (blue P, orange QRS, green T).
- **Trail** = last 0.4 s of the arrow tip; **faint loop** = one full beat.
- **Lead lines** turn green when that lead is currently positive, orange when negative (the "camera" rule in 3D).
- **Right panel** = 12 leads in a 2-second monitor sweep, in sync with the 3D view.
- Everything auto-scales to the recording's amplitude.

## Running it

```bash
pip install -r requirements.txt

python ecg_to_vcg.py                                                  # synthetic -> ecg_data.js
python ecg_to_vcg.py --wfdb ptbxl/00001_hr --out ecg_data_real.js     # real record

python -m http.server 8765
# then open http://localhost:8765/viewer.html                         (synthetic)
#       or http://localhost:8765/viewer.html?data=ecg_data_real.js    (real)
```

A local server is needed because browsers block ES modules on `file://` pages.

## Getting more records

PTB-XL records live at `https://physionet.org/files/ptb-xl/1.0.3/records500/<folder>/<id>_hr.{hea,dat}`, e.g. `records500/00000/00001_hr`. The diagnosis labels for every record are in `ptbxl_database.csv` in the same dataset.

## Known limitations

- Phase colours use fixed time windows, so they're approximate on fast or abnormal rhythms.
- Kors is fitted to average adults; per person it's an estimate.
- The heart is a stand-in ellipsoid.
