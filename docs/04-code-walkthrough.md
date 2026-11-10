# 4. Code walkthrough

```
heart3D/
├── ecg_to_vcg.py        # maths: ECG <-> 3D heart vector, median beat, measurements; writes viewer data
├── viewer.html          # live viewer: 3D vector, vitals, AI reading, 12-lead monitor sweep
├── compare.html         # normal vs abnormal: averaged beats side by side
├── style.css, ui.js     # shared look (clinical glass, dark + light theme) and page helpers
├── heart_model.js       # schematic anatomical heart for both 3D views: beat glow + MI wall shading
├── download_ptbxl.py    # downloads the full PTB-XL dataset (resumable)
├── ptbxl_data.py        # PTB-XL labels (5 superclasses) and train / validation / test splits
├── train_features.py    # model 1: 81 median-beat measurements + gradient-boosted trees
├── train_cnn.py         # model 2: 1D ResNet on the raw 12-lead signal
├── train_location.py    # MI location: which wall of the left ventricle (front, bottom, left side)
├── predict.py           # runs both models on one ECG and explains the answer (for the viewer)
├── make_demo_data.py    # rebuilds every viewer data file with AI readings + data/records.js
├── models/              # trained models and their results
├── ecg_data.js          # generated: synthetic normal heart
├── ecg_data_real.js     # generated: PTB-XL record 00001 (normal)
├── data/                # generated: other PTB-XL records + records.js (the viewer's record list)
├── ptbxl/               # example PTB-XL records (500 Hz); the full 100 Hz download goes in ptbxl/records100/
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
| `median_beat()` | Cuts a window around every beat (450 ms before R, 550 after), lines the beats up (cross-correlation, +-40 ms), takes the median |
| `qrs_bounds()` | QRS onset / offset from the **spatial velocity** \|dh/dt\|: grows outward from the fastest point while the loop is still moving fast |
| `p_wave_bounds()` | P wave onset / offset: the strong bump in \|h\| closest to the QRS (up to 400 ms before it), edges where it falls below a quarter or dips. None when there's no clear P (atrial fibrillation) |
| `analyse_beats()` | Median beat + measurements: PR interval, P duration, QRS duration, QRS / T frontal axis, 3D QRS-T angle, peak amplitude, heart rate |
| `detect_phases()` | Labels P / QRS / T for the whole recording, using the P and QRS timing measured on the median beat |
| `read_record()` | Reads a PhysioNet record directly (16-bit format, as in PTB-XL); about 5x faster than the wfdb library, identical values |
| `load_wfdb()` | Reads a record, filters it, returns leads by name |

Output is a `.js` file (`window.ECG_DATA = {...}`) so the viewer can load it with a plain `<script>` tag.
It holds the full recording, a `beat` object (one averaged heartbeat: vector, 12 leads, phases, R-peak index,
measurements) and, with `--ai`, an `ai` object with both models' reading. `--label` adds a diagnosis name.

**Sanity checks:**
- On synthetic data the reconstruction error is ~1e-15, because the leads were made with the exact same dot products.
  That proves the pipeline is consistent, not that it's medically accurate.
- PR interval on 150 PTB-XL ECGs per group: normal median 180 ms (P found in 149/150), first-degree AV block
  median 250 ms (137/150), atrial fibrillation: no P wave in 42/48. The vector-based PR runs about 20 ms longer than
  single-lead textbook values, because the P wave "starts" as soon as any lead sees it.

## viewer.html

- **Record picker** (from `data/records.js`): five unseen test ECGs, the five comparison examples, and the synthetic heart.
  Every real ECG is labelled "unseen test ECG" or "seen in training", plus the cardiologist's diagnosis.
- **Vitals**: heart rate, PR, QRS width, QRS axis, QRS-T angle from the averaged beat, amber when outside the usual range;
  "Right now" follows playback.
- **3D**: arrow = current heart vector, coloured by phase (P blue, QRS coral, T teal); trail = last 0.4 s;
  thin loop = the averaged beat; lead lines turn green / orange when that lead is positive / negative.
- **Heart** (`heart_model.js`): ventricles, atria and great vessels as a translucent hologram, oriented like a real
  heart (apex down, left and forward). The surface the heart vector points at glows in the phase colour: atria during
  P, ventricles during QRS and T. This shows where the wave is heading, not a measured activation map.
- **Likely heart attack site**: when a model flags MI, the MI location model's wall(s) are shaded amber on the left
  ventricle and listed in the AI panel. "Heart" and "Glow" toggles switch these off.
- **AI reading**: both models' probabilities for the five classes, the flag line (threshold set on validation data),
  and "what drove it" for the selected class. The part of the beat that drove it most glows on the 3D loop.
- **12 leads** in a 2.5 s bedside-monitor sweep, in sync with the 3D view.
- Theme: dark "monitor" by default, light for light-mode systems; the sun button switches and remembers.

## compare.html

- Loads the normal record plus four abnormal ones (list at the top of the script, each with a "what's happening"
  and "what to look for" text). Only the averaged beat of each is used.
- **3D view**: both loops as tubes (thick QRS, thin T, thinnest P), with a dot moving along each. Beats are lined up
  on the R peak, so the dots are at the same moment of the heartbeat.
- **Plane views**: front, top-down and side, the classic way a vectorcardiogram is printed. Dots every 20 ms show
  speed: bunched dots = slow conduction (bundle branch blocks).
- **12 leads** of the averaged beat, normal and abnormal overlaid, with a time cursor.
- **AI reading** of the measurement model for both records (both are training ECGs, as the page says).
- The same heart model sits behind the loops (no beat glow); for the two MI examples the wall named in the
  cardiologist's diagnosis is shaded amber.

| Record | Condition | What shows up |
|---|---|---|
| 00001 | Normal | QRS 82 ms, narrow loop down-left, QRS-T angle 67° |
| 00492 | Anterior (anteroseptal) MI | Early QRS goes backward instead of forward; QS in V2 |
| 00290 | Inferior MI | Weaker early downward forces, axis 9° vs 29° (mild) |
| 00338 | Left bundle branch block | QRS 138 ms, big backward loop, T opposite (QRS-T 155°) |
| 00903 | Right bundle branch block | QRS 136 ms, slow late tail to the right and front |

Records were picked from PTB-XL's `ptbxl_database.csv`: the only diagnosis, 100% likelihood, sinus rhythm,
validated by a cardiologist, and no noise / electrode / extra-beat flags. The unseen test ECGs (fold 10) were picked
the same way: 00057 normal, 01737 old inferior MI, 01429 anterolateral ST depression, 07079 first-degree AV block,
07916 left ventricular hypertrophy.

## The two models

Both predict the 5 PTB-XL superclasses (normal, MI, ST/T change, conduction disturbance, hypertrophy), multi-label.
Folds 1-8 train, fold 9 picks the model and each class's threshold, fold 10 is scored once with `--test`.

**`train_features.py` (measurement model).** Every labelled ECG (21,388) goes through the viewer pipeline; the
median beat becomes 81 numbers: PR interval and P duration (or "no clear P wave"), QRS width, heart rate, axes,
QRS-T angle, initial / terminal / peak QRS vectors, T vector, loop areas, and per lead the QRS max and min, ST level
60 ms after the QRS, and T amplitude. One `HistGradientBoostingClassifier` per class, class-weighted.

**`train_cnn.py` (neural network).** A 1D ResNet (392k weights) on the raw 12 x 1000 signal (10 s at 100 Hz):
a convolution stem, four residual stages (32-128 channels), average + max pooling, 5 outputs. Trained on random
5 s crops with small amplitude jitter, AdamW and a one-cycle learning rate, 25 epochs on 3 CPU threads
(about 1.5 min per epoch on a 4-core laptop). The epoch with the best validation macro AUC is kept.

| Class | Measurement model test AUC | Neural network test AUC |
|---|---|---|
| Normal | 0.938 | 0.948 |
| Heart attack (MI) | 0.928 | 0.931 |
| ST/T change | 0.930 | 0.937 |
| Conduction disturbance | 0.895 | 0.924 |
| Hypertrophy | 0.906 | 0.901 |
| **Macro** | **0.920** | **0.928** |

Validation macro AUC: 0.922 vs 0.933. Published 1D CNNs on the same split reach roughly 0.93 (Strodthoff et al., 2021). The network wins on every class except hypertrophy; the biggest gain is conduction disturbance. The measurement model's advantage is that its reasons are named measurements ("PR interval 260 ms"). On the five unseen demo ECGs both models flag the cardiologist's class every time; for the AV block and the hypertrophy ECG they also flag ST/T change.

**`train_location.py` (MI location).** For ECGs with an MI code, which wall is it on? PTB-XL's MI codes are grouped
into three walls: front and septum (AMI, ASMI, ALMI, INJAS, INJAL), bottom and back (IMI, ILMI, IPLMI, IPMI, PMI,
INJIN, INJIL) and left side (LMI, ALMI, ILMI, IPLMI, INJAL, INJLA, INJIL). Same 81 measurements, gradient-boosted
trees, multi-label, trained on the 4,379 training ECGs that have an MI. Test AUC: front 0.964,
bottom 0.967, left side 0.887 (550 test MIs). The viewer only asks it when a model flags MI.

**`predict.py`** runs both on one ECG (always its 100 Hz version, as in training) and explains the answer by
occlusion: how much each class's probability drops when one part of the beat is made typical (measurement model:
that group of features reset to the training median; network: that part of every beat flattened to baseline).
This shows what the model relies on, not medical cause.

## Running it

```bash
pip install -r requirements.txt
python -m http.server 8765        # then open http://localhost:8765/viewer.html (or compare.html)
```

To rebuild everything from scratch:

```bash
python download_ptbxl.py          # all 21,799 PTB-XL records at 100 Hz, about 0.5 GB
python train_features.py          # measurement model: features (~10 min, cached) + training, validation only
python train_features.py --test   # score the test fold, once the model is final
python train_cnn.py               # neural network, ~35 min on a laptop CPU
python train_cnn.py --test        # score the test fold, once
python train_location.py --test   # MI location model (needs the feature cache), ~1 min
python make_demo_data.py          # all viewer data files with AI readings
```

A local server is needed because browsers block ES modules on `file://` pages. Single records:
`python ecg_to_vcg.py --wfdb ptbxl/00338_hr --out data/ptbxl_00338.js --label "Left bundle branch block" --ai`.

## Known limitations

- The T phase is placed at a fixed offset from the measured QRS, so it's approximate on fast or unusual rhythms.
- QRS and P boundaries are simple threshold methods; expect differences of 10-20 ms from a cardiologist's caliper.
- Frontal axis is unreliable when limb-lead voltage is low (records 00001 and 00903).
- Each comparison condition is one patient, so some differences are between-person variation, not the disease.
- Kors is fitted to average adults; per person it's an estimate.
- The heart is a schematic shape, not this patient's anatomy, and its walls are approximate regions.
- The models are trained and tested on one dataset (PTB-XL, German hospitals, 1989-1996 devices); performance on
  other populations and machines is unknown.
