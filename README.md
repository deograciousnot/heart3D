# heart3D

Turn a 12-lead ECG into a **3D view of the heart's electrical activity**, animated in real time in the browser.

An ECG isn't a picture of the heart: it's 12 one-dimensional "shadows" (projections) of a single 3D arrow that swings around during every heartbeat. This project recovers that arrow (the **vectorcardiogram**) and renders it in three.js next to the 12 leads.

- Works on a **synthetic** normal heart (no download needed)
- Works on **real ECGs** from the PTB-XL dataset (PhysioNet)

## Quick start

```bash
pip install -r requirements.txt
python ecg_to_vcg.py                                               # synthetic data
python ecg_to_vcg.py --wfdb ptbxl/00001_hr --out ecg_data_real.js  # real record
python -m http.server 8765
```

Open http://localhost:8765/viewer.html, or `?data=ecg_data_real.js` for the real ECG.

## Notes (read in order)

1. [How an ECG works](docs/01-how-an-ecg-works.md): the biology, the 12 leads, the hardware
2. [The CS view](docs/02-the-cs-view.md): leads as dot products, reconstruction as least squares
3. [Landscape and ideas](docs/03-landscape-and-ideas.md): what already exists, roadmap
4. [Code walkthrough](docs/04-code-walkthrough.md): what each file and function does

## Data

The ECG in `ptbxl/` is from **PTB-XL** (Wagner et al., 2020), available on PhysioNet under CC BY 4.0: https://physionet.org/content/ptb-xl/1.0.3/

Not a medical device. For learning only.
