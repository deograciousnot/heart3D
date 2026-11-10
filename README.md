# heart3D

Turn a 12-lead ECG into a **3D view of the heart's electrical activity**, animated in real time in the browser.

An ECG isn't a picture of the heart: it's 12 one-dimensional "shadows" (projections) of a single 3D arrow that swings around during every heartbeat. This project recovers that arrow (the **vectorcardiogram**) and renders it in three.js next to the 12 leads.

- Works on a **synthetic** normal heart (no download needed)
- Works on **real ECGs** from the PTB-XL dataset (PhysioNet)
- **Two AI models** reading 5 diagnosis groups: a measurement model (test macro AUC 0.920) and a 1D ResNet on the raw signal (0.928), with "what drove it" shown on the 3D loop
- **Anatomical 3D heart** that glows where the electrical wave is heading, and shades the likely heart attack wall when an MI is flagged
- **Compare normal vs abnormal** hearts (old heart attacks, bundle branch blocks) side by side: `compare.html`

## Quick start

The viewer data is already built, so you only need a local web server:

```bash
python -m http.server 8765
```

Open http://localhost:8765/viewer.html (pick any recording from the list) or http://localhost:8765/compare.html.

## Rebuilding and training

```bash
pip install -r requirements.txt
python download_ptbxl.py          # all 21,799 PTB-XL records at 100 Hz, about 0.5 GB (resumable)
python train_features.py          # measurement model: features (~10 min, cached) + training, validation only
python train_features.py --test   # score the test fold, once the model is final
python train_cnn.py               # 1D ResNet, ~35 min on a laptop CPU (3 threads)
python train_cnn.py --test        # score the test fold, once
python make_demo_data.py          # rebuild all viewer data files with AI readings
```

Models and their results are in `models/`. See the [code walkthrough](docs/04-code-walkthrough.md) for details.

## Notes (read in order)

1. [How an ECG works](docs/01-how-an-ecg-works.md): the biology, the 12 leads, the hardware
2. [The CS view](docs/02-the-cs-view.md): leads as dot products, reconstruction as least squares
3. [Landscape and ideas](docs/03-landscape-and-ideas.md): what already exists, roadmap
4. [Code walkthrough](docs/04-code-walkthrough.md): what each file and function does

## Data

The ECG in `ptbxl/` is from **PTB-XL** (Wagner et al., 2020), available on PhysioNet under CC BY 4.0: https://physionet.org/content/ptb-xl/1.0.3/

Not a medical device. For learning only.
