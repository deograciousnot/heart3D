"""
Rebuild every data file the viewers use, with both models' AI reading, plus data/records.js
(the list behind the viewer's record picker).

    python make_demo_data.py

Two sets of real ECGs:
  - comparison examples: the five records used by compare.html. They come from PTB-XL's training
    folds, so the models have seen them; the viewer says so.
  - unseen test ECGs: one per diagnosis group from fold 10, which no model saw during training or tuning.
"""

import json
import subprocess
import sys

RECORDS = [
    # (record, label, group)
    ("00001", "Normal", "comparison"),
    ("00492", "Anterior (anteroseptal) myocardial infarction", "comparison"),
    ("00290", "Inferior myocardial infarction", "comparison"),
    ("00338", "Left bundle branch block", "comparison"),
    ("00903", "Right bundle branch block", "comparison"),
    ("00057", "Normal", "test"),
    ("01737", "Inferior myocardial infarction (old)", "test"),
    ("01429", "ST depression, anterolateral (ischaemia)", "test"),
    ("07079", "First-degree AV block", "test"),
    ("07916", "Left ventricular hypertrophy", "test"),
]
# compare.html and the README already point at ecg_data_real.js for the normal example.
FILES = {"00001": "ecg_data_real.js"}


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, "ecg_to_vcg.py", *args], check=True)


def main():
    run("--out", "ecg_data.js")
    manifest = [{"file": "ecg_data.js", "label": "Normal (synthetic)", "group": "synthetic"}]
    for rec, label, group in RECORDS:
        out = FILES.get(rec, f"data/ptbxl_{rec}.js")
        run("--wfdb", f"ptbxl/{rec}_hr", "--out", out, "--label", label, "--ai")
        manifest.append({"file": out, "label": label, "group": group, "record": rec})
    with open("data/records.js", "w", encoding="utf-8") as f:
        f.write("window.ECG_RECORDS = " + json.dumps(manifest, indent=1) + ";\n")
    print(f"Wrote data/records.js ({len(manifest)} records)")


if __name__ == "__main__":
    main()
