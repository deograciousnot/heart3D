# 3. What already exists, and where this project could go

The original idea: use ECGs + AI to show a **real-time holographic view of the heart**.

## What already exists

| Level | What it is | Status |
|---|---|---|
| 3D vector loop from ECG | **Vectorcardiography (VCG)**; Dower / Kors matrices turn 12-lead into 3D | Since the 1950s. This project rebuilds it. |
| Electricity painted on a real 3D heart | **ECG imaging (ECGI)**: ~250-electrode vest + CT scan, solves an ill-posed inverse problem with regularisation (Tikhonov etc.). Medtronic sells one (CardioInsight). | Clinical, but heavy and expensive |
| AI on a standard 12-lead to locate problems | E.g. Vektor Medical: trained on simulated hearts to predict *where* an arrhythmia starts. Mayo Clinic AI-ECG detects a weak pump from the ECG alone. | Emerging, some regulator-cleared |
| Actual holograms in cardiology | HoloLens, RealView during catheter procedures | Exist, but driven by CT / ultrasound / catheter mapping, **not** skin ECG |

**Takeaway:** an honest tool shows **electrical activity mapped onto a model heart**, not the patient's actual heart.

## Roadmap for this project

1. ~~Data: PTB-XL (21k real 12-lead ECGs, free on PhysioNet)~~ ✔
2. ~~Reconstruct the 3D vector (Kors)~~ ✔
3. ~~Render it in three.js, in real time, synced with the 12 leads~~ ✔
4. **Compare normal vs abnormal**: load a myocardial infarction (MI) or bundle branch block record next to the normal one; the loop shape changes visibly.
5. **AI layer**: train a classifier on PTB-XL's labels (NORM, MI, STTC, CD, HYP superclasses), and highlight the heart region linked to the finding.
6. **Better heart model**: replace the ellipsoid with a real heart mesh (glTF), orient the loop inside it.
7. **Live hardware (optional)**: ADS1298 chip (8 channels, enough for 12-lead). **Battery power only, never mains**, and no connection to a wall-powered laptop while attached to a person. Real ECG kit is electrically isolated for safety.

## Useful links

- PTB-XL dataset: https://physionet.org/content/ptb-xl/1.0.3/
- PTB Diagnostic database (includes measured Frank / 3D leads, good for checking reconstructions): https://physionet.org/content/ptbdb/1.0.0/
- wfdb Python library: https://github.com/MIT-LCP/wfdb-python
- three.js docs: https://threejs.org/docs/
