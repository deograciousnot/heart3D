"""
Shared PTB-XL helpers for training: labels, splits, record paths.

Labels are the 5 diagnostic superclasses from scp_statements.csv:
    NORM  normal ECG
    MI    myocardial infarction (heart attack, old or new)
    STTC  ST/T change (often ischaemia)
    CD    conduction disturbance (bundle branch blocks, AV blocks, ...)
    HYP   hypertrophy (thickened heart muscle)
One ECG can have several (multi-label). ECGs with none are dropped.

Splits follow the dataset authors: folds 1-8 train, fold 9 validation, fold 10 test.
Folds are split by patient, and folds 9-10 hold only human-validated labels.
"""

import ast
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptbxl")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_NAMES = {
    "NORM": "Normal",
    "MI": "Heart attack (MI)",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}


def load_table(hz=100):
    """
    Returns (ids, paths, folds, Y): one entry per ECG that has at least one diagnostic superclass.
    Y is an (N, 5) 0/1 matrix in CLASSES order; paths are WFDB record paths without extension.
    """
    with open(os.path.join(ROOT, "scp_statements.csv"), encoding="utf-8") as f:
        scp = {row[""]: row for row in csv.DictReader(f)}
    superclass = {code: row["diagnostic_class"] for code, row in scp.items()
                  if row["diagnostic"] == "1.0" and row["diagnostic_class"]}

    column = "filename_lr" if hz == 100 else "filename_hr"
    ids, paths, folds, Y = [], [], [], []
    with open(os.path.join(ROOT, "ptbxl_database.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes = ast.literal_eval(row["scp_codes"])
            classes = {superclass[c] for c in codes if c in superclass}
            if not classes:
                continue
            ids.append(int(row["ecg_id"]))
            paths.append(os.path.join(ROOT, row[column]))
            folds.append(int(row["strat_fold"]))
            Y.append([int(c in classes) for c in CLASSES])
    return np.array(ids), paths, np.array(folds), np.array(Y)


def split_masks(folds):
    """Boolean masks for train (folds 1-8), validation (9) and test (10)."""
    return folds <= 8, folds == 9, folds == 10
