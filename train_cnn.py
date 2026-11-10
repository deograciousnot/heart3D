"""
Baseline 2: a 1D ResNet that reads the raw 12-lead signal (no hand-made measurements).

Architecture (about 0.4 M weights, sized for a laptop CPU):
    12 leads x 1000 samples (10 s at 100 Hz)
    -> stem: conv 7 -> 32 channels, max-pool /2
    -> 4 residual stages (32, 64, 96, 128 channels), each halving time except the first
    -> global average + max pooling -> dropout -> 5 outputs (one probability per class)

A residual block is two convolutions plus a shortcut that adds the input back, so the
network learns corrections to what it already has; that keeps deep networks easy to train.

Training: random 5 s crops (half the compute, and the model sees each beat at different
positions), small random amplitude scaling, AdamW + one-cycle learning rate, binary
cross-entropy per class. The epoch with the best validation macro AUC is kept.
Testing uses the full 10 s.

    python train_cnn.py                 # train, keep best epoch by validation
    python train_cnn.py --test          # also score the test fold with the saved model (do this once)
"""

import argparse
import json
import os
import time

import numpy as np

from ecg_to_vcg import LEAD_NAMES, clean_ecg, read_record
from ptbxl_data import CLASS_NAMES, CLASSES, ROOT, load_table, split_masks

SIGNALS = os.path.join(ROOT, "signals_100hz.npy")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WEIGHTS = os.path.join(MODELS, "cnn.pt")


def load_leads(path):
    """Record -> (12, T) float32 in LEAD_NAMES order, band-passed like everything else in the project."""
    raw, fs, names = read_record(path)
    names = [n.upper().replace("AVR", "aVR").replace("AVL", "aVL").replace("AVF", "aVF") for n in names]
    raw = np.vstack([raw[names.index(n)] for n in LEAD_NAMES])
    return clean_ecg(raw, fs).astype(np.float32), fs


def build_signal_cache(paths):
    """All records in one (N, 12, 1000) float16 array, about 0.5 GB, so training reads memory, not 21k files."""
    out = np.lib.format.open_memmap(SIGNALS, mode="w+", dtype=np.float16, shape=(len(paths), 12, 1000))
    start = time.time()
    for i, p in enumerate(paths):
        x, fs = load_leads(p)
        assert fs == 100 and x.shape == (12, 1000), p
        out[i] = np.clip(x, -20, 20)
        if (i + 1) % 4000 == 0:
            print(f"  cached {i + 1}/{len(paths)} ({time.time() - start:.0f} s)", flush=True)
    out.flush()
    return np.load(SIGNALS, mmap_mode="r")


def make_model():
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(cin, cout, 7, stride, 3, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                nn.Conv1d(cout, cout, 7, 1, 3, bias=False), nn.BatchNorm1d(cout))
            self.skip = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.body(x) + self.skip(x))

    class ResNet1d(nn.Module):
        def __init__(self, n_classes=len(CLASSES)):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(12, 32, 7, 1, 3, bias=False), nn.BatchNorm1d(32),
                                      nn.ReLU(inplace=True), nn.MaxPool1d(2))
            self.stages = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 96, 2), Block(96, 128, 2))
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, x):
            import torch
            h = self.stages(self.stem(x))
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))

    return ResNet1d()


def predict(model, X, mean, std, batch=256):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = (torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)) - mean) / std
            out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out)


def macro_auc(Y, P):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(Y[:, c], P[:, c]) for c in range(Y.shape[1])]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--threads", type=int, default=3, help="CPU threads (default 3 of 8, keeps the laptop usable)")
    parser.add_argument("--test", action="store_true", help="score the saved model on the test fold (do this once)")
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    ids, paths, folds, Y = load_table(hz=100)
    X = np.load(SIGNALS, mmap_mode="r") if os.path.exists(SIGNALS) else None
    if X is None or X.shape[0] != len(paths):
        print(f"Caching {len(paths)} signals...", flush=True)
        X = build_signal_cache(paths)
    X = np.asarray(X)  # into RAM: 0.5 GB as float16
    train, val, test = split_masks(folds)
    tr_idx = np.nonzero(train)[0]

    # Per-lead scale from training data only.
    sample = X[tr_idx[:4000]].astype(np.float32)
    mean = torch.tensor(sample.mean(axis=(0, 2)), dtype=torch.float32)[None, :, None]
    std = torch.tensor(sample.std(axis=(0, 2)) + 1e-6, dtype=torch.float32)[None, :, None]

    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())

    if args.test:
        state = torch.load(WEIGHTS, map_location="cpu")
        model.load_state_dict(state["model"])
        thresholds = state["thresholds"]
        from sklearn.metrics import roc_auc_score
        P_val, P_test = predict(model, X[val], mean, std), predict(model, X[test], mean, std)
        results = {"model": f"1D ResNet on raw 12-lead signal ({n_params:,} weights)",
                   "best_epoch": state["epoch"], "train": int(train.sum()), "val": int(val.sum()),
                   "test": int(test.sum()), "classes": {}}
        for c, cls in enumerate(CLASSES):
            y, p = Y[test, c], P_test[:, c]
            pred = p >= thresholds[c]
            r = results["classes"][cls] = {
                "name": CLASS_NAMES[cls], "val_auc": float(roc_auc_score(Y[val, c], P_val[:, c])),
                "test_auc": float(roc_auc_score(y, p)), "threshold": float(thresholds[c]),
                "test_sensitivity": float((pred & (y == 1)).sum() / (y == 1).sum()),
                "test_specificity": float((~pred & (y == 0)).sum() / (y == 0).sum()),
                "test_positives": int(y.sum())}
            print(f"  {cls:5} val AUC {r['val_auc']:.3f} | test AUC {r['test_auc']:.3f}, sensitivity "
                  f"{r['test_sensitivity']:.2f}, specificity {r['test_specificity']:.2f}")
        results["macro_val_auc"] = macro_auc(Y[val], P_val)
        results["macro_test_auc"] = macro_auc(Y[test], P_test)
        print(f"Macro AUC: validation {results['macro_val_auc']:.3f}, test {results['macro_test_auc']:.3f}")
        with open(os.path.join(MODELS, "cnn_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Saved models/cnn_results.json")
        return

    print(f"Model: {n_params:,} weights. Train {train.sum()}, validation {val.sum()} ECGs, "
          f"{args.epochs} epochs on {args.threads} threads", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    batch, crop = 64, 500
    steps = args.epochs * (len(tr_idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Yt = torch.tensor(Y, dtype=torch.float32)
    best = -1.0
    os.makedirs(MODELS, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        order = rng.permutation(tr_idx)
        for b in range(len(order) // batch):
            idx = np.sort(order[b * batch:(b + 1) * batch])
            offs = rng.integers(0, 1000 - crop + 1, size=len(idx))
            xb = np.stack([X[i, :, o:o + crop] for i, o in zip(idx, offs)]).astype(np.float32)
            xb *= rng.uniform(0.85, 1.15, size=(len(idx), 1, 1)).astype(np.float32)
            xb = (torch.from_numpy(xb) - mean) / std
            loss = loss_fn(model(xb), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        P_val = predict(model, X[val], mean, std)
        auc = macro_auc(Y[val], P_val)
        mark = ""
        if auc > best:
            from train_features import best_threshold
            best = auc
            thresholds = [best_threshold(Y[val, c], P_val[:, c]) for c in range(len(CLASSES))]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_auc": auc,
                        "thresholds": thresholds, "mean": mean, "std": std}, WEIGHTS)
            mark = "  <- best, saved"
        print(f"epoch {epoch:2d}: loss {total / (b + 1):.4f}, val macro AUC {auc:.3f} "
              f"({time.time() - start:.0f} s){mark}", flush=True)
    print(f"Best validation macro AUC {best:.3f}. Run with --test to score the test fold once.")


if __name__ == "__main__":
    main()
