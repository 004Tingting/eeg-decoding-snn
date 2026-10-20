"""Per-subject within-session training/evaluation on BCI-IV 2a.

Protocol (first milestone): for each subject, session T (288 trials) is split
stratified 75/25 into train/val with a fixed seed; per-channel z-score is fit
on train only. Reports per-subject accuracy and mean +/- std over 9 subjects.
Chance level is 25% (4 classes). Session-T -> session-E evaluation with the
official labels is a follow-up once the labels file is in hand.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eeg_models import EEGNet, DeepConvNet  # noqa: E402
from src.snn_models import SpikingEEGNet  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = np.arange(4)


def make_model(name):
    return {"eegnet": EEGNet, "deepconv": DeepConvNet, "spkieeg": SpikingEEGNet}[name]()


def load_subject(subj):
    d = np.load(os.path.join(ROOT, "data", "cache", f"{subj}T.npz"))
    return d["X"], d["y"]


def split_stratified(y, frac=0.75, seed=42):
    rng = np.random.default_rng(seed)
    idx_tr, idx_va = [], []
    for c in CLASSES:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        idx_tr += list(idx[:k]); idx_va += list(idx[k:])
    return np.array(idx_tr), np.array(idx_va)


def stratified_kfold(y, k=5, seed=42):
    """Yield (train_idx, val_idx) for k stratified folds over all trials."""
    rng = np.random.default_rng(seed)
    fold_bins = [[] for _ in range(k)]
    for c in CLASSES:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_bins[j % k].append(i)
    for j in range(k):
        va = np.array(sorted(fold_bins[j]))
        tr = np.array(sorted(i for jj in range(k) if jj != j for i in fold_bins[jj]))
        yield tr, va


def run_kfold(model_name, subj, k=5, epochs=150, patience=30, batch=32, lr=3e-4,
              seed=42, device=None):
    """Stratified k-fold CV over ALL 288 trials (the literature protocol)."""
    torch.manual_seed(seed)
    X, y = load_subject(subj)
    X = X[:, None, :, :]
    fold_accs = []
    for i_tr, i_va in stratified_kfold(y, k=k, seed=seed):
        mu, sd = X[i_tr].mean(axis=(0, 2), keepdims=True), X[i_tr].std(axis=(0, 2), keepdims=True) + 1e-8
        Xn = (X - mu) / sd
        tr = DataLoader(TensorDataset(torch.from_numpy(Xn[i_tr]), torch.from_numpy(y[i_tr])),
                        batch_size=batch, shuffle=True)
        va_x = torch.from_numpy(Xn[i_va]).to(device); va_y = torch.from_numpy(y[i_va]).to(device)
        model = make_model(model_name).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best, wait = 0.0, 0
        for ep in range(epochs):
            model.train()
            for xb, yb in tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                acc = (model(va_x).argmax(1) == va_y).float().mean().item()
            if acc > best:
                best, wait = acc, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        fold_accs.append(best)
    return float(np.mean(fold_accs)), float(np.std(fold_accs)), fold_accs


def augment(xb, noise_std=0.1, max_shift=20):
    """Gaussian noise + per-sample circular time shift (standard for small
    within-session MI training sets)."""
    xb = xb + torch.randn_like(xb) * noise_std
    shifts = torch.randint(-max_shift, max_shift + 1, (xb.size(0),))
    for j in range(xb.size(0)):
        if shifts[j]:
            xb[j] = torch.roll(xb[j], int(shifts[j]), dims=-1)
    return xb


def run_subject(model_name, subj, epochs=150, patience=30, batch=32, lr=3e-4,
                seed=42, aug=False, device=None):
    torch.manual_seed(seed)
    X, y = load_subject(subj)
    X = X[:, None, :, :]  # (n, 1, 22, 750): Conv2d channel dim
    i_tr, i_va = split_stratified(y, seed=seed)
    mu, sd = X[i_tr].mean(axis=(0, 2), keepdims=True), X[i_tr].std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mu) / sd
    tr = DataLoader(TensorDataset(torch.from_numpy(X[i_tr]), torch.from_numpy(y[i_tr])),
                    batch_size=batch, shuffle=True)
    va_x = torch.from_numpy(X[i_va]).to(device); va_y = torch.from_numpy(y[i_va]).to(device)

    model = make_model(model_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, wait = 0.0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            batch_in = augment(xb) if aug else xb
            F.cross_entropy(model(batch_in), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(va_x).argmax(1) == va_y).float().mean().item()
        if acc > best:
            best, wait = acc, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eegnet", choices=["eegnet", "deepconv", "spkieeg"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--aug", action="store_true",
                    help="enable Gaussian-noise + time-shift augmentation (v3; did not help)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "dml"],
                    help="dml = AMD GPU via torch-directml (ELU falls back to CPU per-op)")
    ap.add_argument("--subjects", default="all", help="comma list like A01,A02 or 'all'")
    ap.add_argument("--cv", default="holdout", choices=["holdout", "kfold5"],
                    help="kfold5 = 5-fold stratified CV over all 288 trials (literature protocol)")
    args = ap.parse_args()

    device = None
    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()
        print("device:", torch_directml.device_name(0))

    subjects = ([f"A0{i}" for i in range(1, 10)] if args.subjects == "all"
                else args.subjects.split(","))
    rows, accs = [], []
    if args.cv == "kfold5":
        for s in subjects:
            mean_a, std_a, folds = run_kfold(args.model, s, k=5, epochs=args.epochs,
                                             patience=args.patience, device=device)
            accs.append(mean_a)
            rows.append((s, mean_a, std_a, "|".join(f"{a:.3f}" for a in folds)))
            print(f"{args.model} {s}: kfold5 mean {mean_a * 100:.2f}% ± {std_a * 100:.2f}%", flush=True)
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        out_csv = os.path.join(ROOT, "results", f"{args.model}_kfold5.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["subject", "kfold_mean_acc", "kfold_std", "fold_accs"])
            w.writerows(rows)
            w.writerow(["mean", float(np.mean(accs)), float(np.std(accs)), ""])
        print(f"{args.model} kfold5: mean {np.mean(accs) * 100:.2f}%  std {np.std(accs) * 100:.2f}%  (chance 25%)")
        print("saved ->", out_csv)
        return

    for s in subjects:
        acc, n_ep = run_subject(args.model, s, args.epochs, args.patience,
                                aug=args.aug, device=device)
        accs.append(acc)
        rows.append((s, acc, n_ep))
        print(f"{args.model} {s}: acc {acc * 100:.2f}% (epochs {n_ep})", flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_csv = os.path.join(ROOT, "results", f"{args.model}_subjects.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "val_acc", "epochs"])
        w.writerows(rows)
        w.writerow(["mean", float(np.mean(accs)), ""])
        w.writerow(["std", float(np.std(accs)), ""])
    print(f"{args.model}: mean {np.mean(accs) * 100:.2f}%  std {np.std(accs) * 100:.2f}%  (chance 25%)")
    print("saved ->", out_csv)


if __name__ == "__main__":
    main()
