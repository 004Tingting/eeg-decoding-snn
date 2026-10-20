"""BCI Competition IV-2a preprocessing: GDF -> per-subject NPZ cache.

Pipeline (standard for EEGNet/DeepConvNet reproductions on this dataset):
- read raw GDF (250 Hz, 22 EEG + 3 EOG channels), keep the 22 EEG channels;
- band-pass 4-40 Hz;
- epoch the motor-imagery window [0.5, 3.5] s around each cue (codes 768-771
  = left hand / right hand / feet / tongue), labels remapped to 0-3;
- store as data/cache/<subj>T.npz with X (n, 22, 750) float32 and y (n,).

Note: the official artifact-rejection protocol (trials marked as artifacts in
the evaluation sheets) is approximated here by keeping cue-coded trials only;
documented as a deviation from the strict competition scoring.
"""
import glob
import os
import sys

import numpy as np
import mne

mne.set_log_level("ERROR")

CUE_CODES = [769, 770, 771, 772]
# Encoding as actually read by mne from these GDFs (verified on A01T):
# '768' fires 288x (one per trial, the generic onset marker) and 769-772 fire
# 72x each (the four class cues, one per class, consistent order across the
# session). The official spec numbers the class cues 768-771; mne's GDF reader
# shifts event types by +1 here. Labels are therefore code-769 in 0-3.
# '1023' markers = officially rejected (artifact) trials, dropped below.


def preprocess_subject(gdf_path, out_path, fmin=4.0, fmax=40.0):
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    raw.pick_channels(raw.ch_names[:22])  # 22 EEG channels, drop EOG
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
    events, code_map = mne.events_from_annotations(raw)
    # mne maps annotation descriptions ('768', ...) to internal codes via
    # code_map; resolve through it instead of hardcoding integers.
    rej_code = code_map.get("1023")
    rej = events[events[:, 2] == rej_code][:, 0] if rej_code is not None else np.array([])
    cue_codes = [code_map[str(c)] for c in (769, 770, 771, 772)]  # L/R/foot/tongue
    label_of = {c: i for i, c in enumerate(cue_codes)}
    ev = events[np.isin(events[:, 2], cue_codes)]
    # drop trials whose cue coincides with an official rejection marker (2 s)
    if len(rej):
        keep = np.ones(len(ev), dtype=bool)
        for i, p in enumerate(ev[:, 0]):
            if np.min(np.abs(rej - p)) < 2 * int(raw.info["sfreq"]):
                keep[i] = False
        ev = ev[keep]
    epochs = mne.Epochs(raw, ev, tmin=0.5, tmax=3.5, baseline=None,
                        preload=True, verbose=False)
    X = epochs.get_data(copy=False)[:, :, :750].astype(np.float32)
    y = np.array([label_of[c] for c in ev[:, 2]], dtype=np.int64)
    np.savez_compressed(out_path, X=X, y=y)
    return X.shape[0], np.bincount(y, minlength=4).tolist()


def build_cache(gdf_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(gdf_dir, "**", "*T.gdf"), recursive=True))
    if not files:
        print(f"no *T.gdf under {gdf_dir}"); return
    for fp in files:
        subj = os.path.basename(fp).split(".")[0]
        out = os.path.join(cache_dir, f"{subj}.npz")
        if os.path.exists(out):
            print(f"{subj}: cached"); continue
        n, counts = preprocess_subject(fp, out)
        print(f"{subj}: {n} trials, per-class {counts}")
    print("cache complete ->", cache_dir)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_cache(os.path.join(root, "data"), os.path.join(root, "data", "cache"))
