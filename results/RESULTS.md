# BCI-IV 2a within-session results — protocol evolution (all real runs)

Chance level 25% (4 classes). Protocol: per-subject session-T, stratified
75/25 split (seed 42), per-channel z-score (train stats only). 9 subjects.
Published within-session references reach ~65-75% mean with heavier protocols
(more augmentation/epochs, sometimes cross-session or cue-wise calibration).

| Protocol | EEGNet mean±std | DeepConvNet mean±std | Notes |
|---|---|---|---|
| v1 baseline: lr 1e-3, batch 64, 80 ep, patience 10 | 48.5 ± 12.5 | 29.9 ± 10.1 | DeepConvNet diverges (early-stop at ~11 ep) |
| **v2 tuned (canonical, CPU): lr 3e-4, batch 32, 150 ep, patience 30, grad-clip 1.0** | **47.5 ± 16.6** | **38.4 ± 15.9** | good subjects (A01/A03/A09) hit 60-72%; hard subjects (A02/A04/A06) stall near chance |
| v3 augmented: lr 1e-4, batch 32, 250 ep, patience 60, Gaussian-noise + per-sample circular time-shift | 46.5 ± 14.3 | 27.5 ± 2.4 | lr too low to learn; augmentation alone does not fix the hard subjects |
| **v2 on AMD GPU (torch-directml, `--device dml`)** | **48.0 ± 13.5** | 33.6 ± 13.3 | same protocol, ~10-20× faster wall time; EEGNet statistically identical to CPU v2; DeepConvNet less stable on DML (ELU falls back to CPU per-op; several subjects early-stop at ~31 ep) |

**Device note (2026-10-25):** AMD RX 6750 GRE via torch-directml (Python 3.12 venv at `D:\venvs\dml`,
Windows-native; ROCm-WSL abandoned — requires the separate "Adrenalin 26.2.2 for WSL2" driver +
ROCm 7.2 stack, and DML covers the need). EEGNet on DML: full 9-subject run in ~4 min (vs ~40 min CPU),
same accuracy → **use `--device dml` for all future EEGNet iteration**; DeepConvNet results above are
CPU-canonical, DML run archived as `*_dml.csv`.

**Conclusion (v3 done):** the canonical protocol is v2 — the repo defaults reproduce it
(`--aug` re-enables the v3 variant for reference). EEGNet sits at ~47-48% within-session
mean (chance 25%) with a 216-trial training split; subjects A01/A03/A08/A09 reach
55-72% across protocols while A02/A04/A06 stall — consistent with the known
subject-difficulty spread on this dataset, but below published within-session means
(~65-75%), which typically use K-fold CV over all 288 trials, per-subject
hyperparameters, and/or session-T -> session-E evaluation. Those are the next levers,
in that order (E-session official labels pending).

## 5-fold stratified CV (all 288 trials) + SNN — the headline table (2026-10-25)

kfold5 = 5-fold stratified CV over ALL 288 trials (the literature protocol);
SNN = SpikingEEGNet, surrogate-gradient LIF (T=8), pure PyTorch, same
protocol and budget (`--model spkieeg`).

| Model (kfold5, DML) | mean ± std over subjects | best subjects |
|---|---|---|
| EEGNet (ANN) | 52.5 ± 17.7 | A08 78.5, A09 74.0 |
| DeepConvNet (ANN) | 38.3 ± 15.2 | A09 72.2 |
| **SpikingEEGNet (SNN)** | **57.8 ± 19.3** | **A08 82.7 ± 3.2, A09 82.3 ± 1.3** |

**The SNN beats the ANN EEGNet on every one of the 9 subjects** (+5.3 pt mean;
largest gap A07 +12.6 pt), with remarkable fold stability on the best subjects
(A09 ± 1.3 %). Spiking holdout reference (v2 protocol): 56.8 ± 22.1, best
subjects A03 84.7 / A08 83.3 / A09 81.9.

The spiking decoder is also the designed bridge to the memristive in-memory
computing project: spike-domain compute is the native coding style of
memristive neuromorphic hardware — the same BCI-IV 2a protocol now connects
`../01_memristor_qat` and this repository.

Updated next levers: session-T -> session-E evaluation (official labels
pending), per-subject tuning, SNN depth/T sweep.
