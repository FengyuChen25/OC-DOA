# OC-DOA: DOA Estimation with Orthogonal-Basis CNN

Code for the paper **"OC-DOA"** — direction-of-arrival (DOA) estimation using
an orthogonal-basis CNN, benchmarked against HMC-ViT, LowSNR-CNN, MUSIC, CBF,
and the Cramér–Rao Bound (CRB) under a uniform linear array (ULA, M = 16).

## Repository structure

| Path | Description |
|---|---|
| `Asimulation_final1.py` | Main Monte-Carlo simulation script (switch experiments via `decision`) |
| `OCDOA_model.py` | OC-DOA network (orthogonal-basis CNN, `OrthoBasisCNN1D`) |
| `OCDOA_utils.py` | ESPRIT-based angle estimation from OC-DOA coefficients |
| `lowcnn_model.py` | LowSNR-CNN network |
| `hmc_vit_model.py` | HMC-ViT network |
| `simulation_utils.py` | Unified data generation (`generate_unified_Rx`) and feature extractors |
| `benchmark.py` | Parameter count + inference time benchmark |
| `hmcvit/best.pth` | HMC-ViT pretrained weights |
| `lowsnrcnn/net.pth` | LowSNR-CNN pretrained weights |
| `ocdoa/best_model_obmodel1.pth` | OC-DOA pretrained weights |
| `orthobasis_data_M16/` | Orthogonal basis metadata used by OC-DOA |

## Quick start

```bash
pip install -r requirements.txt
python Asimulation_final1.py
```

In `Asimulation_final1.py`, set the variable `decision` to one of
`N` / `SNR` / `angle` / `power_ratio` / `phase_error` / `correlation`
to reproduce the corresponding experiment (e.g. `decision = "phase_error"`
sweeps the phase-error standard deviation σ = 0, 2, ..., 40 degrees).

Result figures (PDF) and RMSE tables (CSV) are saved to `pic_final/`.

## Run it online (Binder)

Click the badge below to launch an executable environment in your browser —
no local installation required:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FengyuChen25/OC-DOA/HEAD)

### How to run inside Binder (JupyterLab)

1. **Console / Notebook cell** — open a Python console (Launcher → Console) or a
   notebook, then run:
   ```python
   %run Asimulation_final1.py
   ```
2. **Terminal** — open a Terminal (Launcher → Terminal), then run:
   ```bash
   python Asimulation_final1.py
   ```

> **Note**: `python Asimulation_final1.py` works only in a **Terminal**;
> inside a Console/Notebook cell use `%run Asimulation_final1.py` (a cell is a
> Python interpreter, not a shell).
>
> The full Monte-Carlo loop takes a while on Binder's CPU; set `num_epoch = 5`
> in `Asimulation_final1.py` for a quick smoke test before running the full
> experiment.

## Notes

- Model weights are stored with Git LFS; Binder pulls them automatically.
- `requirements.txt` installs a CPU build of PyTorch (Binder sessions have no
  GPU). For local GPU runs, install the CUDA wheel that matches your driver.
- The full Monte-Carlo loop (`num_epoch = 100`) is compute-intensive on CPU;
  reduce `num_epoch` for a quick smoke test.
