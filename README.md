# Caspian Sea Level Prediction with Physics-Informed Neural Networks

**Paper ID 100** — IEEE 3rd International Student Conference on Digital Generation (DG 2026)

Three PINN formulations (water-balance, Budyko-constrained, Priestley-Taylor) predict Caspian Sea level variations. PINN-3 achieves R²=0.6939±0.0032 with 2-17× lower seed variance than an unconstrained CNN-LSTM baseline. All models share an identical backbone — performance differences come purely from physics formulations.

**Dataset:** [Kaggle](https://www.kaggle.com/datasets/chingchonghaha/caspian-sea-hydroclimatic-dataset-1993-2026) | **Code:** [GitHub](https://github.com/fulaura/Caspian-Sea-level-prediction-PINN)

## Authors

Bekzat Ashirbek, Arlan Mirseiit, Amir Nurseit, Anar Rakhimzhanova — *Astana IT University*

---

## Contents

- [Key Features](#key-features)
  - [Physics-Informed Architectures](#physics-informed-architectures)
  - [Learned Homoscedastic Uncertainty Weighting](#learned-homoscedastic-uncertainty-weighting-pinn-2)
  - [Shared CNN-LSTM Backbone](#shared-cnn-lstm-backbone)
  - [Multi-Seed Evaluation](#multi-seed-evaluation-framework)
  - [Simple Baselines](#simple-baselines)
  - [Diebold-Mariano Testing](#diebold-mariano-statistical-testing)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Key Results Summary](#key-results-summary)
- [Citation](#citation)

---

## Key Features

### Physics-Informed Architectures

| Model | Physical Principle | Constraint | Key Innovation |
|-------|-------------------|-----------|----------------|
| **PINN-1** | Mass conservation | Water balance | Learnable scaling of evaporation (α_E), Volga inflow (β_volga), ungauged runoff (γ_other) |
| **PINN-2** | Budyko framework | Runoff partitioning | Time-varying Budyko exponent n_t via Choudhury-Yang; learned homoscedastic uncertainty weighting (Kendall et al., 2017) |
| **PINN-3** | Priestley-Taylor | Energy-balance evaporation | Learnable PT coefficient (α_PT) and wind exchange (C_wind); FAO-56 psychrometrics |
| **Hybrid** | All three combined | Multi-constraint | Joint optimization of 7 learnable physics parameters |

### Learned Homoscedastic Uncertainty Weighting (PINN-2)

Instead of a fixed λ_phys, PINN-2 learns the optimal balance between data and physics losses during training. Two log-variance parameters (σ_data, σ_phys) are optimized by gradient descent, producing adaptive weights:

λ_data = exp(-log σ²_data), λ_phys = exp(-log σ²_phys)

When the physics constraint helps, σ_phys shrinks → higher weight. When it conflicts with the data, σ_phys grows → lower weight. No manual λ tuning needed.

**Code:** [`model/interval_comparison/pinn2_budyko.py`](model/interval_comparison/pinn2_budyko.py)

### Shared CNN-LSTM Backbone

All six models (baseline + 3 PINNs + 2 variants) use the exact same spatio-temporal architecture (~339K parameters), ensuring that performance differences isolate the effect of physics constraints:

1. **Spatial Encoder** — two Conv2d blocks (in → 32 → 64 channels), AdaptiveAvgPool2d(4×4) → 1,024 features/time step
2. **Temporal Aggregator** — two-layer LSTM (hidden=64, dropout=0.5), final hidden state only
3. **Regression Head** — Linear(64→64) → Activation → Dropout(0.5) → Linear(64→D_out)

Activation functions differ: ReLU (baseline, PINN-1), SiLU (PINN-2, PINN-3) — chosen per model during validation.

**Code:** [`model/interval_comparison/`](model/interval_comparison/)

### Multi-Seed Evaluation Framework

60 models trained across 5 seeds × 4 architectures × 3 temporal resolutions. Outputs include per-seed predictions, history logs, and aggregated metrics. Key finding: single-seed baseline results can vary by ±0.0556 in R² — multi-seed evaluation is essential for reliable ranking.

**Orchestrator:** [`master_runner.py`](master_runner.py)
**Training scripts:** [`scripts/training/`](scripts/training/)

### Simple Baselines

Persistence (ΔH_t = ΔH_{t-1}) and climatology (training mean ΔH) computed from test data to contextualize model performance:

| Baseline | RMSE | R² |
|----------|------|----|
| Persistence | 0.0874 | 0.017 |
| Climatology | 0.0878 | 0.000 |
| **PINN-3** | **0.0486** | **0.6939** |

### Diebold-Mariano Statistical Testing

Pairwise DM tests on baseline seed pairs confirm statistically significant inter-seed variability (DM = -2.4 to +2.7, p<0.01 for 7/10 pairs), quantitatively validating the paper's central claim that single-seed evaluations are unreliable.

**Computation script:** [`scripts/evaluation/add_r2_metrics.py`](scripts/evaluation/add_r2_metrics.py)

---

## Quick Start

**Auto (recommended):**
```bash
uv run setup.py          # CUDA auto-detection, venv, deps, dataset
uv run setup.py --cpu    # Force CPU-only PyTorch
```

**Manual:**
```bash
python download_dataset.py          # Kaggle download
pip install -r requirements.txt     # Dependencies
python master_runner.py --stage all --resolution monthly
```

**Compile paper:**
```bash
cd manuscript/latex && tectonic main.tex     # or upload to Overleaf
```

---

## Project Structure

```
paper_v2/
├── manuscript/latex/            # Paper source (IEEEtran)
│   ├── main.tex                 # Full paper (~409 lines, 6 pages)
│   ├── references.bib           # 19 references
│   └── IEEEtran.cls
├── model/                       # Model implementations
│   ├── baseline_lstm/           # Unconstrained CNN-LSTM
│   ├── PINN1_Water_balance/     # PINN-1 (mass conservation)
│   ├── PINN1v2/                 # PINN-2 (Budyko + River variant)
│   ├── pinn3/                   # PINN-3 (Priestley-Taylor)
│   └── interval_comparison/     # All-in-one experiment runner
├── scripts/
│   ├── training/                # Multi-seed, retrain, walk-forward
│   ├── evaluation/              # Metrics, DM tests, R² tables
│   └── plotting/                # All paper figures
├── results/                     # Experiment outputs (960+ CSVs/JSONs)
│   └── master_suite/            # Organized by experiment
├── figures/                     # Generated figures (PNG)
├── cluster/                     # Distributed training (optional)
├── setup.py                     # One-command install
├── download_dataset.py          # Kaggle dataset downloader
├── requirements.txt             # Python dependencies
└── README.md
```

## Dataset

13 gridded channels, 0.25° resolution, 110×90 grid, 1993-2026:

| Source | Organization | Variables |
|--------|-------------|-----------|
| ERA5 | ECMWF Copernicus | Precipitation, temperature, evaporation, radiation, pressure, wind, dewpoint |
| ERA5-Land | ECMWF Copernicus | Basin-scale precipitation, temperature, evapotranspiration, soil moisture |
| GLDAS Noah | NASA GSFC | Snow water equivalent, surface & sub-surface runoff |
| DAHITI (ID 39) | DGFI-TUM | Caspian Sea absolute water level (satellite altimetry) |
| GloFAS | ECMWF Copernicus | Volga River discharge |

Download from [Kaggle](https://www.kaggle.com/datasets/chingchonghaha/caspian-sea-hydroclimatic-dataset-1993-2026). License: CC BY 4.0.

---

## Key Results Summary

| Rank | Model | RMSE (m) | R² |
|------|-------|----------|----|
| 1 | PINN-3 | 0.0486 ± 0.0003 | 0.6939 ± 0.0032 |
| 2 | PINN-2 (retrained) | 0.0521 ± 0.0029 | 0.6470 ± 0.0396 |
| 3 | PINN-1 (retrained, λ=1.0) | 0.0536 ± 0.0017 | 0.6268 ± 0.0230 |
| 4 | PINN-2 River | 0.0543 ± 0.0023 | 0.6174 ± 0.0328 |
| 5 | Baseline (CNN-LSTM) | 0.0568 ± 0.0038 | 0.5803 ± 0.0556 |
| — | Hybrid (single-seed) | 0.0520 | 0.6491 |

**Finding:** Physics constraints reduce seed sensitivity by 2-17×. At 10-day resolution, only PINNs achieve positive R².

---

## Citation

```bibtex
@inproceedings{ashirbek2026pinn,
  title={Physics-Informed Neural Networks for Predicting Caspian Sea Level Variations},
  author={Ashirbek, Bekzat and Mirseiit, Arlan and Nurseit, Amir and Rakhimzhanova, Anar},
  booktitle={IEEE 3rd Int. Student Conf. on Digital Generation (DG)},
  year={2026}
}
```

**License:** Code — MIT. Data — CC BY 4.0.
