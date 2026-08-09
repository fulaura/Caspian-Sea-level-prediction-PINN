# Structured Summary: Caspian Sea Level PINN Diploma Thesis

## Basic Information
- **Title:** Development of Physics-Informed Neural Networks (PINNs) for Predicting Variations in the Caspian Sea Level
- **Authors:** Ashirbekov Bekzat, Mirseiit Arlan, Nurseit Amir
- **Institution:** Astana IT University, School of Artificial Intelligence and Data Science
- **Program:** 6B06103 – Big Data Analysis
- **Supervisor:** Dr. Rakhimzhanova Anar
- **Date:** June 2026
- **Pages:** 89

---

## 1. ABSTRACT

The Caspian Sea is a closed inland basin whose water level depends on the balance between atmospheric forcing, river inflow, evaporation, and basin storage. Reliable prediction is complicated by data fragmentation across satellite, reanalysis, river-discharge, and land-surface products.

This study integrates **ERA5, ERA5-Land, GLDAS, DAHITI, and Volga discharge data** into a unified hydroclimatic dataset covering **1993–2026** and uses it to develop and compare three PINN formulations:
- **PINN-1:** Water-balance PINN
- **PINN-2:** Budyko-constrained runoff-partitioning PINN
- **PINN-3:** Penman/Priestley–Taylor energy-balance PINN

All share a **CNN–LSTM backbone** and are compared against a **no-physics CNN–LSTM baseline**.

**Key results:** Daily resolution was unstable and excluded; monthly aggregation proved most reliable. The no-physics baseline achieved the lowest monthly **RMSE (0.0192 m)**, whereas PINN-3 achieved the highest **correlation (0.8279)** and **R² (0.6841)**. A five-seed test confirmed PINN-3 stability (**RMSE = 0.0486 ± 0.0003**). The 2018–2026 test period corresponds to a stronger declining regime than most of the training period.

**Keywords:** Caspian Sea level, Physics-Informed Neural Networks, PINN, CNN–LSTM, water balance, Budyko, Penman–Monteith, ERA5, GLDAS, DAHITI.

---

## 2. CASPIAN SEA HYDROLOGICAL DESCRIPTION

### Physical Setting
- The Caspian Sea is the world's largest inland body of water — a **closed basin** not connected to the ocean.
- Water level depends on regional weather patterns, river flows, and how water/energy move within the basin.
- The **drainage basin is roughly 10× larger than the sea surface**, fed by approximately **130 rivers**, with the **Volga River supplying the dominant share** of total inflow.
- Monthly changes are caused mainly by: rainfall on the lake surface, evaporation, river inflow, and seasonal water storage in surrounding land.

### Observed Decline
- Water level dropped by **~1.5 m between 1996 and 2021**, mainly due to less Volga River inflow and higher evaporation from rising temperatures.
- CMIP6 projections indicate continuing substantial decline under intermediate and high-emission scenarios during the 21st century.
- Projected evaporation increases are expected to exceed precipitation increases, resulting in a progressively more negative water balance.

### Data Challenges
- Information is scattered across different sources with inconsistent resolutions and formats.
- CMIP6 models with incorrect Caspian Sea surface representation had to be excluded because surface area directly affects water-balance fluxes.
- The shallow northern Caspian is especially sensitive to level changes — small vertical changes produce large horizontal shoreline shifts.

### Data Sources Integrated
| Source | Type | Variables |
|--------|------|-----------|
| ERA5 | Atmospheric reanalysis | Precipitation, evaporation, temperature, dewpoint, wind (u10, v10), surface pressure, net radiation |
| ERA5-Land | Land-surface reanalysis | Basin precipitation, temperature, humidity, radiation, PET, reference ET, soil moisture |
| GLDAS (Noah V2.0/V2.1) | Hydrological data | Snow water equivalent, surface runoff, sub-surface runoff |
| DAHITI (ID 39) | Satellite altimetry | Caspian Sea water level (TOPEX/Poseidon, Jason-1/2/3, Sentinel-6A, pass 133) |
| GloFAS | River discharge | Volga River discharge |

---

## 3. LITERATURE REVIEW

### Key References and Findings

1. **Raissi et al. (2019)** [6] — Foundational PINN framework: neural network trained to minimize both data misfit and governing PDE residual.

2. **Samant & Prange (2023)** [7] — CMIP6-based projections of 21st century Caspian Sea level decline. Excluded models with unrealistic Caspian representation. Evaporation increase > precipitation increase drives negative water balance.

3. **Eftekhari et al. (2024)** [3] — Machine learning (Decision Tree, Random Forest, MARS) for southern Caspian coast using GRACE/GRACE-FO. MARS best performer. Statistical approach without basin-wide water-balance structure.

4. **Saraceni et al. (2025)** [8] — Water mass balance procedure using ERA5-Land + lake-level observations to reconstruct closed-lake level changes. Precipitation biases in ERA5-Land identified as limitation.

5. **Niedda et al. (2014)** [5] — 85-year physically-based simulation of closed catchment-lake system in semi-arid Mediterranean. Precipitation dominant control on multi-year streamflow variability.

6. **Huang et al. (2025)** [1] — Physics-informed sea-surface-height prediction in South China Sea using geostrophic constraints and land masking. Stronger physical constraint does not always produce better forecasts.

7. **Bhasme et al. (2021)** [9] — Physics-informed ML for hydrology preserving water-balance consistency without full PDE specification.

8. **Cheng et al. (2025)** [10] — Global runoff partitioning using Budyko-constrained machine learning. Runoff and baseflow estimation improve under physically plausible partitioning regime.

9. **Cuomo et al. (2022)** [11] — PINN review: identifies weak theoretical guarantees, optimization instability, loss balancing challenges.

10. **Podina et al. (2023)** [2] — Universal PINNs combining known operator structure with neural component for unknown terms, using symbolic regression for hidden dynamics.

11. **Liu et al. (2025)** [12] — Time-varying Budyko parameters via Ensemble Kalman Filter and ML. Runoff partitioning not constant over time.

12. **Esmaeilzadeh & Amirzadeh (2024)** [4] — Replication study confirming physics-guided approach adds structure and interpretability.

### Key Equations from Literature

**Generic PINN Residual:**
```
f(t, x) = u_t + N[u; θ]                    (2.1)
```

**Generic PINN Objective Function:**
```
L_total = L_data + λ_phys L_phys + λ_bc L_bc + λ_ic L_ic    (2.2)
```

**Closed-Basin Water Balance:**
```
dS/dt = P + Q_in − E − Q_out + G           (2.3)
```

**Storage-to-Level Conversion:**
```
ΔH ≈ ΔS / A                                (2.4)
```

**Budyko-Type Evapotranspiration Partitioning (Fu formulation):**
```
ET/P = 1 + φ − (1 + φ^ω)^(1/ω),   φ = PET/P    (2.5)
```

**Hydrological Mass-Consistency Relation:**
```
P = ET + Q + ΔS                            (2.6)
```

**Ensemble Kalman Update:**
```
x_{a,t} = x_{f,t} + K_t (y_t − H x_{f,t})   (2.7)
```

**Geostrophic Balance:**
```
u_g = −(g/f) ∂ζ/∂y,   v_g = (g/f) ∂ζ/∂x   (2.8)
```

**MARS Prediction Function:**
```
ŷ = β₀ + Σₘ βₘ hₘ(x)                       (2.10)
```

### Identified Research Gaps
1. No well-established PINN specifically for the Caspian Sea
2. Missing framework linking catchment runoff, basin water balance, and whole-sea level
3. Dynamic lake geometry underrepresented
4. Combining multiple heterogeneous data sources with explicit uncertainty
5. Constraint selection and weighting unresolved
6. Fixed hydrological relationships too rigid for changing climate

---

## 4. METHODOLOGY — ALL EQUATIONS

### Research Design
Two coupled tasks: (i) unified monthly hydroclimatic dataset construction, (ii) three PINN formulations compared against no-physics baseline.

### Data Split
| Split | Period | Length | Share |
|-------|--------|--------|-------|
| Training | Jan 1993–Dec 2014 | 22 years | 67% |
| Validation | Jan 2015–Dec 2017 | 3 years | 9% |
| Testing | Jan 2018–Feb 2026 | 8 years | 24% |

### Spatial Domain
- Lake bounding box: 47.5°N, 45.5°W, 36.5°S, 54.5°E
- Grid resolution: 110 × 90 spatial grid
- Monthly dataset: (396, 110, 90); 10-day: (1193, 110, 90); Daily: (12103, 110, 90)

### Core Predictor Variables
- **Atmospheric:** Total precipitation, evaporation, 2m air temperature, 2m dewpoint temperature, surface pressure
- **Energy/Wind:** Surface net solar radiation, surface net thermal radiation, 10m zonal/meridional wind
- **Hydrological:** Surface runoff, sub-surface runoff, snow water equivalent
- **Spatial mask:** Land–sea mask
- **River/Target:** Volga discharge (m³/s), absolute water level, delta water level

### Aridity Index (used in Budyko)
```
AI_budyko = E_pot_basin / P_basin              (3.1)
```

### Total Runoff (conceptual)
```
Q_total = Q_surface + Q_baseflow              (3.2)
```

### Proxy Inflow
```
Q_in,proxy ≈ P_basin − ET_basin               (3.3)
```

### PINN-1: Water Balance / Mass Conservation

**Physical water-level change estimate:**
```
Δh_phys = (P_lake − α_E·E_lake) + β_volga·(Q_volga·Δt)/A_lake
          + γ_other·max(0, P_basin − PET_basin)          (3.4)
```
Where:
- α_E = learnable evaporation correction factor
- β_volga = scaling factor for Volga discharge contribution
- γ_other = learnable scaling for residual basin runoff
- A_lake = constant reference surface area (~371,000 km²)

**Physics loss:**
```
L_phys = MSE(Δh_pred, Δh_phys)                 (3.5)
```

**Total objective:**
```
L_total = L_data + λ_phys·L_phys               (3.6)
```

**Area sensitivity:**
```
δ(Δh_Q)/Δh_Q ≈ −δA/A_lake
```
A 5% area reduction underestimates inflow-to-level contribution by ~5%.

### PINN-2: Budyko-Constrained Model

**Aridity index (time-varying):**
```
φ_t = PET_basin,t / P_basin,t                   (3.7)
```

**Budyko ET estimate:**
```
ET_budyko,t = [φ_t / (1 + φ_t^n_t)^(1/n_t)] · P_basin,t    (3.8)
```
Where n_t = time-varying Budyko shape parameter predicted by the model.

**Physical components:**
```
runoff_contrib_t = (P_basin,t − ET_budyko,t) · scale_t    (3.9)
lake_contrib_t = P_lake,t − E_lake,t                      (3.10)
Δh_phys,t = runoff_contrib_t + lake_contrib_t              (3.11)
```
Where scale_t = dynamic runoff-to-lake routing coefficient.

**Budyko physics loss:**
```
L_phys = MSE(Δh_pred, Δh_phys)                 (3.12)
```

**Total objective:**
```
L_total = L_data + λ_phys·L_phys               (3.13)
```

### PINN-3: Penman/Priestley–Taylor Energy-Balance Constraint

**Saturation vapor pressure:**
```
e_s(T_t) = 0.6108·exp(17.27·T_t / (T_t + 237.3))      (3.14)
```

**Actual vapor pressure:**
```
e_a(T_d,t) = 0.6108·exp(17.27·T_d,t / (T_d,t + 237.3))  (3.15)
```

**Vapor pressure deficit:**
```
VPD_t = max(0, e_s(T_t) − e_a(T_d,t))                  (3.16)
```

**Slope of saturation vapor pressure curve:**
```
Δ_t = 4098·e_s(T_t) / (T_t + 237.3)²                   (3.17)
```

**Psychrometric constant (FAO-56 simplified):**
```
γ_t = 0.665 × 10⁻³·P_t                                 (3.18)
```
Where P_t is surface pressure in kPa.

**Priestley–Taylor diagnostic evaporation:**
```
E_lake,t^phys = α_PT,t·[Δ_t/(Δ_t+γ_t)]·R_n,t^eq          (Radiation term)
              + C_wind,t·[γ_t/(Δ_t+γ_t)]·f(U_t)·VPD_t   (Aerodynamic term)   (3.19)
```
Where:
- α_PT,t ≥ 1.0 (softplus-constrained, reference free-water value ≈1.26)
- C_wind,t > 0 (aerodynamic transfer scale)
- f(U_t) = 1 + 0.536·U_t (wind function)
- R_n,t^eq = net radiation converted to water-equivalent depth (λ_v·ρ_w ≈ 2.5 × 10⁹ J/m³)

**Physical water-level estimate for PINN-3:**
```
Δh_phys,t = (P_lake,t − E_lake,t^phys)
            + γ_runoff,t·max(0, P_basin,t − PET_basin,t)      (3.20)
```

**Physics loss:**
```
L_phys = MSE(Δh_pred,t, Δh_phys,t)                  (3.21)
```

**Total objective:**
```
L_total = L_data + λ_phys·L_phys                     (3.22)
```
λ_phys = 1.0, identical to PINN-2.

### Training Configuration
- **Optimizer:** AdamW (weight decay 0.05)
- **Learning rate:** Cosine annealing from 3×10⁻⁴
- **Gradient clipping:** 1.0
- **Epochs:** 300 with early stopping (patience 50)
- **Dropout:** 0.5
- **Physics weight:** λ_phys = 1.0 (fixed, no sensitivity analysis performed)

### CNN–LSTM Backbone
- Input: (batch, 6 time steps, 13 channels, 110, 90)
- CNN: 2 convolutional blocks (13→32→64 channels, SiLU/ReLU, max-pooling, adaptive avg pooling to 4×4)
- Feature dimension per time step: 1024 (64 × 4 × 4)
- LSTM: 2 layers, hidden size 64
- Regression head: Linear → SiLU → Dropout → Output
- Outputs: Δh (all), + physics parameters per formulation

---

## 5. RESULTS — ALL TABLES WITH NUMBERS

### 5.1 Temporal Resolution Screening (PINN-2)

**Table 4.1: Daily Dataset (Excluded)**
| Metric | Value |
|--------|-------|
| Resolution | 1 day |
| RMSE | 0.0147 |
| Correlation | -0.0330 |
| R² | -1.2358 |
| Initial val loss | 9.65 × 10⁻⁵ |
| Final val loss | 2.15 × 10⁻⁴ |

**Table 4.2: 10-Day Dataset**
| Metric | Value |
|--------|-------|
| Resolution | 10 days |
| RMSE | 0.0502 |
| Correlation | 0.3780 |
| R² | 0.0126 |
| Initial val loss | 2.49 × 10⁻³ |
| Final val loss | 2.98 × 10⁻³ |

**Table 4.3: Monthly Dataset**
| Metric | Value |
|--------|-------|
| Resolution | 1 month |
| RMSE | 0.0555 |
| Correlation | 0.7907 |
| R² | 0.6007 |
| Initial val loss | 6.91 × 10⁻³ |
| Final val loss | 3.08 × 10⁻³ |

### 5.2 Main Comparative Result — Monthly Ablation Study

**Table 4.4: Monthly Comparison (All Models)**
| Model | Physics Constraint | λ_phys | Resolution | RMSE | Correlation | R² |
|-------|-------------------|--------|------------|------|-------------|-----|
| CNN–LSTM baseline | None | 0 | 1 month | **0.0192** | 0.7945 | 0.6305 |
| PINN-1 | Water balance | 1 | 1 month | 0.0196 | 0.7846 | 0.6154 |
| PINN-2 | Budyko | 1 | 1 month | 0.0555 | 0.7907 | 0.6007 |
| PINN-3 | Penman / energy bal. | 1 | 1 month | 0.0494 | **0.8279** | **0.6841** |

**Key finding:** No-physics baseline achieves lowest RMSE. PINN-3 achieves highest correlation and R².

### 5.3 10-Day Dataset Comparison

**Table 4.5: PINN-1 on 10-Day**
| Metric | Value |
|--------|-------|
| RMSE | 0.0433 |
| Correlation | 0.4957 |
| R² | 0.2211 |

**Table 4.6: PINN-2 on 10-Day**
| Metric | Value |
|--------|-------|
| RMSE | 0.0502 |
| Correlation | 0.3780 |
| R² | 0.0126 |

**Table 4.7: PINN-3 on 10-Day**
| Metric | Value |
|--------|-------|
| RMSE | 0.0432 |
| Correlation | 0.4989 |
| R² | 0.2469 |

**Table 4.8: 10-Day Comparative Summary**
| Model | Resolution | RMSE | Correlation | R² |
|-------|-----------|------|-------------|-----|
| PINN-1 | 10 days | 0.0433 | 0.4957 | 0.2211 |
| PINN-2 | 10 days | 0.0502 | 0.3780 | 0.0126 |
| PINN-3 | 10 days | **0.0432** | **0.4989** | **0.2469** |

### 5.4 Monthly Dataset Comparison

**Table 4.9: PINN-1 on Monthly**
| Metric | Value |
|--------|-------|
| RMSE | 0.0196 |
| Correlation | 0.7846 |
| R² | 0.6154 |

**Table 4.10: PINN-2 on Monthly**
| Metric | Value |
|--------|-------|
| RMSE | 0.0555 |
| Correlation | 0.7907 |
| R² | 0.6007 |

**Table 4.11: PINN-3 on Monthly**
| Metric | Value |
|--------|-------|
| RMSE | 0.0494 |
| Correlation | 0.8279 |
| R² | 0.6841 |

**Also reported for PINN-3 reconstructed absolute water level:**
- MAE: 0.0895
- RMSE: 0.1088
- R²: 0.9401

**Table 4.12: Final Comparison (All Resolutions)**
| Model | Resolution | RMSE | Correlation | R² |
|-------|-----------|------|-------------|-----|
| PINN-1 | 10 days | 0.0433 | 0.4957 | 0.2211 |
| PINN-2 | 10 days | 0.0502 | 0.3780 | 0.0126 |
| PINN-3 | 10 days | 0.0432 | 0.4989 | 0.2469 |
| PINN-1 | 1 month | **0.0196** | 0.7846 | 0.6154 |
| PINN-2 | 1 month | 0.0555 | 0.7907 | 0.6007 |
| PINN-3 | 1 month | 0.0494 | 0.8279 | 0.6841 |

### 5.5 PINN-3 Multi-Seed Stability

**Table 4.13: Five-Seed Stability of Monthly PINN-3**
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| RMSE (ΔH), m | 0.0486 | 0.0003 | 0.0483 | 0.0488 |
| Correlation (ΔH) | 0.8341 | 0.0016 | 0.8319 | 0.8361 |
| R² (ΔH) | 0.6939 | 0.0032 | 0.6909 | 0.6981 |
| RMSE (H), m | 0.1857 | 0.0397 | 0.1260 | 0.2266 |
| R² (H) | 0.8194 | 0.0712 | 0.7405 | 0.9198 |

PINN-3 ΔH metrics are highly stable across seeds (RMSE std = 0.0003 m). Absolute-level reconstruction shows larger variability.

### 5.6 Key Interpretive Notes
- **Regime shift:** Test period (2018–2026) has a stronger declining trend than training period (1993–2014). High absolute-level R² partly reflects trend-following ability.
- **ΔH metrics are more conservative indicators** of month-to-month predictive skill than reconstructed absolute water level H.
- **Constant A_lake assumption** (~371,000 km²) introduces scaling uncertainty in the physics residual, especially during strong decline.
- **PINN-2 dual role:** Used for both temporal-resolution screening and as Budyko reference — its metrics should not be interpreted as independent post-selection evidence.
- **Conceptual overlap between PINN-1 and PINN-2:** PINN-1's residual basin-surplus term partially overlaps with PINN-2's runoff-partitioning logic.

---

## 6. CONCLUSION

### Main Contributions
1. **Reproducible multi-source data integration pipeline** combining ERA5, ERA5-Land, GLDAS, DAHITI, and Volga discharge into unified monthly, 10-day, and daily datasets for 1993–2026.

2. **Empirical comparison** of three PINN formulations (water balance, Budyko runoff partitioning, Penman/Priestley–Taylor energy balance) against a no-physics CNN–LSTM baseline.

### Quantitative Outcome
- **No-physics CNN–LSTM baseline:** Lowest monthly RMSE = **0.0192 m**
- **PINN-3:** Highest correlation = **0.8279** and R² = **0.6841**
- **PINN-3 stability:** ΔH RMSE = **0.0486 ± 0.0003** across 5 seeds

### Methodological Insight
When the data-driven backbone already receives rich atmospheric, hydrological, and discharge inputs, the unconstrained network can capture substantial statistical signal directly from data. The numerical advantage of adding a physics-informed loss depends on careful calibration of loss weighting, residual scaling, and the match between constraints and observed hydroclimatic regime.

### Temporal Resolution Finding
- **Monthly** aggregation is the most reliable scale for physically constrained learning
- **Daily** resolution produced unstable predictions (excluded)
- **10-day** resolution preserved intermediate variability but had limited explanatory power

### Limitations Identified
1. Constant reference lake-area assumption (should be replaced by dynamic A(H) or bathymetry-based storage–elevation)
2. Conceptual overlap between PINN-1 residual basin term and PINN-2 Budyko partitioning
3. PINN-2 used for both screening and as reference (not independent post-selection)
4. Fixed physics-loss weight (λ_phys = 1.0) without sensitivity analysis
5. Multi-seed evaluation only for PINN-3 (PINN-1, PINN-2, baseline trained from single seed)
6. Volga discharge treated differently across formulations (explicit in PINN-1, feature-only in PINN-2/3)
7. Source heterogeneity (ERA5, ERA5-Land, GLDAS, DAHITI, GloFAS each with own uncertainty profile)

### Future Work Recommendations
- Repeated-seed training for all four models
- Retraining after fixing temporal resolution
- Walk-forward/expanding-window validation on non-stationary climate windows
- Sensitivity analysis of λ_phys
- Dynamic lake geometry (A(H) or bathymetry-based storage–elevation)
- Uncertainty quantification (Monte Carlo dropout)
- Ablation studies of individual physical mechanisms
- Hybrid PINN combining water balance, Budyko, and Penman constraints
- Direct validation of learned physical coefficients against independent hydrological references

---

## 7. BIBLIOGRAPHY (Key References)

| # | Citation | Focus |
|---|----------|-------|
| [1] | Huang et al. (2025), *Remote Sensing* 17(23):3838 | Physics-informed SSH prediction, South China Sea |
| [2] | Podina et al. (2023), *ICML/PMLR 202* | Universal PINNs, symbolic operator discovery |
| [3] | Eftekhari et al. (2024) | ML for southern Caspian using GRACE/GRACE-FO |
| [4] | Esmaeilzadeh & Amirzadeh (2024), *arXiv:2402.13911* | Replication of physics-guided hydrological ML |
| [5] | Niedda et al. (2014), *J. Hydrology* 517:732–745 | Closed catchment-lake hydrological simulation |
| [6] | Raissi et al. (2019), *J. Comp. Physics* 378:686–707 | Foundational PINN framework |
| [7] | Samant & Prange (2023), *Comm. Earth & Env.* | CMIP6 Caspian Sea level decline projections |
| [8] | Saraceni et al. (2025), *J. Hydrometeorology* 26(1):49–67 | ERA5-Land water balance for closed lakes |
| [9] | Bhasme et al. (2021), *arXiv:2104.11009* | Physics-informed ML for hydrology |
| [10] | Cheng et al. (2025), *Water Resources Research* 61(10) | Budyko-constrained global runoff partitioning |
| [11] | Cuomo et al. (2022), *arXiv:2201.05624* | PINN review — challenges and future |
| [12] | Liu et al. (2025), *J. Hydrology Regional Studies* 59:102348 | Time-varying Budyko parameters |
