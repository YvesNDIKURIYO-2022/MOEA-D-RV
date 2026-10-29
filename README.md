# MOEA/D-RV

**A Risk-Guided Mutation Strategy with Hybrid Decomposition and Risk Control for Dynamic Multi-Objective Optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Paper-Under%20Review-lightgrey.svg)](#citation)

---

## Abstract

MOEA/D-RV is a dynamic multi-objective evolutionary algorithm (DMOEA) for tracking time-varying Pareto fronts under environmental change. Its principal contribution is a **risk-guided mutation strategy** that scales mutation intensity per decision variable according to a fitness-derivative estimate of that variable's sensitivity to change. The algorithm additionally integrates hybrid PBI/Tchebycheff decomposition with adaptive switching, Monte Carlo scenario sampling, mean–variance risk control, hypervolume-guided selection, predictive change detection, and adaptive niching. The manuscript has been submitted to *Applied Soft Computing* (2026) and is currently under review.

---

## Repository Layout

```
MOEA-D-RV/
├── .gitignore                                      # excludes results_*/, statistical_analysis/, __pycache__/, *.pyc, *.npz
├── LICENSE                                         # MIT
├── README.md                                       # this file
├── Set1-MOEAD-RV Vs Benchmarks.py                  # Set 1 — comparison vs 5 MOEA/D variants
├── Set2-MOEAD-RV Vs Benchmarks in multiple environment configurations.py   # Set 2 — Standard / Mild / Rapid
├── Set3-MOEAD-RV- Ablation study.py                # Set 3 — six-variant ablation
└── Set4-MOEAD-RV-Statistical Analysis.py           # Set 4 — post-hoc statistical analysis of Sets 1–3
```

Each script creates its own output folder on first run:

| Script | Writes to | Contents |
|---|---|---|
| Set 1 | `results_set1/` | per-run JSON, IGD and MHV convergence plots per problem |
| Set 2 | `results_set2/` | per-run JSON, IGD convergence plots per (problem, configuration) |
| Set 3 | `results_set3/` | per-variant JSON, ablation IGD plots, component-contribution bar chart |
| Set 4 | `statistical_analysis/` | summary JSON per set, Friedman rank bar chart, critical-difference diagram, p-value heatmap, Wilcoxon improvement chart, MIGD boxplot |

**Note.** The `.gitignore` excludes the four `results_*`/`statistical_analysis/` output folders because the raw output from 30 runs × 14 problems × 6 algorithms is several gigabytes and is reproducible from the scripts. Do not delete these folders locally — Set 4 depends on the three result folders being present.

---

## Experiment Campaigns

| Set | Script | Reproduces |
|---|---|---|
| **Set 1** | `Set1-MOEAD-RV Vs Benchmarks.py` | Six MOEA/D variants on CEC2018 DF1–DF14 → Tables 7–11 (MIGD, MHV, Friedman ranks, Wilcoxon tests) |
| **Set 2** | `Set2-MOEAD-RV Vs Benchmarks in multiple environment configurations.py` | Three environmental configurations (Standard, Mild, Rapid) → Tables 12–13 (win counts, Friedman ranks per configuration) |
| **Set 3** | `Set3-MOEAD-RV- Ablation study.py` | Six-variant ablation (Full, w/o Risk, w/o HV, w/o Prediction, w/o Niching, w/o Scenario) → Tables 14–16 (ablation ranks, Wilcoxon tests, component importance) |
| **Set 4** | `Set4-MOEAD-RV-Statistical Analysis.py` | Post-hoc statistical analysis consuming the three result folders from Sets 1–3 — does not produce new experimental data |

All experiments use **30 independent runs** per problem.

---

## Experimental Results

The results below correspond to the tables in the submitted manuscript. They are reproduced here for cross-verification.

### Set 1 — Standard Benchmark

| Algorithm | Friedman rank | Wins (of 14) |
|---|---|---|
| **MOEA/D-RV** | **1.86** | **10/14** |
| MOEA/D-KNN | 2.21 | 3/14 |
| MOEA/D-PPS | 3.29 | 0/14 |
| MOEA/D-AGR | 3.36 | 1/14 |
| MOEA/D-HSS | 4.93 | 0/14 |
| MOEA/D | 5.36 | 0/14 |

Friedman test: χ² = 41.62, p = 7.02 × 10⁻⁸. All five pairwise Wilcoxon comparisons against MOEA/D-RV are statistically significant after Holm–Bonferroni correction (p < 0.05).

### Set 2 — Environmental Configurations

| Configuration | n_t | τ_t | MOEA/D-RV wins | Friedman rank |
|---|---|---|---|---|
| Standard | 10 | 10 | 10/14 | 1.86 |
| Mild | 10 | 20 | 10/14 | 1.93 |
| Rapid | 5 | 10 | 11/14 | 1.79 |

MOEA/D-RV ranks first under all three configurations, indicating that its advantage over the competing algorithms is not sensitive to the change frequency or severity of the environment.

### Set 3 — Ablation Study

| Variant | Average MIGD improvement | p-value | Significant |
|---|---|---|---|
| **w/o Risk** | **+14.3%** | **0.012** | **Yes** |
| w/o Prediction | +13.1% | 0.108 | No |
| w/o Niching | +1.2% | 0.476 | No |
| w/o Scenario | +1.2% | 0.768 | No |
| w/o HV | −288.5% | 0.979 | No |

Risk-guided mutation is the **dominant and only individually significant component**. The hypervolume-guided selection exhibits a geometry-dependent limitation on the discontinuous Pareto front of DF8 and on the three-objective problems DF10–DF14, as documented in Section 5.4 of the manuscript.

Friedman ranks for the six ablation variants (lower is better): w/o HV 2.14, Full 3.21, w/o Niching 3.21, w/o Scenario 3.29, w/o Prediction 4.00, w/o Risk 5.14.

### Set 4 — Statistical Analysis

Set 4 does not produce new experimental data; it consumes the JSON files in `results_set1/`, `results_set2/`, and `results_set3/` and generates:

- `statistical_analysis/set1/summary.json` — Friedman test, ranks, Wilcoxon comparisons, win/loss counts for Set 1
- `statistical_analysis/set2/{Standard,Mild,Rapid}/summary.json` — same for each configuration
- `statistical_analysis/set3/summary.json` — Friedman test, ranks, Wilcoxon tests, component importance for the ablation study
- `statistical_analysis/combined_summary.json` — consolidated summary
- `statistical_analysis/*/plots/` — Friedman rank bar chart, critical-difference diagram, p-value heatmap, Wilcoxon improvement chart, MIGD boxplot

The statistical conclusions reported in the manuscript (χ² = 41.62, p = 7.02 × 10⁻⁸ for Set 1; Friedman rank 1.86; all five Wilcoxon tests significant after Holm–Bonferroni correction) are reproduced by running Set 4 on the Set 1 result JSON.

---

## Algorithmic Components

### Core Innovation

**Risk-guided mutation.** For each decision variable x_j and each subproblem k, the local risk ρ_{k,j}(t) is estimated by finite differences of the risk-aware scalarized fitness. The risk is aggregated over a sliding window of s = 5 generations and all N subproblems, normalized to [0, 1], and mapped to a per-variable mutation intensity σ_j(t) = σ_base · (1 + α · R̂_j(t)), with σ_base = 0.15 and α = 1.5. The mutation distribution is symmetric (polynomial mutation with η_m = 20); the adaptation operates exclusively on the perturbation magnitude, not on its sign.

### Supporting Components

| Component | Description |
|---|---|
| Hybrid decomposition | Adaptive switching between PBI (θ_PBI = 5.0) and Tchebycheff scalarization, governed by a hypervolume-stagnation criterion |
| Scenario sampling | Monte Carlo evaluation with M(t) ∈ [5, 50] scenarios, sampled from N(ẑ*(t+1), σ²(t)) |
| Mean–variance risk control | Risk-aware fitness g_risk(x) = E[g(x)] + κ(t) · Var[g(x)], with κ_base = 0.5 |
| Hypervolume-guided selection | Population update by exact HVC for m = 2, Monte Carlo HVC (n_approx = 2000) for m = 3 |
| Predictive change detection | Linear regression on the last three ideal points; detection threshold ε = 0.005 |
| Adaptive niching | Shared fitness with adaptive niche radius (σ_share updated via archive-quality rule) |

---

## Parameters

### Common Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Population size | N | 100 |
| Neighborhood size | T | 20 |
| Independent runs | R | **30** |
| Number of environmental changes | K | 30 |
| Warm-up generations | — | 50 |
| Change frequency (Standard / Mild) | n_t | 10 |
| Change frequency (Rapid) | n_t | 5 |
| Change severity (Standard / Rapid) | τ_t | 10 |
| Change severity (Mild) | τ_t | 20 |

### MOEA/D-RV Specific Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Base mutation intensity | σ_base | 0.15 |
| Risk sensitivity coefficient | α | 1.5 |
| Risk history length | s | 5 |
| Base scenario count | M_base | 30 |
| PBI penalty | θ_PBI | 5.0 |
| Base risk aversion | κ_base | 0.5 |
| Detection threshold | ε | 0.005 |
| Niche radius (initial) | σ_share | 0.15 |
| Monte Carlo HV samples (m = 3) | n_approx | 2000 |

---

## Benchmark Suite

The CEC2018 dynamic multi-objective optimization benchmark (DF1–DF14) is used throughout. Definitions are available from the pymoo library at <https://pymoo.org/problems/dynamic/df.html>.

| Problem | Objectives | PF shape | PS shape | Dynamic characteristics |
|---|---|---|---|---|
| DF1 | 2 | Convex | Linear | Linear change |
| DF2 | 2 | Convex | Linear | Nonlinear change |
| DF3 | 2 | Convex | Linear | Mixed linear–nonlinear |
| DF4 | 2 | Convex | Linear | Exponential change |
| DF5 | 2 | Concave | Rotational | Rotational change |
| DF6 | 2 | Concave | Rotational | Mixed rotational–linear |
| DF7 | 2 | Discontinuous | Linear | Discontinuous PF |
| DF8 | 2 | Convex | Discontinuous | Discontinuous PS |
| DF9 | 2 | Convex | Mixed | Mixed discontinuous |
| DF10 | 3 | Convex | Linear | Linear change |
| DF11 | 3 | Convex | Linear | Nonlinear change |
| DF12 | 3 | Concave | Rotational | Rotational change |
| DF13 | 3 | Mixed | Linear | Complex change |
| DF14 | 3 | Mixed | Linear | Abrupt change |

---

## Installation and Usage

### Prerequisites

Python 3.9 or later is required. The dependencies are:

```
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
tqdm >= 4.62.0
```

### Installation

```bash
git clone https://github.com/YvesNDIKURIYO-2022/MOEA-D-RV.git
cd MOEA-D-RV
pip install -r requirements.txt
```

### Running the Experiments

```bash
# Set 1 — Standard benchmark comparison (creates results_set1/)
python "Set1-MOEAD-RV Vs Benchmarks.py"

# Set 2 — Three environmental configurations (creates results_set2/)
python "Set2-MOEAD-RV Vs Benchmarks in multiple environment configurations.py"

# Set 3 — Ablation study (creates results_set3/)
python "Set3-MOEAD-RV- Ablation study.py"

# Set 4 — Statistical analysis (consumes the three result folders; creates statistical_analysis/)
python "Set4-MOEAD-RV-Statistical Analysis.py"
```

**Order matters.** Run Sets 1–3 before Set 4 — Set 4 reads the newest `.json` file in each result folder and will report "missing" for any set whose folder is empty.

### Reproducibility Notes

- All experiments use the seeds documented in each script. Different seeds will produce slightly different per-problem results, but the Friedman ranks and the statistical conclusions are stable across seeds in our preliminary tests.
- Total wall-clock time for Sets 1–3 is approximately 74 hours on a 12-core Intel Core i5-1340P with 32 GB RAM. Set 4 is post-processing only and runs in under a minute.
- The Monte Carlo hypervolume approximation for three-objective problems introduces slight run-to-run variation; this is the source of the geometry-dependent limitation documented in the ablation study.

---

## Computational Cost

Reported per run in the Standard configuration:

| Algorithm | Total evaluations | Wall-clock (h) |
|---|---|---|
| MOEA/D | 38,879 | 0.97 |
| MOEA/D-KNN | ≈ 38,879 | 0.89 |
| MOEA/D-PPS | ≈ 38,879 | 0.85 |
| MOEA/D-AGR | ≈ 42,000 | 0.87 |
| MOEA/D-HSS | ≈ 39,024 | 0.93 |
| **MOEA/D-RV** | **390,075** | **5.60** |

MOEA/D-RV consumes approximately ten times the objective evaluations and 5.8 times the wall-clock time of the baseline. The additional cost is attributable to finite-difference risk estimation, scenario sampling, and hypervolume computation. The manuscript (Section 5.5) documents the cost–accuracy trade-off and the mitigations (adaptive scenario count, risk computation caching, reduced niching frequency).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Ndikuriyo2026MOEADRV,
  title   = {A Risk-Guided Mutation Strategy with Hybrid Decomposition and
             Risk Control for Dynamic Multi-Objective Optimization (MOEA/D-RV)},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Under review}
}
```

The risk-control mechanisms are adapted from the authors' previous study:

```bibtex
@article{Ndikuriyo2026Robust,
  title   = {A multi-objective robust optimization based on evolutionary algorithm
             for container routing problem under risks and uncertainties},
  author  = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, D. D.},
  journal = {Applied Soft Computing},
  volume  = {197},
  pages   = {115176},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.

---

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 71971220) and the Natural Science Foundation of Hunan Province, China (Grant Nos. 2023JJ30710 and 2022JJ31020). The authors thank the School of Traffic and Transportation Engineering at Central South University for institutional support.

---

## Contact

For questions about the code or the manuscript, please contact:

- **Yves Ndikuriyo** — yvesndikuriyo@csu.edu.cn
- **Yinggui Zhang** — ygzhang@csu.edu.cn

Central South University, School of Traffic and Transportation Engineering, Changsha 410075, Hunan, China.
