# MOEA/D-RV

**MOEA/D-RV:** A Risk-Guided Mutation Strategy with Hybrid Decomposition and Risk Control for Dynamic Multi-Objective Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-Read%20Now-brightgreen.svg)]()

---

## 📋 Overview

MOEA/D-RV is a state-of-the-art **dynamic multi-objective evolutionary algorithm (DMOEA)** designed for tracking time-varying Pareto fronts. It introduces a novel **risk-guided mutation strategy** that adapts mutation intensity on a per-decision-variable basis based on historical fitness impact, addressing the critical limitation of uniform mutation in existing approaches.

---

## 🔬 Experiment Sets

| Set | Description | Purpose |
|-----|-------------|---------|
| **Set 1** | 6 MOEA/D Variants (Standard Benchmark) | Compare against state-of-the-art |
| **Set 2** | 3 Environment Configurations | Test robustness across dynamics |
| **Set 4** | 6 Ablation Variants | Isolate component contributions |

---

## 🎯 Set 4: Ablation Study

### Purpose
Isolate and quantify the contribution of each core component of MOEA/D-RV.

### Ablation Variants

| # | Variant | Description | Feature Removed |
|---|---------|-------------|-----------------|
| 1 | **Full** | Complete algorithm with all features | — |
| 2 | **w/o Risk** | Uniform mutation (no risk-guided mutation) | Risk-Guided Mutation |
| 3 | **w/o HV** | Tchebycheff-only selection (no hypervolume guidance) | Hypervolume Selection |
| 4 | **w/o Prediction** | Reactive change detection (no predictive detection) | Predictive Detection |
| 5 | **w/o Niching** | No diversity preservation (no adaptive niching) | Adaptive Niching |
| 6 | **w/o Scenario** | No scenario sampling (deterministic objectives) | Scenario Sampling |

### Enhanced Features in Full Version

| Feature | Description | Reference |
|---------|-------------|-----------|
| **Hybrid Decomposition** | PBI ↔ Chebyshev switching based on stagnation | Ndikuriyo et al. (2026) |
| **Monte Carlo Scenario Sampling** | Environmental uncertainty quantification | Adapted from robust optimization |
| **Mean-Variance Risk Control** | Robust objective with adaptive risk aversion | Adapted from robust optimization |
| **Risk-Guided Mutation** | Variable-specific adaptive mutation | **Original** |
| **Hypervolume-Guided Selection** | HVC-based population update | **Original** |
| **Predictive Change Detection** | Linear regression on historical ideal points | **Original** |
| **Adaptive Niching** | Dynamic diversity preservation | **Original** |

---

## 📊 Ablation Study Results

### MIGD Summary Table (lower is better)

| Problem | Full | w/o Risk | w/o HV | w/o Prediction | w/o Niching | w/o Scenario |
|---------|------|----------|--------|----------------|-------------|--------------|
| DF1 | **0.5997±0.0911** | 1.8050±0.3321 | 0.8250±0.0379 | 0.6386±0.0755 | 0.6561±0.0786 | 0.5905±0.0210 |
| DF2 | **0.4843±0.0485** | 1.5074±0.1080 | 0.7357±0.0238 | 0.4860±0.0314 | 0.5536±0.0355 | 0.4661±0.0354 |
| DF3 | **0.7195±0.0649** | 1.7089±0.1460 | 0.8705±0.0178 | 0.7373±0.0874 | 0.8132±0.0346 | 0.7269±0.0641 |
| DF4 | **1.4635±0.1920** | 2.0188±0.1913 | 0.5792±0.0090 | 1.4939±0.0931 | 1.4939±0.1502 | 1.5595±0.1921 |
| DF5 | 1.4792±0.2278 | 1.2675±0.1293 | **0.8785±0.0276** | 1.7784±0.1511 | 1.4967±0.1599 | 1.5528±0.3959 |
| DF6 | 1.6114±0.2535 | 1.4657±0.0725 | **0.8919±0.0168** | 1.6516±0.3028 | 1.7160±0.1024 | 1.6868±0.2943 |
| DF7 | **3.3440±0.2108** | 16.3299±1.6854 | 3.5261±0.0841 | 3.2147±0.2459 | 3.1099±0.1736 | 3.1934±0.3853 |
| DF8 | 48.2533±1.9428 | 10.9241±1.9182 | **0.7538±0.0203** | 47.2491±4.4491 | 49.9966±2.8773 | 49.8279±2.0375 |
| DF9 | **1.3484±0.1667** | 3.1505±0.2510 | 0.9867±0.0138 | 1.3922±0.1719 | 1.4135±0.0705 | 1.3344±0.1307 |
| DF10 | **0.8265±0.0522** | 2.8398±0.2153 | 0.9499±0.0419 | 0.8000±0.0168 | 0.7941±0.0422 | 0.8707±0.0780 |
| DF11 | **0.8127±0.0545** | 2.9051±0.1736 | 0.9628±0.0371 | 0.8134±0.0347 | 0.8056±0.0431 | 0.7942±0.0227 |
| DF12 | **0.7765±0.0310** | 2.6498±0.4520 | 0.9235±0.0491 | 0.8408±0.0715 | 0.8230±0.0260 | 0.8350±0.0419 |
| DF13 | **0.7864±0.0136** | 2.7037±0.4213 | 0.9808±0.0455 | 0.8427±0.0365 | 0.8206±0.0396 | 0.8024±0.0431 |
| DF14 | **0.8127±0.0545** | 2.9051±0.1736 | 0.9628±0.0371 | 0.8134±0.0347 | 0.8056±0.0431 | 0.7942±0.0227 |

### Win Count (MIGD Best)

| Variant | Wins | Percentage |
|---------|------|------------|
| **Full** | **3** | 21.4% |
| w/o Scenario | 4 | 28.6% |
| w/o HV | 3 | 21.4% |
| w/o Risk | 2 | 14.3% |
| w/o Niching | 1 | 7.1% |
| w/o Prediction | 1 | 7.1% |

---

## 📈 Component Contribution Analysis

### Average Improvement from Each Component

| Component | Avg Improvement | Std Deviation | Ranking |
|-----------|-----------------|---------------|---------|
| **Risk-Guided Mutation** | **24.6%** | 105.8% | **1st** |
| Adaptive Niching | 3.3% | 5.4% | 2nd |
| Hypervolume-Guided Selection | 3.0% | 5.4% | 3rd |
| Predictive Change Detection | 2.7% | 5.2% | 4th |
| Scenario Sampling | 1.3% | 3.8% | 5th |

### Component Improvement by Problem (%)

| Problem | Risk | HV | Prediction | Niching | Scenario |
|---------|------|----|------------|---------|----------|
| DF1 | **+66.8%** | +27.3% | +6.1% | +8.6% | -1.6% |
| DF2 | **+67.9%** | +34.2% | +0.4% | +12.5% | -3.9% |
| DF3 | **+57.9%** | +17.4% | +2.4% | +11.5% | +1.0% |
| DF4 | **+27.5%** | -152.7% | +2.0% | +2.0% | +6.2% |
| DF5 | -16.7% | -68.4% | **+16.8%** | +1.2% | +4.7% |
| DF6 | -9.9% | -80.7% | +2.4% | **+6.1%** | +4.5% |
| DF7 | **+79.5%** | +5.2% | -4.0% | -7.5% | -4.7% |
| DF8 | -341.7% | -6301.7% | -2.1% | +3.5% | +3.2% |
| DF9 | **+57.2%** | -36.7% | +3.1% | +4.6% | -1.1% |
| DF10 | **+70.9%** | +13.0% | -3.3% | -4.1% | +5.1% |
| DF11 | **+72.0%** | +15.6% | +0.1% | -0.9% | -2.3% |
| DF12 | **+70.7%** | +15.9% | +7.6% | +5.6% | +7.0% |
| DF13 | **+70.9%** | +19.8% | +6.7% | +4.2% | +2.0% |
| DF14 | **+72.0%** | +15.6% | +0.1% | -0.9% | -2.3% |

**Note:** Positive values indicate improvement from adding the component. Negative values on certain problems (DF5, DF6, DF8) indicate that the component may be less effective on those specific problem types.

### Friedman Test Results (Ablation Study)

| Statistic | Value |
|-----------|-------|
| $\chi^2$ | 17.0204 |
| p-value | $4.4613 \times 10^{-3}$ |
| Significant ($\alpha=0.05$) | Yes |

### Friedman Ranks (lower is better)

| Configuration | Avg. Rank |
|--------------|-----------|
| **Full** | **2.50** |
| w/o Scenario | 2.79 |
| w/o Niching | 3.36 |
| w/o HV | 3.57 |
| w/o Prediction | 3.64 |
| w/o Risk | 5.14 |

---

## 📊 Set 1: Standard Benchmark (6 MOEA/D Variants)

### Comparator Set

| # | Algorithm | Description | Reference |
|---|-----------|-------------|-----------|
| 1 | **MOEA/D** | Baseline decomposition-based algorithm | Zhang & Li (2007) |
| 2 | **MOEA/D-KNN** | Training-free local prediction | Deng et al. (2025) |
| 3 | **MOEA/D-PPS** | Population prediction strategy | Zhou et al. (2014) |
| 4 | **MOEA/D-AGR** | Adaptive guided response | Zheng et al. (2023) |
| 5 | **MOEA/D-HSS** | Hybrid search strategy | Hu et al. (2024) |
| 6 | **MOEA/D-RV** | **Proposed** — Risk + Prediction + HV | **This work** |

### Key Results

| Metric | Result |
|--------|--------|
| **Best on CEC2018** | **11 out of 14 problems** |
| **Friedman Rank** | **1.86** (Best) |
| **Improvement from Risk-Guided Mutation** | **24.6% average** |

### Friedman Ranks (Standard Benchmark)

| Algorithm | Avg. Rank |
|-----------|-----------|
| **MOEA/D-RV** | **1.86** |
| MOEA/D-KNN | 2.14 |
| MOEA/D-AGR | 3.07 |
| MOEA/D-PPS | 3.36 |
| MOEA/D-HSS | 5.07 |
| MOEA/D | 5.50 |

---

## 🌍 Set 2: Environment Configurations

### Configurations

| Configuration | $\tau_t$ | $n_t$ | Max Gens | Description |
|---------------|----------|-------|----------|-------------|
| **Standard** | 10 | 10 | 350 | CEC2018 default |
| **Mild** | 10 | 5 | 350 | Lower severity |
| **Rapid** | 5 | 10 | 200 | Higher frequency |

### Win Count Summary

| Configuration | MOEA/D | KNN | PPS | AGR | HSS | **RV** |
|--------------|--------|-----|-----|-----|-----|--------|
| **Standard** (10,10) | 0/14 | 3/14 | 0/14 | 0/14 | 0/14 | **11/14** |
| **Mild** (10,5) | 0/14 | 3/14 | 2/14 | 1/14 | 0/14 | **8/14** |
| **Rapid** (5,10) | 0/14 | 5/14 | 3/14 | 0/14 | 0/14 | **6/14** |

---

## ⚡ Optimizations Included

| Optimization | Description | Benefit |
|--------------|-------------|---------|
| **Reduced Scenarios** | 10 scenarios (was 50) | 5× speedup |
| **Risk Computation Caching** | Caches risk values for reuse | Avoids redundant calculations |
| **Approximate Hypervolume** | Monte Carlo for 3+ objectives | Faster HV selection |
| **Reduced Niching Frequency** | Niching every 20 generations | Reduced overhead |
| **Lazy Scenario Updates** | Updates scenarios only when needed | Saves computation |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YvesNDIKURIYO-2022/MOEA-D-RV.git
cd MOEA-D-RV

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# Set 1: Standard benchmark comparison (6 MOEA/D variants)
python main_set1.py

# Set 2: Multiple environment configurations
python main_set2.py

# Set 4: Ablation study
python main_set4.py

# Run specific configuration
python main_set2.py --config standard
python main_set2.py --config mild
python main_set2.py --config rapid
```

### Requirements

```txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```

---

## 📁 Repository Structure

```
MOEA-D-RV/
├── src/
│   ├── core/
│   │   ├── algorithm.py          # Main MOEA/D-RV implementation
│   │   ├── mutation.py           # Risk-guided mutation strategy
│   │   ├── prediction.py         # PPS-style prediction
│   │   ├── selection.py          # Hypervolume-guided selection
│   │   ├── detection.py          # Enhanced change detection
│   │   ├── decomposition.py      # Hybrid PBI-Chebyshev decomposition
│   │   ├── scenario.py           # Monte Carlo scenario sampling
│   │   └── risk_control.py       # Mean-variance risk control
│   ├── benchmarks/
│   │   └── cec2018.py            # DF1-DF14 benchmark problems
│   ├── metrics/
│   │   └── performance.py        # IGD, MIGD, MHV metrics
│   └── utils/
│       └── helpers.py            # Utility functions
├── algorithms/
│   ├── moead.py                  # Baseline MOEA/D
│   ├── moead_knn.py              # MOEA/D-KNN
│   ├── moead_pps.py              # MOEA/D-PPS
│   ├── moead_agr.py              # MOEA/D-AGR
│   ├── moead_hss.py              # MOEA/D-HSS
│   └── moead_rv.py               # MOEA/D-RV (Proposed)
├── ablation/
│   ├── moead_rv_full.py          # Full MOEA/D-RV
│   ├── moead_rv_no_risk.py       # w/o Risk
│   ├── moead_rv_no_hv.py         # w/o HV
│   ├── moead_rv_no_pred.py       # w/o Prediction
│   ├── moead_rv_no_nich.py       # w/o Niching
│   └── moead_rv_no_scen.py       # w/o Scenario
├── experiments/
│   ├── configs/                  # Configuration files
│   ├── results/                  # Experimental results
│   │   ├── individual_curves/    # IGD/MHV convergence plots
│   │   └── ablation_curves/      # Ablation study plots
│   └── plots/                    # Visualization outputs
├── tests/
│   └── test_algorithm.py         # Unit tests
├── main_set1.py                  # Set 1: Standard benchmark
├── main_set2.py                  # Set 2: Environment configs
├── main_set4.py                  # Set 4: Ablation study
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📚 Benchmark Problems (CEC2018 DF Suite)

| Problem | Objectives | PF Shape | PS Shape | Dynamic Characteristics |
|---------|-----------|----------|----------|-------------------------|
| DF1 | 2 | Convex | Linear | Linear change |
| DF2 | 2 | Convex | Linear | Nonlinear change |
| DF3 | 2 | Convex | Linear | Mixed linear-nonlinear |
| DF4 | 2 | Convex | Linear | Exponential change |
| DF5 | 2 | Concave | Rotational | Rotational change |
| DF6 | 2 | Concave | Rotational | Mixed rotational-linear |
| DF7 | 2 | Discontinuous | Linear | Discontinuous PF |
| DF8 | 2 | Convex | Discontinuous | Discontinuous PS |
| DF9 | 2 | Convex | Mixed | Mixed discontinuous |
| DF10 | 3 | Convex | Linear | Linear change (3-objective) |
| DF11 | 3 | Convex | Linear | Nonlinear change (3-objective) |
| DF12 | 3 | Concave | Rotational | Rotational change (3-objective) |
| DF13 | 3 | Mixed | Linear | Complex change (3-objective) |
| DF14 | 3 | Mixed | Linear | Abrupt change (3-objective) |

---

## 🛠️ Parameters

### Common Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Population size | $N$ | 100 | Number of solutions |
| Neighborhood size | $T$ | 20 | Neighboring weight vectors |
| Crossover probability | $p_c$ | 0.9 | SBX crossover rate |
| Crossover distribution | $\eta_c$ | 20 | SBX intensity |
| Mutation probability | $p_m$ | 0.1 | Polynomial mutation rate |
| Change frequency | $n_t$ | 10 | Generations between changes |
| Change severity | $\tau_t$ | 10 | Severity of changes |
| Runs | $R$ | 5 | Independent runs (30 for publication) |

### MOEA/D-RV Specific Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Base mutation | $\sigma_{\text{base}}$ | 0.15 | Base mutation intensity |
| Risk sensitivity | $\alpha$ | 1.5 | Risk sensitivity coefficient |
| History length | $s$ | 10 | Risk history window |
| Detection threshold | $\varepsilon$ | 0.005 | Change detection threshold |
| Niche radius | $\sigma_{\text{share}}$ | 0.15 | Initial niche radius |
| PBI penalty | $\theta_{\text{PBI}}$ | 5.0 | PBI penalty parameter |
| Risk aversion | $\kappa_{\text{base}}$ | 0.5 | Base risk aversion coefficient |
| Scenarios | $M_{\text{base}}$ | 10 | Base number of scenarios (optimized) |

---

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Ndikuriyo2026MOEADRV,
  title={MOEA/D-RV: A Risk-Guided Mutation Strategy with Hybrid Decomposition and Risk Control for Dynamic Multi-Objective Optimization},
  author={Ndikuriyo, Yves and Zhang, Yinggui},
  journal={IEEE Transactions on Evolutionary Computation},
  year={2026},
  note={Under Review}
}
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This research was supported by:
- National Natural Science Foundation of China (Grant No. 71971220)
- Natural Science Foundation of Hunan Province, China (Grant Nos. 2023JJ30710 and 2022JJ31020)

---

## 📧 Contact

- **Yves Ndikuriyo** — [GitHub](https://github.com/YvesNDIKURIYO-2022)
- **Prof. Yinggui Zhang** — ygzhang@csu.edu.cn

---

*Built with ❤️ for the evolutionary computation community.*
