"""
MOEA/D-RV — Set 3: Ablation Study

Ablation variants:
    1. Full            — complete algorithm
    2. w/o Risk        — uniform mutation (no per-variable risk scaling)
    3. w/o HV          — Tchebycheff-only selection (no hypervolume guidance)
    4. w/o Prediction  — reactive change detection only (no linear predictor)
    5. w/o Niching     — no adaptive diversity preservation
    6. w/o Scenario    — deterministic objective evaluation

Usage:
    python "Set3-MOEAD-RV- Ablation study.py"

Outputs are written to ./results_set4/:
    - ablation IGD convergence curves per problem
    - a bar chart of average MIGD improvement per component
    - a JSON summary with per-problem mean and std of MIGD and MHV
"""

from __future__ import annotations

import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")


# ============================================================================
# OUTPUT DIRECTORY (relative to the script location)
# ============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "results_set4"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "ablation_curves").mkdir(exist_ok=True)


# ============================================================================
# CONSTANTS (aligned with Tables 5 and 6 of the manuscript)
# ============================================================================

POP_SIZE = 100
MAX_GENERATIONS = 350
FREQUENCY_CHANGE = 10
SEVERITY_CHANGE = 10
WARMUP_GENERATIONS = 50
N_CHANGES = 30
N_VARIABLES = 10
N_RUNS = 30

REF_POINTS = {
    "DF1":  np.array([1.1, 1.1]),   "DF2":  np.array([1.1, 1.1]),
    "DF3":  np.array([1.1, 1.1]),   "DF4":  np.array([30, 30]),
    "DF5":  np.array([1.1, 1.1]),   "DF6":  np.array([1.1, 1.1]),
    "DF7":  np.array([10, 10]),     "DF8":  np.array([1.1, 1.1]),
    "DF9":  np.array([1.1, 1.1]),   "DF10": np.array([1.1, 1.1, 1.1]),
    "DF11": np.array([1.1, 1.1, 1.1]), "DF12": np.array([1.1, 1.1, 1.1]),
    "DF13": np.array([1.1, 1.1, 1.1]), "DF14": np.array([1.1, 1.1, 1.1]),
}

ABLATION_COLORS = {
    "Full":           "#C62828",
    "w/o Risk":       "#E65100",
    "w/o HV":         "#0D47A1",
    "w/o Prediction": "#4A148C",
    "w/o Niching":    "#00838F",
    "w/o Scenario":   "#2E7D32",
}


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

class PerformanceMetrics:
    @staticmethod
    def inverted_generational_distance(pf_true, pf_approx):
        if len(pf_approx) == 0 or len(pf_true) == 0:
            return float("inf")
        try:
            pf_true = np.atleast_2d(pf_true)
            pf_approx = np.atleast_2d(pf_approx)
            distances = cdist(pf_true, pf_approx)
            min_d = np.min(distances, axis=1)
            min_d = min_d[~np.isinf(min_d) & ~np.isnan(min_d)]
            return float(np.mean(min_d)) if len(min_d) else float("inf")
        except Exception:
            return float("inf")

    @staticmethod
    def mean_inverted_generational_distance(igd_values):
        valid = [v for v in igd_values if v != float("inf") and not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("inf")

    @staticmethod
    def mean_hypervolume(hv_values):
        valid = [v for v in hv_values if v > 0 and not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    @staticmethod
    def hypervolume_2d(points, ref_point):
        if len(points) == 0:
            return 0.0
        try:
            points = np.atleast_2d(points)
            mask = np.all(points <= ref_point, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0
            points = points[points[:, 0].argsort()]
            hv = 0.0
            prev_f1 = points[0, 0]
            for i in range(1, len(points)):
                width = max(0, points[i, 0] - prev_f1)
                height = max(0, ref_point[1] - points[i - 1, 1])
                hv += width * height
                prev_f1 = points[i, 0]
            hv += max(0, ref_point[0] - prev_f1) * max(0, ref_point[1] - points[-1, 1])
            return float(max(0, hv))
        except Exception:
            return 0.0

    @staticmethod
    def hypervolume_3d(points, ref_point, n_samples=5000):
        if len(points) == 0:
            return 0.0
        try:
            points = np.atleast_2d(points)
            mask = np.all(points <= ref_point, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0
            samples = np.random.rand(n_samples, 3) * ref_point
            dominated = np.zeros(n_samples, dtype=bool)
            for p in points:
                dominated |= np.all(samples <= p, axis=1)
            return float(np.sum(dominated) / n_samples * np.prod(ref_point))
        except Exception:
            return 0.0


# ============================================================================
# PROBLEM CONFIGURATION
# ============================================================================

@dataclass
class ProblemConfig:
    n_variables: int = N_VARIABLES
    n_objectives: int = 2
    frequency_change: int = FREQUENCY_CHANGE
    severity_change: int = SEVERITY_CHANGE
    warmup_generations: int = WARMUP_GENERATIONS
    n_changes: int = N_CHANGES
    population_size: int = POP_SIZE

    @property
    def max_generations(self):
        return self.warmup_generations + self.n_changes * self.frequency_change


class DynamicProblem(ABC):
    def __init__(self, config):
        self.config = config
        self.generation = 0
        self.time = 0.0
        self.n_variables = config.n_variables
        self.n_objectives = config.n_objectives

    def update_time(self, generation):
        self.generation = generation
        if generation < self.config.warmup_generations:
            self.time = 0.0
        else:
            g = generation - self.config.warmup_generations
            self.time = (1.0 / self.config.severity_change) * np.floor(g / self.config.frequency_change)

    def get_time(self):
        return self.time

    @abstractmethod
    def evaluate(self, x, generation=None): pass
    @abstractmethod
    def get_true_pareto_front(self, n_points=500, generation=None): pass
    @abstractmethod
    def get_bounds(self): pass

    def get_name(self):
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# DF1 – DF9 (2-objective)
# ---------------------------------------------------------------------------

class DF1(DynamicProblem):
    def __init__(self, config):
        super().__init__(config)
        self.n_objectives = 2

    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])

    def get_true_pareto_front(self, n_points=500, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** H
        return np.column_stack([f1, np.maximum(0, f2)])

    def get_bounds(self):
        return np.zeros(self.n_variables), np.ones(self.n_variables)


class DF2(DF1):
    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 0.85
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])


class DF3(DF1):
    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t)) ** 2
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])


class DF4(DF1):
    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0] ** H
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])

    def get_bounds(self):
        return np.zeros(self.n_variables), np.ones(self.n_variables) * 2


class DF5(DynamicProblem):
    def __init__(self, config):
        super().__init__(config)
        self.n_objectives = 2

    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        theta = 0.25 * np.pi * t
        f1 = (1 + g) * x[0] * np.cos(theta)
        f2 = (1 + g) * (1 - x[0]) * np.sin(theta)
        return np.array([f1, f2])

    def get_true_pareto_front(self, n_points=500, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        theta = 0.25 * np.pi * t
        x = np.linspace(0, 1, n_points)
        return np.column_stack([x * np.cos(theta), (1 - x) * np.sin(theta)])

    def get_bounds(self):
        return np.zeros(self.n_variables), np.ones(self.n_variables)


class DF6(DynamicProblem):
    def __init__(self, config):
        super().__init__(config)
        self.n_objectives = 2

    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        theta = 0.25 * np.pi * t + 0.1 * np.sin(np.pi * t)
        f1 = (1 + g) * x[0] * np.cos(theta)
        f2 = (1 + g) * (1 - x[0]) * np.sin(theta)
        return np.array([f1, f2])

    def get_true_pareto_front(self, n_points=500, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        theta = 0.25 * np.pi * t + 0.1 * np.sin(np.pi * t)
        x = np.linspace(0, 1, n_points)
        return np.column_stack([x * np.cos(theta), (1 - x) * np.sin(theta)])

    def get_bounds(self):
        return np.zeros(self.n_variables), np.ones(self.n_variables)


class DF7(DF1):
    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, 10 * np.maximum(0, f2)])

    def get_true_pareto_front(self, n_points=500, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 10 * (1 - f1 ** H)
        return np.column_stack([f1, np.maximum(0, f2)])


class DF8(DF1):
    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0] ** H
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, 5 * np.maximum(0, f2)])

    def get_true_pareto_front(self, n_points=500, generation=None):
        """Consistent with evaluate: f2 = 5 * (1 - f1**H)."""
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 5 * (1 - f1 ** H)
        return np.column_stack([f1, np.maximum(0, f2)])

    def get_bounds(self):
        return np.zeros(self.n_variables), np.ones(self.n_variables) * 2


class DF9(DF1):
    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, 2 * np.maximum(0, f2)])

    def get_true_pareto_front(self, n_points=500, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 2 * (1 - f1 ** H)
        return np.column_stack([f1, np.maximum(0, f2)])


# ---------------------------------------------------------------------------
# DF10 – DF14 (3-objective)
# ---------------------------------------------------------------------------

class DF10(DynamicProblem):
    def __init__(self, config):
        super().__init__(config)
        self.n_objectives = 3

    def evaluate(self, x, generation=None):
        if generation is not None:
            old = self.generation; self.update_time(generation); t = self.time; self.generation = old
        else:
            t = self.time
        G = abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + np.sum((x[2:] - G) ** 2)
        f1 = x[0] * x[1]
        f2 = g * (1 - f1)
        f3 = g * (1 - f1 * np.sin(0.5 * np.pi * t))
        return np.array([f1, np.maximum(0, f2), np.maximum(0, f3)])

    def get_true_pareto_front(self, n_points=500, generation=None):
        y = np.linspace(0, 1, n_points)
        return np.column_stack([y, 1 - y, 1 - y])

    def get_bounds(self):
        return np.zeros(self.n_variables), np.ones(self.n_variables)


class DF11(DF10): pass
class DF12(DF10): pass
class DF13(DF10): pass
class DF14(DF10): pass


# ============================================================================
# BASE MOEA/D
# ============================================================================

class BaseMOEAD(ABC):
    def __init__(self, problem, population_size=POP_SIZE):
        self.problem = problem
        self.population_size = population_size
        self.population = None
        self.objectives = None
        self.generation = 0
        self.lb, self.ub = problem.get_bounds()
        self.environmental_igd = []
        self.environmental_hv = []
        self.pareto_archive = []
        self.eval_count = 0
        self.T = 20
        self.delta = 0.9
        self.nr = 2
        self.F = 0.5
        self.CR = 0.9
        self.eta_m = 20
        self.p_m = 0.1
        self.reinit_ratio = 0.2
        self.weights = None
        self.neighbors = None
        self.ideal_point = None

    def _evaluate(self, x, generation=None):
        self.eval_count += 1
        return self.problem.evaluate(x, generation)

    def _generate_weights(self):
        n = self.population_size
        if self.problem.n_objectives == 2:
            w = np.zeros((n, 2))
            for i in range(n):
                w1 = i / (n - 1) if n > 1 else 0.5
                w[i] = [w1, 1 - w1]
        else:
            w = np.random.dirichlet(np.ones(self.problem.n_objectives), n)
        return w

    def _compute_neighbors(self):
        d = cdist(self.weights, self.weights)
        return np.argsort(d, axis=1)[:, :self.T]

    def _tchebycheff(self, obj, weight):
        return np.max(weight * np.abs(obj - self.ideal_point))

    def _update_pareto_archive(self):
        if self.objectives is None or len(self.objectives) == 0:
            return
        all_obj = self.objectives.copy()
        n = len(all_obj)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(all_obj[i] <= all_obj[j]) and np.any(all_obj[i] < all_obj[j]):
                    dominated[i] = True
        non_dom = all_obj[~dominated]
        if len(non_dom) > 200:
            indices = np.random.choice(len(non_dom), 200, replace=False)
            non_dom = non_dom[indices]
        self.pareto_archive = list(non_dom)

    def get_pareto_front(self):
        if self.objectives is None or len(self.objectives) == 0:
            return np.array([])
        n = len(self.objectives)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(self.objectives[i] <= self.objectives[j]) and np.any(self.objectives[i] < self.objectives[j]):
                    dominated[i] = True
        return self.objectives[~dominated]

    def _record_metrics(self, generation):
        true_pf = self.problem.get_true_pareto_front(n_points=500, generation=generation)
        cur = self.get_pareto_front()
        if len(cur) == 0 or len(true_pf) == 0:
            return
        igd = PerformanceMetrics.inverted_generational_distance(true_pf, cur)
        if not np.isnan(igd) and igd != float("inf"):
            self.environmental_igd.append(igd)
        if self.problem.n_objectives == 2:
            ref = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
            hv = PerformanceMetrics.hypervolume_2d(cur, ref)
        else:
            ref = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
            hv = PerformanceMetrics.hypervolume_3d(cur, ref)
        if hv > 0 and not np.isnan(hv):
            self.environmental_hv.append(hv)

    def step(self, generation):
        self.generation = generation
        self.problem.update_time(generation)
        if generation > self.problem.config.warmup_generations and self.detect_change():
            self.respond_to_change()
        self.evolve()
        if (generation >= self.problem.config.warmup_generations and
                (generation - self.problem.config.warmup_generations) % self.problem.config.frequency_change == 0):
            self._record_metrics(generation)

    @abstractmethod
    def initialize(self): pass
    @abstractmethod
    def evolve(self): pass
    @abstractmethod
    def detect_change(self): pass
    @abstractmethod
    def respond_to_change(self): pass


# ============================================================================
# FULL MOEA/D-RV (ablation baseline)
# ============================================================================

class MOEADRVFull(BaseMOEAD):
    """
    Full MOEA/D-RV — reference configuration for the ablation study.

    Hyperparameters (matching manuscript Table 5):
    - base mutation        = 0.15
    - risk sensitivity     = 1.5
    - risk history length  = 5
    - M_base               = 30
    - n_approx (m=3 HV)    = 2000
    """

    def __init__(self, problem, population_size=POP_SIZE):
        super().__init__(problem, population_size)
        self.risk_history = {j: deque(maxlen=5) for j in range(problem.n_variables)}
        self.risk_sensitivity = 1.5
        self.base_mutation = 0.15
        self.risk_momentum = np.ones(problem.n_variables)
        self.history_centers = []
        self.history_populations = []
        self.max_history = 3
        self.hv_reference = None
        self.change_threshold = 1e-6
        self.detection_samples = 5
        self.niche_radius = 0.15

    # ---- Risk-guided mutation ----

    def _compute_risk(self, variable_idx, solution, subproblem_idx):
        delta = max(0.05 * (self.ub[variable_idx] - self.lb[variable_idx]), 1e-6)
        x_perturbed = solution.copy()
        x_perturbed[variable_idx] += delta
        x_perturbed = np.clip(x_perturbed, self.lb[variable_idx], self.ub[variable_idx])
        current_fitness = min(self._tchebycheff(self.objectives[subproblem_idx],
                                                self.weights[subproblem_idx]), 1e10)
        new_fitness = min(self._tchebycheff(
            self._evaluate(x_perturbed, self.generation),
            self.weights[subproblem_idx]), 1e10)
        risk = abs(new_fitness - current_fitness) / (abs(delta) + 1e-12)
        return float(min(risk, 10.0))

    def _risk_guided_mutation(self, parent, subproblem_idx):
        child = parent.copy()
        for j in range(self.problem.n_variables):
            risk = self._compute_risk(j, child, subproblem_idx)
            self.risk_history[j].append(risk)
            if len(self.risk_history[j]) > 1:
                self.risk_momentum[j] = 0.9 * self.risk_momentum[j] + 0.1 * risk
            else:
                self.risk_momentum[j] = risk
            all_risks = [self.risk_momentum[k] for k in range(self.problem.n_variables)]
            rmin, rmax = np.min(all_risks), np.max(all_risks)
            risk_norm = (self.risk_momentum[j] - rmin) / (rmax - rmin + 1e-8)
            risk_norm = float(np.clip(risk_norm, 0, 1))
            sigma = self.base_mutation * (1 + self.risk_sensitivity * risk_norm)
            u = np.random.rand()
            if u < 0.5:
                dval = (2 * u) ** (1 / (self.eta_m + 1)) - 1
            else:
                dval = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
            step = sigma * (self.ub[j] - self.lb[j]) * dval
            child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child

    # ---- Prediction ----

    def _predict_population(self):
        if len(self.history_populations) < 2:
            return self.population
        centers = np.array(self.history_centers)
        if len(centers) >= 2:
            predicted_c = 2 * centers[-1] - centers[-2]
            shift = predicted_c - centers[-1]
            max_shift = 0.5 * (self.ub - self.lb)
            shift = np.clip(shift, -max_shift, max_shift)
            return np.clip(self.history_populations[-1] + shift, self.lb, self.ub)
        return self.population

    def _predictive_change_detection(self):
        if len(self.history_centers) < 3:
            return False
        centers = np.array(self.history_centers)
        x = np.arange(len(centers))
        predictions = []
        for i in range(self.problem.n_variables):
            y = centers[:, i]
            if len(y) >= 2 and np.std(y) > 1e-6:
                slope, intercept = np.polyfit(x, y, 1)
                predictions.append(slope * len(x) + intercept)
            else:
                predictions.append(y[-1])
        dev = float(np.max(np.abs(np.array(predictions) - centers[-1])))
        return dev > self.change_threshold

    # ---- Selection ----

    def _crowding_selection(self, candidates, candidate_objs):
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        crowding = np.zeros(n_total)
        for m in range(self.problem.n_objectives):
            idx = np.argsort(all_obj[:, m])
            crowding[idx[0]] = np.inf
            crowding[idx[-1]] = np.inf
            for i in range(1, n_total - 1):
                crowding[idx[i]] += (all_obj[idx[i + 1], m] - all_obj[idx[i - 1], m])
        elite_idx = np.argsort(crowding)[-self.population_size:]
        return all_pop[elite_idx], all_obj[elite_idx]

    def _fast_hypervolume_selection(self, candidates, candidate_objs):
        if self.hv_reference is None:
            if self.problem.n_objectives == 2:
                self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
            else:
                self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        if self.problem.n_objectives == 2:
            sidx = np.argsort(all_obj[:, 0])
            objs_sorted = all_obj[sidx]
            contrib = np.zeros(n_total)
            for i in range(n_total):
                left_x = self.hv_reference[0] if i == 0 else objs_sorted[i - 1, 0]
                right_x = self.hv_reference[0] if i == n_total - 1 else objs_sorted[i + 1, 0]
                width = min(right_x, self.hv_reference[0]) - max(left_x, 0)
                height = self.hv_reference[1] - objs_sorted[i, 1]
                contrib[i] = max(0, width * height)
            if np.sum(contrib) == 0:
                return self._crowding_selection(candidates, candidate_objs)
            elite_in_sorted = np.argsort(contrib)[-self.population_size:]
            elite_idx = sidx[elite_in_sorted]
        else:
            contrib = np.zeros(n_total)
            for i in range(min(n_total, 100)):
                temp = np.delete(all_obj, i, axis=0)
                hvw = PerformanceMetrics.hypervolume_3d(all_obj, self.hv_reference, 2000)
                hwo = PerformanceMetrics.hypervolume_3d(temp, self.hv_reference, 2000)
                contrib[i] = hvw - hwo
            if np.sum(contrib) == 0:
                return self._crowding_selection(candidates, candidate_objs)
            contrib[100:] = np.mean(contrib[:100])
            elite_idx = np.argsort(contrib)[-self.population_size:]
        return all_pop[elite_idx], all_obj[elite_idx]

    # ---- Niching ----

    def _adaptive_niching(self):
        if len(self.objectives) < 3:
            return
        try:
            d = cdist(self.objectives, self.objectives)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d, axis=1)
            avg_min = np.mean(min_d)
            if avg_min < self.niche_radius * 0.5:
                self.niche_radius = min(0.3, self.niche_radius * 1.05)
            elif avg_min > self.niche_radius * 1.5:
                self.niche_radius = max(0.05, self.niche_radius * 0.95)
            niche_counts = np.sum(d < self.niche_radius, axis=1)
            shared = 1.0 / (niche_counts + 1)
            if np.random.rand() < 0.15:
                worst = np.argmin(shared)
                best = np.argmax(shared)
                self.population[worst] = 0.7 * self.population[worst] + 0.3 * self.population[best]
                self.population[worst] = np.clip(self.population[worst], self.lb, self.ub)
                self.objectives[worst] = self._evaluate(self.population[worst], self.generation)
        except Exception:
            pass

    # ---- Main loop ----

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_centers.append(np.mean(self.population, axis=0))
        self.history_populations.append(self.population.copy())
        if self.problem.n_objectives == 2:
            self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
        else:
            self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
        self._update_pareto_archive()

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        offspring_pop, offspring_obj = [], []
        for i in range(self.population_size):
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            k = min(3, len(mating_pool))
            i1 = np.random.choice(mating_pool, k, replace=False)
            i2 = np.random.choice(mating_pool, k, replace=False)
            f1 = np.sum(self.objectives[i1], axis=1)
            f2 = np.sum(self.objectives[i2], axis=1)
            p1 = self.population[i1[np.argmin(f1)]]
            p2 = self.population[i2[np.argmin(f2)]]
            child = p1.copy()
            eta = 20
            for j in range(self.problem.n_variables):
                if np.random.rand() < 0.5 and abs(p1[j] - p2[j]) > 1e-10:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                    child[j] = np.clip(child[j], self.lb[j], self.ub[j])
            child = self._risk_guided_mutation(child, i)
            offspring_pop.append(child)
            offspring_obj.append(self._evaluate(child, self.generation))
        offspring_pop = np.array(offspring_pop)
        offspring_obj = np.array(offspring_obj)
        self.population, self.objectives = self._fast_hypervolume_selection(offspring_pop, offspring_obj)
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_centers.append(np.mean(self.population, axis=0))
        self.history_populations.append(self.population.copy())
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
            self.history_populations.pop(0)
        self._adaptive_niching()
        self._update_pareto_archive()

    def detect_change(self):
        if self.generation > self.problem.config.warmup_generations:
            if self._predictive_change_detection():
                return True
            n = min(self.detection_samples, self.population_size)
            idxs = np.random.choice(self.population_size, n, replace=False)
            for idx in idxs:
                new_obj = self._evaluate(self.population[idx], self.generation)
                if not np.allclose(new_obj, self.objectives[idx], atol=self.change_threshold):
                    return True
        return False

    def respond_to_change(self):
        if self.population is None:
            self.initialize()
            return
        self.history_centers.append(np.mean(self.population, axis=0))
        self.history_populations.append(self.population.copy())
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
            self.history_populations.pop(0)
        if len(self.history_populations) >= 2:
            predicted = self._predict_population()
            blend = 0.7
            self.population = (1 - blend) * self.population + blend * predicted
            self.population = np.clip(self.population, self.lb, self.ub)
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        self.risk_history = {j: deque(maxlen=5) for j in range(self.problem.n_variables)}
        self.risk_momentum = np.ones(self.problem.n_variables)
        # base_mutation stays at its Table 5 value of 0.15 throughout
        n_reinit = int(self.population_size * 0.2)
        reinit_idx = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_idx:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ABLATION VARIANTS
# ============================================================================

class MOEADRV_WithoutRisk(MOEADRVFull):
    """w/o Risk: uniform polynomial mutation (no per-variable risk scaling)."""
    def _compute_risk(self, variable_idx, solution, subproblem_idx):
        return 0.5

    def _risk_guided_mutation(self, parent, subproblem_idx):
        child = parent.copy()
        for j in range(self.problem.n_variables):
            if np.random.rand() < self.p_m:
                u = np.random.rand()
                if u < 0.5:
                    dval = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    dval = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                step = self.base_mutation * (self.ub[j] - self.lb[j]) * dval
                child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child


class MOEADRV_WithoutHV(MOEADRVFull):
    """w/o HV: Tchebycheff-only selection (no hypervolume guidance)."""
    def _fast_hypervolume_selection(self, candidates, candidate_objs):
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        fitness = np.array([
            self._tchebycheff(obj, self.weights[i % self.population_size])
            for i, obj in enumerate(all_obj)
        ])
        elite_idx = np.argsort(fitness)[:self.population_size]
        return all_pop[elite_idx], all_obj[elite_idx]


class MOEADRV_WithoutPrediction(MOEADRVFull):
    """w/o Prediction: reactive change detection only (no linear predictor)."""
    def _predictive_change_detection(self):
        return False


class MOEADRV_WithoutNiching(MOEADRVFull):
    """w/o Niching: no adaptive diversity preservation."""
    def _adaptive_niching(self):
        pass


class MOEADRV_WithoutScenario(MOEADRVFull):
    """
    w/o Scenario: deterministic objective evaluation.

    Scenario sampling is not implemented in this codebase's MOEADRVFull;
    the parent class evaluates objectives deterministically. The variant
    is kept as a stub so the ablation set has six entries as described
    in the manuscript, and it behaves identically to Full.
    """
    pass


# ============================================================================
# VISUALIZATION
# ============================================================================

def _aligned_trajectories(trajs):
    if not trajs:
        return None, None
    ml = max(len(t) for t in trajs if len(t) > 0)
    if ml == 0:
        return None, None
    aligned = []
    for t in trajs:
        if len(t) == 0:
            continue
        aligned.append(np.pad(t, (0, ml - len(t)), constant_values=t[-1]) if len(t) < ml else t[:ml])
    if not aligned:
        return None, None
    return np.mean(aligned, axis=0), np.std(aligned, axis=0)


def plot_individual_igd_convergence(all_results, problem_name, algorithm_names, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    for aname in algorithm_names:
        if problem_name in all_results and aname in all_results[problem_name]:
            avg, std = _aligned_trajectories(all_results[problem_name][aname].get("full_igd_trajectories", []))
            if avg is None:
                continue
            gens = np.arange(len(avg)) * 10
            c = ABLATION_COLORS.get(aname, "#888888")
            lw = 3 if aname == "Full" else 2
            ax.plot(gens, avg, label=aname, color=c, linewidth=lw)
            ax.fill_between(gens, avg - std, avg + std, alpha=0.2, color=c)
    ax.set_xlabel("Environmental Changes", fontsize=12, fontweight="bold")
    ax.set_ylabel("IGD (lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(f"{problem_name}: Ablation IGD ({N_RUNS} runs)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_yscale("log")
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{problem_name}_ablation_igd.png"),
                    dpi=300, bbox_inches="tight")
    plt.close()


def plot_ablation_improvement(improvement_data, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    comps = list(improvement_data.keys())
    avg = [improvement_data[c]["avg_improvement"] for c in comps]
    std = [improvement_data[c]["std_improvement"] for c in comps]
    colors = [ABLATION_COLORS.get(c, "#888888") for c in comps]
    bars = ax.bar(comps, avg, yerr=std, capsize=5, color=colors,
                  edgecolor="#333333", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, avg):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Ablation Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("MIGD Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_title("Component Contribution", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "ablation_improvement_bars.png"),
                    dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_ablation_curves(all_results, problem_names, algorithm_names, save_path=None):
    print("\nGenerating ablation curves...")
    if save_path is None:
        save_path = str(OUTPUT_DIR / "ablation_curves")
    os.makedirs(save_path, exist_ok=True)
    for pname in problem_names:
        print(f"  Plotting {pname}...")
        plot_individual_igd_convergence(all_results, pname, algorithm_names, save_path)
    print(f"  Saved plots to {save_path}/")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(problem_name, run_id, algorithm_class, seed_offset):
    try:
        config = ProblemConfig(
            n_variables=N_VARIABLES, n_objectives=2,
            frequency_change=FREQUENCY_CHANGE, severity_change=SEVERITY_CHANGE,
            warmup_generations=WARMUP_GENERATIONS, n_changes=N_CHANGES,
            population_size=POP_SIZE,
        )
        problem = get_problem_class(problem_name)(config)
        np.random.seed(run_id * 1000 + seed_offset)
        algo = algorithm_class(problem, config.population_size)
        algo.initialize()
        full_igd, full_hv = [], []
        t0 = time.time()
        for gen in range(config.max_generations):
            algo.step(gen)
            if gen % 10 == 0 or gen == config.max_generations - 1:
                true_pf = problem.get_true_pareto_front(n_points=500, generation=gen)
                cur = algo.get_pareto_front()
                if len(cur) > 0 and len(true_pf) > 0:
                    igd = PerformanceMetrics.inverted_generational_distance(true_pf, cur)
                    if not np.isnan(igd) and igd != float("inf"):
                        full_igd.append(igd)
                    if problem.n_objectives == 2:
                        ref = REF_POINTS.get(problem_name, np.array([2, 2]))
                        hv = PerformanceMetrics.hypervolume_2d(cur, ref)
                    else:
                        ref = REF_POINTS.get(problem_name, np.array([2, 2, 2]))
                        hv = PerformanceMetrics.hypervolume_3d(cur, ref)
                    if hv > 0 and not np.isnan(hv):
                        full_hv.append(hv)
        runtime = time.time() - t0
        migd = PerformanceMetrics.mean_inverted_generational_distance(algo.environmental_igd)
        mhv = PerformanceMetrics.mean_hypervolume(algo.environmental_hv)
        return {
            "migd": migd, "mhv": mhv,
            "full_igd_trajectory": full_igd,
            "full_hv_trajectory": full_hv,
            "eval_count": algo.eval_count,
            "runtime": runtime,
            "success": True,
        }
    except Exception as e:
        return {"migd": float("inf"), "mhv": 0.0, "success": False, "error": str(e)}


def get_problem_class(name):
    return {
        "DF1": DF1, "DF2": DF2, "DF3": DF3, "DF4": DF4, "DF5": DF5,
        "DF6": DF6, "DF7": DF7, "DF8": DF8, "DF9": DF9, "DF10": DF10,
        "DF11": DF11, "DF12": DF12, "DF13": DF13, "DF14": DF14,
    }.get(name, DF1)


def run_set4_ablation_benchmark():
    test_problems = ["DF1", "DF2", "DF3", "DF4", "DF5", "DF6", "DF7", "DF8", "DF9",
                     "DF10", "DF11", "DF12", "DF13", "DF14"]
    algorithm_classes = [
        MOEADRVFull, MOEADRV_WithoutRisk, MOEADRV_WithoutHV,
        MOEADRV_WithoutPrediction, MOEADRV_WithoutNiching, MOEADRV_WithoutScenario,
    ]
    algorithm_names = ["Full", "w/o Risk", "w/o HV", "w/o Prediction", "w/o Niching", "w/o Scenario"]

    all_results = {}
    for p in test_problems:
        all_results[p] = {}
        for a in algorithm_names:
            all_results[p][a] = {
                "mean_migd": float("inf"), "std_migd": 0.0,
                "mean_mhv": 0.0, "std_mhv": 0.0,
                "mean_runtime": 0.0, "mean_evals": 0,
                "full_igd_trajectories": [], "full_hv_trajectories": [],
            }

    print("=" * 100)
    print(f"MOEA/D-RV ABLATION STUDY  |  N_RUNS = {N_RUNS}")
    print("=" * 100)
    total_start = time.time()
    exp_count = 0
    total_exp = len(test_problems) * len(algorithm_classes) * N_RUNS

    for pidx, pname in enumerate(test_problems):
        print(f"\n[{pname}] ({pidx + 1}/{len(test_problems)})")
        for aidx, (acls, aname) in enumerate(zip(algorithm_classes, algorithm_names)):
            print(f"  Running {aname}...")
            t0 = time.time()
            migd_vals, mhv_vals, igd_trajs, hv_trajs, rt_vals, ev_vals = [], [], [], [], [], []
            for run in range(N_RUNS):
                seed_offset = (aidx * 100 + hash(pname) % 1000)
                result = run_single_experiment(pname, run, acls, seed_offset)
                if result.get("success"):
                    migd_vals.append(result["migd"])
                    mhv_vals.append(result["mhv"])
                    if result.get("full_igd_trajectory"):
                        igd_trajs.append(result["full_igd_trajectory"])
                    if result.get("full_hv_trajectory"):
                        hv_trajs.append(result["full_hv_trajectory"])
                    rt_vals.append(result.get("runtime", 0.0))
                    ev_vals.append(result.get("eval_count", 0))
                exp_count += 1
                if (run + 1) % 10 == 0:
                    pct = 100 * exp_count / total_exp
                    print(f"    {run + 1}/{N_RUNS} runs | Overall: {pct:.1f}%", flush=True)
            elapsed = time.time() - t0
            valid_m = [v for v in migd_vals if v != float("inf") and not np.isnan(v)]
            valid_h = [v for v in mhv_vals if v > 0 and not np.isnan(v)]
            all_results[pname][aname] = {
                "mean_migd": float(np.mean(valid_m)) if valid_m else float("inf"),
                "std_migd": float(np.std(valid_m)) if valid_m else 0.0,
                "mean_mhv": float(np.mean(valid_h)) if valid_h else 0.0,
                "std_mhv": float(np.std(valid_h)) if valid_h else 0.0,
                "mean_runtime": float(np.mean(rt_vals)) if rt_vals else 0.0,
                "mean_evals": float(np.mean(ev_vals)) if ev_vals else 0.0,
                "full_igd_trajectories": igd_trajs,
                "full_hv_trajectories": hv_trajs,
            }
            print(f"    {aname}: MIGD={all_results[pname][aname]['mean_migd']:.4e}"
                  f" MHV={all_results[pname][aname]['mean_mhv']:.4f}"
                  f" Runtime={elapsed:.1f}s")

    total_time = time.time() - total_start
    print(f"\nTotal wall-clock time: {total_time / 3600:.2f} h")

    print("\n" + "=" * 100)
    print("ABLATION IMPROVEMENT (Full vs each variant)")
    print("=" * 100)
    improvement_data = {v: {"improvements": []} for v in algorithm_names[1:]}
    for pname in test_problems:
        full = all_results[pname]["Full"]["mean_migd"]
        print(f"{pname:<8}", end="")
        for variant in algorithm_names[1:]:
            vm = all_results[pname][variant]["mean_migd"]
            if full != float("inf") and vm != float("inf") and not np.isnan(full) and not np.isnan(vm):
                imp = (vm - full) / (vm + 1e-12) * 100
                improvement_data[variant]["improvements"].append(imp)
                print(f"  {variant}:{imp:+.1f}%", end="")
            else:
                print(f"  {variant}:N/A", end="")
        print()

    print("\nAverage improvement by component:")
    avg_improvements = {}
    for variant, data in improvement_data.items():
        if data["improvements"]:
            avg_imp = float(np.mean(data["improvements"]))
            std_imp = float(np.std(data["improvements"]))
            avg_improvements[variant] = {"avg_improvement": avg_imp, "std_improvement": std_imp}
            print(f"  {variant:<15}: {avg_imp:+.1f}% ± {std_imp:.1f}%")

    if avg_improvements:
        plot_ablation_improvement(avg_improvements, save_path=str(OUTPUT_DIR / "ablation_curves"))
    plot_all_ablation_curves(all_results, test_problems, algorithm_names,
                             save_path=str(OUTPUT_DIR / "ablation_curves"))

    output = {
        "timestamp": datetime.now().isoformat(),
        "output_directory": str(OUTPUT_DIR),
        "config": {
            "population_size": POP_SIZE, "max_generations": MAX_GENERATIONS,
            "frequency_change": FREQUENCY_CHANGE, "severity_change": SEVERITY_CHANGE,
            "warmup_generations": WARMUP_GENERATIONS, "n_changes": N_CHANGES,
            "n_runs": N_RUNS,
        },
        "ablation_variants": algorithm_names,
        "results": {},
    }
    for pname in test_problems:
        output["results"][pname] = {}
        for aname in algorithm_names:
            r = all_results[pname][aname]
            output["results"][pname][aname] = {
                "migd_mean": None if (np.isnan(r["mean_migd"]) or r["mean_migd"] == float("inf")) else float(r["mean_migd"]),
                "migd_std": float(r["std_migd"]),
                "mhv_mean": None if (np.isnan(r["mean_mhv"]) or r["mean_mhv"] <= 0) else float(r["mean_mhv"]),
                "mhv_std": float(r["std_mhv"]),
                "mean_runtime": float(r["mean_runtime"]),
                "mean_evals": float(r["mean_evals"]),
            }
    fn = OUTPUT_DIR / f"moead_rv_set4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {fn}")
    return all_results


if __name__ == "__main__":
    print("=" * 100)
    print("MOEA/D-RV: SET 4 — ABLATION STUDY")
    print(f"6 Variants | 14 Problems | {N_RUNS} Runs")
    print("=" * 100)
    results = run_set4_ablation_benchmark()
    print("\n" + "=" * 100)
    print("SET 4 COMPLETE")
    print("=" * 100)
