"""
MOEA/D-RV — Set 1: Standard Benchmark

Dynamic Multi-Objective Optimization Algorithm

Core features:
1. Risk-Guided Mutation — variable-specific adaptive mutation (magnitude only)
2. Prediction (PPS-style) — population prediction for fast tracking
3. Hypervolume-Guided Selection — elite selection by hypervolume contribution

Compared algorithms (6 MOEA/D variants):
1. MOEA/D      — baseline decomposition-based
2. MOEA/D-KNN  — training-free local prediction
3. MOEA/D-PPS  — population prediction strategy
4. MOEA/D-AGR  — adaptive guided response
5. MOEA/D-HSS  — hybrid search strategy
6. MOEA/D-RV   — proposed: risk-guided mutation + prediction + hypervolume selection

Usage:
    python "Set1-MOEAD-RV Vs Benchmarks.py"

Outputs are written to ``./results_set1/`` in the current working directory:
    - individual convergence curves (IGD, MHV) for each DF problem
    - a JSON file with per-problem mean and standard deviation of MIGD and MHV
"""

from __future__ import annotations

import json
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
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================================
# OUTPUT DIRECTORY (relative to the script location)
# ============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "results_set1"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "individual_curves").mkdir(exist_ok=True)


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

COLORS = {
    "MOEA/D":     "#1B5E20",
    "MOEA/D-KNN": "#0D47A1",
    "MOEA/D-PPS": "#E65100",
    "MOEA/D-AGR": "#4A148C",
    "MOEA/D-HSS": "#00838F",
    "MOEA/D-RV":  "#C62828",
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
            min_distances = np.min(distances, axis=1)
            min_distances = min_distances[~np.isinf(min_distances) & ~np.isnan(min_distances)]
            if len(min_distances) == 0:
                return float("inf")
            return float(np.mean(min_distances))
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
            width = max(0, ref_point[0] - prev_f1)
            height = max(0, ref_point[1] - points[-1, 1])
            hv += width * height
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
            gen_since_warmup = generation - self.config.warmup_generations
            self.time = (1.0 / self.config.severity_change) * np.floor(
                gen_since_warmup / self.config.frequency_change
            )

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
            old = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old
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
            old = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old
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
    """DF5: concave PF, rotational change."""
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
    """DF6: concave PF, mixed rotational-linear PS."""
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
# BASE MOEA/D CLASS
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
            weights = np.zeros((n, 2))
            for i in range(n):
                w1 = i / (n - 1) if n > 1 else 0.5
                weights[i] = [w1, 1 - w1]
        else:
            weights = np.random.dirichlet(np.ones(self.problem.n_objectives), n)
        return weights

    def _compute_neighbors(self):
        dist = cdist(self.weights, self.weights)
        return np.argsort(dist, axis=1)[:, :self.T]

    def _tchebycheff(self, obj, weight):
        return np.max(weight * np.abs(obj - self.ideal_point))

    def _de_operator(self, idx, mating_pool):
        if len(mating_pool) < 3:
            candidates = np.random.choice(self.population_size, 3, replace=False)
        else:
            candidates = np.random.choice(mating_pool, 3, replace=False)
        r1, r2, r3 = candidates
        child = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        parent = self.population[idx]
        for j in range(len(child)):
            if np.random.rand() > self.CR:
                child[j] = parent[j]
        for j in range(len(child)):
            if np.random.rand() < self.p_m:
                delta = np.random.rand()
                if delta < 0.5:
                    child[j] += (self.ub[j] - self.lb[j]) * ((2 * delta) ** (1 / (self.eta_m + 1)) - 1)
                else:
                    child[j] += (self.ub[j] - self.lb[j]) * (1 - (2 * (1 - delta)) ** (1 / (self.eta_m + 1)))
                child[j] = np.clip(child[j], self.lb[j], self.ub[j])
        return child

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
        non_dominated = all_obj[~dominated]
        if len(non_dominated) > 200:
            indices = np.random.choice(len(non_dominated), 200, replace=False)
            non_dominated = non_dominated[indices]
        self.pareto_archive = list(non_dominated)

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

    def step(self, generation):
        self.generation = generation
        self.problem.update_time(generation)
        if generation > self.problem.config.warmup_generations and self.detect_change():
            self.respond_to_change()
        self.evolve()
        if (generation >= self.problem.config.warmup_generations and
                (generation - self.problem.config.warmup_generations) % self.problem.config.frequency_change == 0):
            self._record_metrics(generation)

    def _record_metrics(self, generation):
        true_pf = self.problem.get_true_pareto_front(n_points=500, generation=generation)
        current_pf = self.get_pareto_front()
        if len(current_pf) == 0 or len(true_pf) == 0:
            return
        igd = PerformanceMetrics.inverted_generational_distance(true_pf, current_pf)
        if not np.isnan(igd) and igd != float("inf"):
            self.environmental_igd.append(igd)
        if self.problem.n_objectives == 2:
            ref_point = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
            hv = PerformanceMetrics.hypervolume_2d(current_pf, ref_point)
        else:
            ref_point = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
            hv = PerformanceMetrics.hypervolume_3d(current_pf, ref_point)
        if hv > 0 and not np.isnan(hv):
            self.environmental_hv.append(hv)

    @abstractmethod
    def initialize(self): pass
    @abstractmethod
    def evolve(self): pass
    @abstractmethod
    def detect_change(self): pass
    @abstractmethod
    def respond_to_change(self): pass


# ============================================================================
# ALGORITHM 1: MOEA/D (BASELINE)
# ============================================================================

class MOEAD(BaseMOEAD):
    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self._update_pareto_archive()

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            child = self._de_operator(i, mating_pool)
            child_obj = self._evaluate(child, self.generation)
            self.ideal_point = np.minimum(self.ideal_point, child_obj)
            replaced = 0
            for j in self.neighbors[i]:
                if replaced >= self.nr:
                    break
                if self._tchebycheff(child_obj, self.weights[j]) < self._tchebycheff(self.objectives[j], self.weights[j]):
                    self.population[j] = child.copy()
                    self.objectives[j] = child_obj.copy()
                    replaced += 1
        self._update_pareto_archive()

    def detect_change(self):
        if self.generation > self.problem.config.warmup_generations:
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self._evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False

    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        n_reinit = int(self.population_size * self.reinit_ratio)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 2: MOEA/D-KNN
# ============================================================================

class MOEADKNN(BaseMOEAD):
    def __init__(self, problem, population_size=POP_SIZE):
        super().__init__(problem, population_size)
        self.history_populations = []
        self.max_history = 3
        self.k_neighbors = 5

    def _knn_prediction(self, current_pop):
        if len(self.history_populations) < 2:
            return current_pop
        past_pop = self.history_populations[-2]
        if len(past_pop) == 0:
            return current_pop
        try:
            knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(past_pop)))
            knn.fit(past_pop)
            predicted = np.zeros_like(current_pop)
            for i, solution in enumerate(current_pop):
                distances, indices = knn.kneighbors(solution.reshape(1, -1))
                if len(indices[0]) > 0:
                    weights = 1.0 / (distances[0] + 1e-8)
                    weights = weights / np.sum(weights)
                    predicted[i] = np.zeros(self.problem.n_variables)
                    for j, idx in enumerate(indices[0]):
                        predicted[i] += weights[j] * past_pop[idx % len(past_pop)]
                else:
                    predicted[i] = solution
        except Exception:
            return current_pop
        return np.clip(predicted, self.lb, self.ub)

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_populations.append(self.population.copy())
        self._update_pareto_archive()

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            child = self._de_operator(i, mating_pool)
            child_obj = self._evaluate(child, self.generation)
            self.ideal_point = np.minimum(self.ideal_point, child_obj)
            replaced = 0
            for j in self.neighbors[i]:
                if replaced >= self.nr:
                    break
                if self._tchebycheff(child_obj, self.weights[j]) < self._tchebycheff(self.objectives[j], self.weights[j]):
                    self.population[j] = child.copy()
                    self.objectives[j] = child_obj.copy()
                    replaced += 1
        self._update_pareto_archive()

    def detect_change(self):
        if self.generation > self.problem.config.warmup_generations:
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self._evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False

    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        self.history_populations.append(self.population.copy())
        if len(self.history_populations) > self.max_history:
            self.history_populations.pop(0)
        predicted = self._knn_prediction(self.population)
        self.population = predicted
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 3: MOEA/D-PPS
# ============================================================================

class MOEADPPS(BaseMOEAD):
    def __init__(self, problem, population_size=POP_SIZE):
        super().__init__(problem, population_size)
        self.history_centers = []
        self.history_populations = []
        self.max_history = 3

    def _compute_center(self, population):
        return np.mean(population, axis=0)

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_centers.append(self._compute_center(self.population))
        self.history_populations.append(self.population.copy())
        self._update_pareto_archive()

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            child = self._de_operator(i, mating_pool)
            child_obj = self._evaluate(child, self.generation)
            self.ideal_point = np.minimum(self.ideal_point, child_obj)
            replaced = 0
            for j in self.neighbors[i]:
                if replaced >= self.nr:
                    break
                if self._tchebycheff(child_obj, self.weights[j]) < self._tchebycheff(self.objectives[j], self.weights[j]):
                    self.population[j] = child.copy()
                    self.objectives[j] = child_obj.copy()
                    replaced += 1
        self._update_pareto_archive()

    def detect_change(self):
        if self.generation > self.problem.config.warmup_generations:
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self._evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False

    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        self.history_centers.append(self._compute_center(self.population))
        self.history_populations.append(self.population.copy())
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
            self.history_populations.pop(0)
        if len(self.history_centers) >= 2:
            centers = np.array(self.history_centers[-2:])
            predicted_center = 2 * centers[-1] - centers[-2]
            shift = predicted_center - centers[-1]
            self.population = self.history_populations[-1] + shift
            self.population = np.clip(self.population, self.lb, self.ub)
        if len(self.population) < self.population_size:
            additional = self.population_size - len(self.population)
            new_inds = np.random.uniform(self.lb, self.ub, (additional, self.problem.n_variables))
            self.population = np.vstack([self.population, new_inds])
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 4: MOEA/D-AGR
# ============================================================================

class MOEADAGR(BaseMOEAD):
    def __init__(self, problem, population_size=POP_SIZE):
        super().__init__(problem, population_size)
        self.history_solutions = []
        self.history_centroids = []
        self.max_history = 4
        self.agr_alpha = 0.3
        self.change_severity = 0.5

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_solutions.append(self.population.copy())
        self.history_centroids.append(np.mean(self.population, axis=0))
        self._update_pareto_archive()

    def _predict_with_agr(self):
        if len(self.history_solutions) < 2:
            return self.population
        centroids = np.array([np.mean(hist, axis=0) for hist in self.history_solutions])
        if len(centroids) < 2:
            return self.population
        movements = centroids[-1] - centroids[-2]
        predicted_centroid = centroids[-1] + movements
        if self.change_severity > 0.7:
            shift = predicted_centroid - centroids[-1]
            predicted_pop = self.history_solutions[-1] + shift
        else:
            shift = self.agr_alpha * (predicted_centroid - centroids[-1])
            predicted_pop = self.history_solutions[-1] + shift
        n = len(predicted_pop)
        for i in range(n):
            noise = np.random.randn(self.problem.n_variables) * 0.05 * self.change_severity
            predicted_pop[i] += noise * (self.ub - self.lb)
        return np.clip(predicted_pop, self.lb, self.ub)

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            child = self._de_operator(i, mating_pool)
            child_obj = self._evaluate(child, self.generation)
            self.ideal_point = np.minimum(self.ideal_point, child_obj)
            replaced = 0
            for j in self.neighbors[i]:
                if replaced >= self.nr:
                    break
                if self._tchebycheff(child_obj, self.weights[j]) < self._tchebycheff(self.objectives[j], self.weights[j]):
                    self.population[j] = child.copy()
                    self.objectives[j] = child_obj.copy()
                    replaced += 1
        self._update_pareto_archive()

    def detect_change(self):
        if self.generation > self.problem.config.warmup_generations:
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self._evaluate(self.population[test_idx], self.generation)
            if not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6):
                self.change_severity = min(1.0, self.change_severity * 1.1)
                return True
        self.change_severity = max(0.1, self.change_severity * 0.95)
        return False

    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        self.history_solutions.append(self.population.copy())
        self.history_centroids.append(np.mean(self.population, axis=0))
        if len(self.history_solutions) > self.max_history:
            self.history_solutions.pop(0)
            self.history_centroids.pop(0)
        predicted = self._predict_with_agr()
        blend_ratio = min(0.8, self.change_severity)
        self.population = (1 - blend_ratio) * self.population + blend_ratio * predicted
        self.population = np.clip(self.population, self.lb, self.ub)
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        n_reinit = int(self.population_size * 0.2 * self.change_severity)
        if n_reinit > 0:
            reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
            for idx in reinit_indices:
                self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
                self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 5: MOEA/D-HSS
# ============================================================================

class MOEADHSS(BaseMOEAD):
    def __init__(self, problem, population_size=POP_SIZE):
        super().__init__(problem, population_size)
        self.search_range = 0.15
        self.range_adapt_rate = 0.05
        self.change_intensity_history = []

    def _adaptive_search_mutation(self, parent, change_intensity=0.5):
        child = parent.copy()
        adaptive_range = self.search_range * (1 + change_intensity)
        adaptive_range = max(0.05, min(0.3, adaptive_range))
        for j in range(len(parent)):
            if np.random.rand() < 0.15:
                step = adaptive_range * (self.ub[j] - self.lb[j]) * np.random.randn()
                child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child

    def _estimate_change_intensity(self):
        if len(self.environmental_igd) < 2:
            return 0.5
        recent_igd = self.environmental_igd[-1]
        prev_igd = self.environmental_igd[-2]
        intensity = min(1.0, abs(recent_igd - prev_igd) / prev_igd) if prev_igd > 0 else 0.5
        self.change_intensity_history.append(intensity)
        if len(self.change_intensity_history) > 5:
            self.change_intensity_history.pop(0)
        return float(np.mean(self.change_intensity_history)) if self.change_intensity_history else 0.5

    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self._update_pareto_archive()

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        intensity = self._estimate_change_intensity()
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            child = self._de_operator(i, mating_pool)
            child = self._adaptive_search_mutation(child, intensity)
            child_obj = self._evaluate(child, self.generation)
            self.ideal_point = np.minimum(self.ideal_point, child_obj)
            replaced = 0
            for j in self.neighbors[i]:
                if replaced >= self.nr:
                    break
                if self._tchebycheff(child_obj, self.weights[j]) < self._tchebycheff(self.objectives[j], self.weights[j]):
                    self.population[j] = child.copy()
                    self.objectives[j] = child_obj.copy()
                    replaced += 1
        if len(self.environmental_igd) > 3:
            recent_perf = np.mean(self.environmental_igd[-3:])
            if len(self.environmental_igd) >= 6:
                prev_perf = np.mean(self.environmental_igd[-6:-3])
                if recent_perf < prev_perf:
                    self.search_range = max(0.05, self.search_range * (1 - self.range_adapt_rate))
                else:
                    self.search_range = min(0.3, self.search_range * (1 + self.range_adapt_rate))
        self._update_pareto_archive()

    def detect_change(self):
        if self.generation > self.problem.config.warmup_generations:
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self._evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False

    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        self.search_range = min(0.3, self.search_range * 1.3)
        n_reinit = int(self.population_size * 0.25)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 6: MOEA/D-RV
# ============================================================================

class MOEADRV(BaseMOEAD):
    """
    MOEA/D-RV: risk-guided mutation (magnitude only) + PPS-style prediction
    + hypervolume-guided selection.

    Hyperparameters (matching manuscript Table 5):
    - base mutation        = 0.15
    - risk sensitivity     = 1.5
    - risk history length  = 5
    - M_base               = 30
    - n_approx (m=3 HV)    = 2000
    """

    def __init__(self, problem, population_size=POP_SIZE):
        super().__init__(problem, population_size)

        # Risk-guided mutation
        self.risk_history = {j: deque(maxlen=5) for j in range(problem.n_variables)}
        self.risk_sensitivity = 1.5
        self.base_mutation = 0.15
        self.risk_momentum = np.ones(problem.n_variables)

        # Prediction (PPS-style)
        self.history_centers = []
        self.history_populations = []
        self.max_history = 3

        # Hypervolume selection
        self.hv_reference = None

        # Enhanced change detection
        self.change_threshold = 1e-6
        self.detection_samples = 5

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

    # ---- RISK-GUIDED MUTATION (magnitude only) ----

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
            risk_min = np.min(all_risks)
            risk_max = np.max(all_risks)
            risk_norm = (self.risk_momentum[j] - risk_min) / (risk_max - risk_min + 1e-8)
            risk_norm = float(np.clip(risk_norm, 0, 1))

            sigma = self.base_mutation * (1 + self.risk_sensitivity * risk_norm)

            # Symmetric polynomial mutation (no directional bias)
            u = np.random.rand()
            if u < 0.5:
                delta_val = (2 * u) ** (1 / (self.eta_m + 1)) - 1
            else:
                delta_val = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))

            step = sigma * (self.ub[j] - self.lb[j]) * delta_val
            child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child

    # ---- PREDICTION ----

    def _predict_population(self):
        if len(self.history_populations) < 2:
            return self.population
        centers = np.array(self.history_centers)
        if len(centers) >= 2:
            predicted_center = 2 * centers[-1] - centers[-2]
            shift = predicted_center - centers[-1]
            max_shift = 0.5 * (self.ub - self.lb)
            shift = np.clip(shift, -max_shift, max_shift)
            predicted_pop = self.history_populations[-1] + shift
            return np.clip(predicted_pop, self.lb, self.ub)
        return self.population

    # ---- HYPERVOLUME SELECTION ----

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

    def _hypervolume_selection(self, candidates, candidate_objs):
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        if self.problem.n_objectives == 2:
            sorted_idx = np.argsort(all_obj[:, 0])
            all_obj_sorted = all_obj[sorted_idx]
            contributions = np.zeros(n_total)
            for i in range(n_total):
                left_x = self.hv_reference[0] if i == 0 else all_obj_sorted[i - 1, 0]
                right_x = self.hv_reference[0] if i == n_total - 1 else all_obj_sorted[i + 1, 0]
                width = min(right_x, self.hv_reference[0]) - max(left_x, 0)
                height = self.hv_reference[1] - all_obj_sorted[i, 1]
                contributions[i] = max(0, width * height)
            if np.sum(contributions) == 0:
                return self._crowding_selection(candidates, candidate_objs)
            elite_idx_in_sorted = np.argsort(contributions)[-self.population_size:]
            elite_idx = sorted_idx[elite_idx_in_sorted]
        else:
            contributions = np.zeros(n_total)
            for i in range(min(n_total, 100)):
                temp_obj = np.delete(all_obj, i, axis=0)
                hv_without = PerformanceMetrics.hypervolume_3d(temp_obj, self.hv_reference, 2000)
                hv_with = PerformanceMetrics.hypervolume_3d(all_obj, self.hv_reference, 2000)
                contributions[i] = hv_with - hv_without
            if np.sum(contributions) == 0:
                return self._crowding_selection(candidates, candidate_objs)
            contributions[100:] = np.mean(contributions[:100])
            elite_idx = np.argsort(contributions)[-self.population_size:]
        return all_pop[elite_idx], all_obj[elite_idx]

    def detect_change(self):
        if self.generation <= self.problem.config.warmup_generations:
            return False
        if self.population is None or self.objectives is None or len(self.population) == 0:
            return False
        n_samples = min(self.detection_samples, self.population_size)
        test_indices = np.random.choice(self.population_size, n_samples, replace=False)
        changes_detected = 0
        for idx in test_indices:
            new_obj = self._evaluate(self.population[idx], self.generation)
            if not np.allclose(new_obj, self.objectives[idx], atol=self.change_threshold):
                changes_detected += 1
        return changes_detected > n_samples // 2

    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        offspring_pop = []
        offspring_obj = []
        for i in range(self.population_size):
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            n_tournament = min(3, len(mating_pool))
            idx1 = np.random.choice(mating_pool, n_tournament, replace=False)
            idx2 = np.random.choice(mating_pool, n_tournament, replace=False)
            fitness1 = np.sum(self.objectives[idx1], axis=1)
            fitness2 = np.sum(self.objectives[idx2], axis=1)
            parent1 = self.population[idx1[np.argmin(fitness1)]]
            parent2 = self.population[idx2[np.argmin(fitness2)]]
            child = parent1.copy()
            eta = 20
            for j in range(self.problem.n_variables):
                if np.random.rand() < 0.5 and abs(parent1[j] - parent2[j]) > 1e-10:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    child[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                    child[j] = np.clip(child[j], self.lb[j], self.ub[j])
            child = self._risk_guided_mutation(child, i)
            offspring_pop.append(child)
            offspring_obj.append(self._evaluate(child, self.generation))
        offspring_pop = np.array(offspring_pop)
        offspring_obj = np.array(offspring_obj)
        self.population, self.objectives = self._hypervolume_selection(offspring_pop, offspring_obj)
        self.ideal_point = np.min(self.objectives, axis=0)
        self._update_pareto_archive()

    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        self.history_centers.append(np.mean(self.population, axis=0))
        self.history_populations.append(self.population.copy())
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
            self.history_populations.pop(0)
        if len(self.history_populations) >= 2:
            predicted = self._predict_population()
            blend_ratio = 0.7
            self.population = (1 - blend_ratio) * self.population + blend_ratio * predicted
            self.population = np.clip(self.population, self.lb, self.ub)
        self.objectives = np.array([self._evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        # Reset the risk memory; do not grow base_mutation (keeps Table 5 consistent)
        self.risk_history = {j: deque(maxlen=5) for j in range(self.problem.n_variables)}
        self.risk_momentum = np.ones(self.problem.n_variables)
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self._evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# VISUALIZATION
# ============================================================================

def _aligned_trajectories(trajectories):
    if not trajectories:
        return None, None
    max_len = max(len(t) for t in trajectories if len(t) > 0)
    if max_len == 0:
        return None, None
    aligned = []
    for t in trajectories:
        if len(t) == 0:
            continue
        padded = np.pad(t, (0, max_len - len(t)), constant_values=t[-1]) if len(t) < max_len else t[:max_len]
        aligned.append(padded)
    if not aligned:
        return None, None
    return np.mean(aligned, axis=0), np.std(aligned, axis=0)


def _plot_metric(all_results, problem_name, algorithm_names, metric_key, ylabel, filename, save_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for algo_name in algorithm_names:
        if problem_name in all_results and algo_name in all_results[problem_name]:
            trajs = all_results[problem_name][algo_name].get(metric_key, [])
            avg, std = _aligned_trajectories(trajs)
            if avg is None:
                continue
            gens = np.arange(len(avg)) * 10
            color = COLORS.get(algo_name, "#888888")
            ax.plot(gens, avg, label=algo_name, color=color, linewidth=2.5)
            ax.fill_between(gens, avg - std, avg + std, alpha=0.2, color=color)
    ax.set_xlabel("Environmental Changes", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(f"{problem_name}: {ylabel.split(' ')[0]} Convergence ({N_RUNS} runs, mean ± std)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    if "IGD" in ylabel:
        ax.set_yscale("log")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_individual_igd_convergence(all_results, problem_name, algorithm_names, save_path=None):
    _plot_metric(all_results, problem_name, algorithm_names,
                 "full_igd_trajectories", "IGD (lower is better)",
                 f"{problem_name}_igd_convergence.png", save_path)


def plot_individual_mhv_convergence(all_results, problem_name, algorithm_names, save_path=None):
    _plot_metric(all_results, problem_name, algorithm_names,
                 "full_hv_trajectories", "MHV (higher is better)",
                 f"{problem_name}_mhv_convergence.png", save_path)


def plot_all_individual_curves(all_results, problem_names, algorithm_names, save_path=None):
    print("\nGenerating individual convergence curves...")
    if save_path is None:
        save_path = str(OUTPUT_DIR / "individual_curves")
    os.makedirs(save_path, exist_ok=True)
    for pname in problem_names:
        print(f"  Plotting {pname}...")
        plot_individual_igd_convergence(all_results, pname, algorithm_names, save_path)
        plot_individual_mhv_convergence(all_results, pname, algorithm_names, save_path)
    print(f"  Saved plots to {save_path}/")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(problem_name, run_id, algorithm_class, seed_offset):
    try:
        config = ProblemConfig(
            n_variables=N_VARIABLES,
            n_objectives=2,
            frequency_change=FREQUENCY_CHANGE,
            severity_change=SEVERITY_CHANGE,
            warmup_generations=WARMUP_GENERATIONS,
            n_changes=N_CHANGES,
            population_size=POP_SIZE,
        )
        problem = get_problem_class(problem_name)(config)
        np.random.seed(run_id * 1000 + seed_offset)
        algo = algorithm_class(problem, config.population_size)
        algo.initialize()

        full_igd = []
        full_hv = []
        t_start = time.time()
        for gen in range(config.max_generations):
            algo.step(gen)
            if gen % 10 == 0 or gen == config.max_generations - 1:
                true_pf = problem.get_true_pareto_front(n_points=500, generation=gen)
                current_pf = algo.get_pareto_front()
                if len(current_pf) > 0 and len(true_pf) > 0:
                    igd = PerformanceMetrics.inverted_generational_distance(true_pf, current_pf)
                    if not np.isnan(igd) and igd != float("inf"):
                        full_igd.append(igd)
                    if problem.n_objectives == 2:
                        ref_point = REF_POINTS.get(problem_name, np.array([2, 2]))
                        hv = PerformanceMetrics.hypervolume_2d(current_pf, ref_point)
                    else:
                        ref_point = REF_POINTS.get(problem_name, np.array([2, 2, 2]))
                        hv = PerformanceMetrics.hypervolume_3d(current_pf, ref_point)
                    if hv > 0 and not np.isnan(hv):
                        full_hv.append(hv)
        runtime = time.time() - t_start
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


def run_set1_benchmark():
    test_problems = ["DF1", "DF2", "DF3", "DF4", "DF5", "DF6", "DF7", "DF8", "DF9",
                     "DF10", "DF11", "DF12", "DF13", "DF14"]
    algorithm_classes = [MOEAD, MOEADKNN, MOEADPPS, MOEADAGR, MOEADHSS, MOEADRV]
    algorithm_names = ["MOEA/D", "MOEA/D-KNN", "MOEA/D-PPS", "MOEA/D-AGR", "MOEA/D-HSS", "MOEA/D-RV"]

    all_results = {}
    for pname in test_problems:
        all_results[pname] = {}
        for aname in algorithm_names:
            all_results[pname][aname] = {
                "mean_migd": float("inf"), "std_migd": 0.0,
                "mean_mhv": 0.0, "std_mhv": 0.0,
                "mean_runtime": 0.0, "mean_evals": 0,
                "full_igd_trajectories": [], "full_hv_trajectories": [],
            }

    print("=" * 100)
    print(f"MOEA/D-RV BENCHMARK  |  N_RUNS = {N_RUNS}")
    print("=" * 100)
    total_start = time.time()

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
                if (run + 1) % 10 == 0:
                    print(f"    {run + 1}/{N_RUNS} runs complete", flush=True)
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
    print("MIGD SUMMARY (lower is better)")
    print("=" * 100)
    print(f"{'Problem':<10}" + "".join(f" {n:<20}" for n in algorithm_names))
    wins = {n: 0 for n in algorithm_names}
    for pname in test_problems:
        row_best = min(all_results[pname][n]["mean_migd"] for n in algorithm_names)
        print(f"{pname:<10}", end="")
        for n in algorithm_names:
            v = all_results[pname][n]["mean_migd"]
            s = all_results[pname][n]["std_migd"]
            mark = "*" if abs(v - row_best) < 1e-12 else " "
            print(f" {v:.4e}±{s:.2e}{mark}", end="")
            if mark == "*":
                wins[n] += 1
        print()
    print("\nWin counts (MIGD):")
    for n in algorithm_names:
        print(f"  {n}: {wins[n]}/{len(test_problems)}")

    plot_all_individual_curves(all_results, test_problems, algorithm_names,
                               save_path=str(OUTPUT_DIR / "individual_curves"))

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "population_size": POP_SIZE,
            "max_generations": MAX_GENERATIONS,
            "frequency_change": FREQUENCY_CHANGE,
            "severity_change": SEVERITY_CHANGE,
            "warmup_generations": WARMUP_GENERATIONS,
            "n_changes": N_CHANGES,
            "n_runs": N_RUNS,
        },
        "algorithm_names": algorithm_names,
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
    fn = OUTPUT_DIR / f"moead_rv_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {fn}")
    return all_results


if __name__ == "__main__":
    print("=" * 100)
    print("MOEA/D-RV: SET 1 — STANDARD BENCHMARK")
    print(f"6 MOEA/D Variants | 14 DF Problems | {N_RUNS} Runs")
    print("=" * 100)
    results = run_set1_benchmark()
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)
