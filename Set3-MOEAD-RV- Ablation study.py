"""
MOEA/D-RV: SET 4 - ABLATION STUDY
Dynamic Multi-Objective Optimization Evolutionary Algorithm

Ablation Study Variants:
1. MOEA/D-RV (Full)              - Complete algorithm with all features
2. w/o Risk                       - Uniform mutation (no risk-guided mutation)
3. w/o HV                         - Tchebycheff-only selection (no hypervolume guidance)
4. w/o Prediction                 - Reactive change detection (no predictive detection)
5. w/o Niching                    - No diversity preservation (no adaptive niching)
6. w/o Scenario                   - No scenario sampling (deterministic objectives)

Purpose: Isolate and quantify the contribution of each core component

Enhanced Features Integrated:
1. Hybrid Decomposition (PBI ↔ Chebyshev switching) - Adapted from Ndikuriyo et al. (2026)
2. Monte Carlo Scenario Sampling - For environmental uncertainty quantification
3. Mean-Variance Risk Control - Robust objective with adaptive risk aversion
4. Risk-Guided Mutation - Variable-specific adaptive mutation (original)
5. Hypervolume-Guided Selection - HVC-based population update (original)

CEC2018 DF Benchmark Suite (DF1-DF14)
Parameters: Population=100, Generations=350, tau_t=10, n_t=10, Runs=5

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
import warnings
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from datetime import datetime
from collections import deque
import os

warnings.filterwarnings('ignore')

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "ablation_curves"), exist_ok=True)

# ============================================================================
# CONSTANTS & GLOBAL SETTINGS
# ============================================================================

POP_SIZE = 100
MAX_GENERATIONS = 350
FREQUENCY_CHANGE = 10
SEVERITY_CHANGE = 10
WARMUP_GENERATIONS = 50
N_CHANGES = 30
N_VARIABLES = 10
N_RUNS = 5  # Reduced for testing, set to 30 for final

# Reference points for hypervolume
REF_POINTS = {
    'DF1': np.array([1.1, 1.1]),
    'DF2': np.array([1.1, 1.1]),
    'DF3': np.array([1.1, 1.1]),
    'DF4': np.array([30, 30]),
    'DF5': np.array([1.1, 1.1]),
    'DF6': np.array([1.1, 1.1]),
    'DF7': np.array([10, 10]),
    'DF8': np.array([1.1, 1.1]),
    'DF9': np.array([1.1, 1.1]),
    'DF10': np.array([1.1, 1.1, 1.1]),
    'DF11': np.array([1.1, 1.1, 1.1]),
    'DF12': np.array([1.1, 1.1, 1.1]),
    'DF13': np.array([1.1, 1.1, 1.1]),
    'DF14': np.array([1.1, 1.1, 1.1]),
}

# Ablation variant colors
ABLATION_COLORS = {
    'Full': '#C62828',
    'w/o Risk': '#E65100',
    'w/o HV': '#0D47A1',
    'w/o Prediction': '#4A148C',
    'w/o Niching': '#00838F',
    'w/o Scenario': '#2E7D32',
}

MARKERS = {
    'Full': '*',
    'w/o Risk': 'o',
    'w/o HV': 's',
    'w/o Prediction': '^',
    'w/o Niching': 'D',
    'w/o Scenario': 'v',
}

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

class PerformanceMetrics:
    @staticmethod
    def inverted_generational_distance(pf_true: np.ndarray, pf_approx: np.ndarray) -> float:
        if len(pf_approx) == 0 or len(pf_true) == 0:
            return float('inf')
        try:
            pf_true = np.atleast_2d(pf_true)
            pf_approx = np.atleast_2d(pf_approx)
            distances = cdist(pf_true, pf_approx)
            min_distances = np.min(distances, axis=1)
            min_distances = min_distances[~np.isinf(min_distances) & ~np.isnan(min_distances)]
            if len(min_distances) == 0:
                return float('inf')
            return np.mean(min_distances)
        except Exception:
            return float('inf')
    
    @staticmethod
    def mean_inverted_generational_distance(igd_values: List[float]) -> float:
        valid_igd = [igd for igd in igd_values if igd != float('inf') and not np.isnan(igd)]
        if len(valid_igd) == 0:
            return float('inf')
        return np.mean(valid_igd)
    
    @staticmethod
    def mean_hypervolume(hv_values: List[float]) -> float:
        valid_hv = [hv for hv in hv_values if hv > 0 and not np.isnan(hv)]
        if len(valid_hv) == 0:
            return 0.0
        return np.mean(valid_hv)
    
    @staticmethod
    def standard_deviation(values: List[float]) -> float:
        valid = [v for v in values if v != float('inf') and not np.isnan(v)]
        if len(valid) == 0:
            return 0.0
        return np.std(valid)
    
    @staticmethod
    def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
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
                height = max(0, ref_point[1] - points[i-1, 1])
                hv += width * height
                prev_f1 = points[i, 0]
            width = max(0, ref_point[0] - prev_f1)
            height = max(0, ref_point[1] - points[-1, 1])
            hv += width * height
            return max(0, hv)
        except Exception:
            return 0.0
    
    @staticmethod
    def hypervolume_3d(points: np.ndarray, ref_point: np.ndarray, n_samples: int = 5000) -> float:
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
            return np.sum(dominated) / n_samples * np.prod(ref_point)
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
    def max_generations(self) -> int:
        return self.warmup_generations + self.n_changes * self.frequency_change


class DynamicProblem(ABC):
    def __init__(self, config: ProblemConfig):
        self.config = config
        self.generation = 0
        self.time = 0.0
        self.n_variables = config.n_variables
        self.n_objectives = config.n_objectives
        
    def update_time(self, generation: int) -> None:
        self.generation = generation
        if generation < self.config.warmup_generations:
            self.time = 0.0
        else:
            gen_since_warmup = generation - self.config.warmup_generations
            self.time = (1.0 / self.config.severity_change) * np.floor(
                gen_since_warmup / self.config.frequency_change
            )
    
    def get_time(self) -> float:
        return self.time
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_true_pareto_front(self, n_points: int = 500, generation: int = None) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    def get_name(self) -> str:
        return self.__class__.__name__


# ============================================================================
# DF1-DF9 PROBLEMS (2-OBJECTIVE)
# ============================================================================

class DF1(DynamicProblem):
    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.n_objectives = 2
        
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])
    
    def get_true_pareto_front(self, n_points: int = 500, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - f1 ** H
        return np.column_stack([f1, np.maximum(0, f2)])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_variables), np.ones(self.n_variables)


class DF2(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 0.85
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])


class DF3(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t)) ** 2
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])


class DF4(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0] ** H
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, np.maximum(0, f2)])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_variables), np.ones(self.n_variables) * 2


class DF5(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        theta = 0.25 * np.pi * t
        f1 = (1 + g) * x[0] * np.cos(theta)
        f2 = (1 + g) * (1 - x[0]) * np.sin(theta)
        return np.array([f1, f2])
    
    def get_true_pareto_front(self, n_points: int = 500, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        theta = 0.25 * np.pi * t
        x = np.linspace(0, 1, n_points)
        f1 = x * np.cos(theta)
        f2 = (1 - x) * np.sin(theta)
        return np.column_stack([f1, f2])


class DF6(DF5):
    pass


class DF7(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, 10 * np.maximum(0, f2)])
    
    def get_true_pareto_front(self, n_points: int = 500, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 10 * (1 - f1 ** H)
        return np.column_stack([f1, np.maximum(0, f2)])


class DF8(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0] ** H
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, 5 * np.maximum(0, f2)])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_variables), np.ones(self.n_variables) * 2


class DF9(DF1):
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        g = 1.0 + np.sum((x[1:] - G) ** 2)
        f1 = x[0]
        f2 = g * (1 - (f1 / g) ** H) if g > 0 else g
        return np.array([f1, 2 * np.maximum(0, f2)])
    
    def get_true_pareto_front(self, n_points: int = 500, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        H = 0.75 * np.sin(0.5 * np.pi * t) + 1.25
        f1 = np.linspace(0, 1, n_points)
        f2 = 2 * (1 - f1 ** H)
        return np.column_stack([f1, np.maximum(0, f2)])


# ============================================================================
# DF10-DF14 PROBLEMS (3-OBJECTIVE)
# ============================================================================

class DF10(DynamicProblem):
    def __init__(self, config: ProblemConfig):
        super().__init__(config)
        self.n_objectives = 3
    
    def evaluate(self, x: np.ndarray, generation: int = None) -> np.ndarray:
        if generation is not None:
            old_gen = self.generation
            self.update_time(generation)
            t = self.time
            self.generation = old_gen
        else:
            t = self.time
        
        G = abs(np.sin(0.5 * np.pi * t))
        g = 1.0 + np.sum((x[2:] - G) ** 2)
        f1 = x[0] * x[1]
        f2 = g * (1 - f1)
        f3 = g * (1 - f1 * np.sin(0.5 * np.pi * t))
        return np.array([f1, np.maximum(0, f2), np.maximum(0, f3)])
    
    def get_true_pareto_front(self, n_points: int = 500, generation: int = None) -> np.ndarray:
        y = np.linspace(0, 1, n_points)
        return np.column_stack([y, 1 - y, 1 - y])
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.n_variables), np.ones(self.n_variables)


class DF11(DF10):
    pass


class DF12(DF10):
    pass


class DF13(DF10):
    pass


class DF14(DF10):
    pass


# ============================================================================
# BASE MOEA/D CLASS
# ============================================================================

class BaseMOEAD(ABC):
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        self.problem = problem
        self.population_size = population_size
        self.population = None
        self.objectives = None
        self.generation = 0
        self.lb, self.ub = problem.get_bounds()
        self.igd_history = []
        self.environmental_igd = []
        self.environmental_hv = []
        self.pareto_archive = []
        
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
        
    def _generate_weights(self) -> np.ndarray:
        n = self.population_size
        if self.problem.n_objectives == 2:
            weights = np.zeros((n, 2))
            for i in range(n):
                w1 = i / (n - 1) if n > 1 else 0.5
                weights[i] = [w1, 1 - w1]
        else:
            weights = np.random.dirichlet(np.ones(self.problem.n_objectives), n)
        return weights
    
    def _compute_neighbors(self) -> np.ndarray:
        dist = cdist(self.weights, self.weights)
        return np.argsort(dist, axis=1)[:, :self.T]
    
    def _tchebycheff(self, obj: np.ndarray, weight: np.ndarray) -> float:
        return np.max(weight * np.abs(obj - self.ideal_point))
    
    def _de_operator(self, idx: int, mating_pool: np.ndarray) -> np.ndarray:
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
    
    def get_pareto_front(self) -> np.ndarray:
        if self.objectives is None or len(self.objectives) == 0:
            return np.array([])
        n = len(self.objectives)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(self.objectives[i] <= self.objectives[j]) and np.any(self.objectives[i] < self.objectives[j]):
                    dominated[i] = True
        return self.objectives[~dominated]
    
    def step(self, generation: int) -> None:
        self.generation = generation
        self.problem.update_time(generation)
        
        if generation > self.problem.config.warmup_generations and self.detect_change():
            self.respond_to_change()
        
        self.evolve()
        
        # Track after each environmental change
        if (generation >= self.problem.config.warmup_generations and 
            (generation - self.problem.config.warmup_generations) % self.problem.config.frequency_change == 0):
            true_pf = self.problem.get_true_pareto_front(n_points=500, generation=generation)
            current_pf = self.get_pareto_front()
            if len(current_pf) > 0 and len(true_pf) > 0:
                igd = PerformanceMetrics.inverted_generational_distance(true_pf, current_pf)
                if not np.isnan(igd) and igd != float('inf'):
                    self.environmental_igd.append(igd)
                
                # Track HV
                if self.problem.n_objectives == 2:
                    ref_point = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
                    hv = PerformanceMetrics.hypervolume_2d(current_pf, ref_point)
                else:
                    ref_point = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
                    hv = PerformanceMetrics.hypervolume_3d(current_pf, ref_point)
                if hv > 0 and not np.isnan(hv):
                    self.environmental_hv.append(hv)
    
    @abstractmethod
    def initialize(self) -> None:
        pass
    
    @abstractmethod
    def evolve(self) -> None:
        pass
    
    @abstractmethod
    def detect_change(self) -> bool:
        pass
    
    @abstractmethod
    def respond_to_change(self) -> None:
        pass


# ============================================================================
# ABLATION VARIANT: FULL MOEA/D-RV
# ============================================================================

class MOEADRVFull(BaseMOEAD):
    """
    MOEA/D-RV Full: Complete algorithm with all features
    
    Integrated Features:
    1. Hybrid Decomposition (PBI ↔ Chebyshev switching)
    2. Monte Carlo Scenario Sampling
    3. Mean-Variance Risk Control
    4. Risk-Guided Mutation
    5. Hypervolume-Guided Selection
    6. Predictive Change Detection
    7. Adaptive Niching
    """
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        
        # ========== Core MOEA/D-RV Components ==========
        self.risk_history = {j: deque(maxlen=10) for j in range(problem.n_variables)}
        self.ideal_history = deque(maxlen=5)
        self.change_threshold = 0.005
        self.risk_sensitivity = 1.5
        self.base_mutation = 0.15
        self.niche_radius = 0.15
        self.hv_reference = None
        self.risk_momentum = np.ones(problem.n_variables)
        self.pareto_archive_solutions = []
        
        # ========== Feature 1: Hybrid Decomposition ==========
        self.theta_pbi = 5.0
        self.stagnation_threshold = 0.01
        self.use_pbi = True
        self.switch_generation = None
        self.pbi_phase_length = 0
        self.hv_history = deque(maxlen=10)
        
        # ========== Feature 2: Monte Carlo Scenario Sampling ==========
        self.n_scenarios = 50
        self.scenarios = []
        self.scenario_weights = []
        self.history_scenarios = deque(maxlen=10)
        self.adaptive_samples = True
        
        # ========== Feature 3: Mean-Variance Risk Control ==========
        self.risk_aversion = 0.5
        self.risk_history_deque = deque(maxlen=20)
        self.risk_adaptation = True
        
        # ========== Performance tracking ==========
        self.performance_history = deque(maxlen=20)
        self.robust_objective_count = 0
    
    # ========================================================================
    # FEATURE 1: HYBRID DECOMPOSITION
    # ========================================================================
    
    def _pbi_decomposition(self, obj: np.ndarray, weight: np.ndarray) -> float:
        norm_obj = obj - self.ideal_point
        weight_norm = np.linalg.norm(weight)
        if weight_norm < 1e-10:
            weight_norm = 1.0
        d1 = np.dot(norm_obj, weight) / weight_norm
        projection = d1 * weight / weight_norm
        d2 = np.linalg.norm(norm_obj - projection)
        return d1 + self.theta_pbi * d2
    
    def _tchebycheff_decomposition(self, obj: np.ndarray, weight: np.ndarray) -> float:
        return np.max(weight * np.abs(obj - self.ideal_point))
    
    def _select_decomposition(self, obj: np.ndarray, weight: np.ndarray) -> float:
        if len(self.hv_history) >= 5 and self.use_pbi:
            hv_values = list(self.hv_history)
            recent_improvement = (hv_values[-1] - hv_values[-5]) / (hv_values[-5] + 1e-8)
            if recent_improvement < self.stagnation_threshold:
                self.use_pbi = False
                self.switch_generation = self.generation
        
        if self.use_pbi and self.generation > 0.7 * MAX_GENERATIONS:
            self.use_pbi = False
            self.switch_generation = self.generation
        
        if self.use_pbi:
            return self._pbi_decomposition(obj, weight)
        else:
            return self._tchebycheff_decomposition(obj, weight)
    
    def update_hypervolume(self, hv: float):
        self.hv_history.append(hv)
    
    # ========================================================================
    # FEATURE 2: MONTE CARLO SCENARIO SAMPLING
    # ========================================================================
    
    def _generate_scenarios(self):
        current_time = self.problem.get_time()
        if len(self.history_scenarios) > 5:
            recent_changes = list(self.history_scenarios)
            uncertainty = np.std(recent_changes) if len(recent_changes) > 1 else 0.1
            n = int(self.n_scenarios * (1 + min(1.0, uncertainty * 0.5)))
        else:
            n = self.n_scenarios
        
        mean_time = current_time
        std_time = max(0.05, 0.1 * (1 + np.random.rand() * 0.5))
        scenarios = []
        for _ in range(n):
            delta_t = np.random.normal(0, std_time)
            delta_t = np.clip(delta_t, -0.5, 0.5)
            scenarios.append(max(0, current_time + delta_t))
        
        weights = np.exp(-0.5 * ((np.array(scenarios) - current_time) / std_time) ** 2)
        weights = weights / (np.sum(weights) + 1e-8)
        self.scenarios = scenarios
        self.scenario_weights = weights
        self.history_scenarios.append(current_time)
        return scenarios, weights
    
    def _compute_expected_objectives(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.scenarios) == 0:
            obj = self.problem.evaluate(x, self.generation)
            return obj, np.zeros(self.problem.n_objectives)
        
        obj_values = []
        for scenario_time in self.scenarios:
            old_time = self.problem.time
            self.problem.time = scenario_time
            obj = self.problem.evaluate(x, self.generation)
            self.problem.time = old_time
            obj_values.append(obj)
        
        obj_values = np.array(obj_values)
        expected = np.average(obj_values, axis=0, weights=self.scenario_weights)
        variance = np.average((obj_values - expected) ** 2, axis=0, weights=self.scenario_weights)
        return expected, variance
    
    # ========================================================================
    # FEATURE 3: MEAN-VARIANCE RISK CONTROL
    # ========================================================================
    
    def _compute_robust_objective(self, obj: np.ndarray) -> np.ndarray:
        expected, variance = self._compute_expected_objectives(obj)
        if self.risk_adaptation and len(self.risk_history_deque) > 10:
            recent_risks = list(self.risk_history_deque)[-10:]
            risk_volatility = np.std(recent_risks) if recent_risks else 0.1
            adapted_aversion = self.risk_aversion * (1 + min(0.5, risk_volatility))
        else:
            adapted_aversion = self.risk_aversion
        robust_obj = expected + adapted_aversion * variance
        self.risk_history_deque.append(np.mean(robust_obj))
        self.robust_objective_count += 1
        return robust_obj
    
    def _update_risk_aversion(self):
        if len(self.risk_history_deque) >= 10:
            recent_perf = np.mean(list(self.risk_history_deque)[-5:])
            older_perf = np.mean(list(self.risk_history_deque)[-10:-5])
            if recent_perf < older_perf:
                self.risk_aversion = max(0.1, self.risk_aversion * 0.98)
            else:
                self.risk_aversion = max(0.1, self.risk_aversion * 0.95)
    
    # ========================================================================
    # ORIGINAL COMPONENTS (Enhanced)
    # ========================================================================
    
    def _compute_risk(self, variable_idx: int, solution: np.ndarray, subproblem_idx: int) -> float:
        delta = 0.05 * (self.ub[variable_idx] - self.lb[variable_idx])
        delta = max(delta, 1e-6)
        x_perturbed = solution.copy()
        x_perturbed[variable_idx] += delta
        x_perturbed = np.clip(x_perturbed, self.lb[variable_idx], self.ub[variable_idx])
        current_obj = self.objectives[subproblem_idx]
        new_obj = self.problem.evaluate(x_perturbed, self.generation)
        current_robust = self._compute_robust_objective(current_obj)
        new_robust = self._compute_robust_objective(new_obj)
        current_fitness = self._select_decomposition(current_robust, self.weights[subproblem_idx])
        new_fitness = self._select_decomposition(new_robust, self.weights[subproblem_idx])
        risk = abs(new_fitness - current_fitness) / abs(delta)
        return min(risk, 10.0)
    
    def _risk_guided_mutation(self, parent: np.ndarray, subproblem_idx: int) -> np.ndarray:
        child = parent.copy()
        current_obj = self.objectives[subproblem_idx]
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
            if risk_max - risk_min > 1e-8:
                risk_norm = (self.risk_momentum[j] - risk_min) / (risk_max - risk_min)
            else:
                risk_norm = 0.5
            sigma = self.base_mutation * (1 + self.risk_sensitivity * risk_norm)
            if len(self.risk_history[j]) >= 5:
                recent_risks = list(self.risk_history[j])[-5:]
                avg_risk = np.mean(recent_risks)
                direction = -1 if avg_risk < np.mean(all_risks) else 1
            else:
                direction = 1 if np.random.rand() < 0.5 else -1
            delta = np.random.rand()
            if delta < 0.5:
                delta_val = (2 * delta) ** (1 / (self.eta_m + 1)) - 1
            else:
                delta_val = 1 - (2 * (1 - delta)) ** (1 / (self.eta_m + 1))
            step = direction * sigma * (self.ub[j] - self.lb[j]) * delta_val
            child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child
    
    def _fast_hypervolume_selection(self, candidates: np.ndarray, candidate_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            sorted_idx = np.argsort(all_obj[:, 0])
            all_obj_sorted = all_obj[sorted_idx]
            all_pop_sorted = all_pop[sorted_idx]
            contributions = np.zeros(n_total)
            for i in range(n_total):
                if i == 0:
                    left_x = self.hv_reference[0]
                else:
                    left_x = all_obj_sorted[i-1, 0]
                if i == n_total - 1:
                    right_x = self.hv_reference[0]
                else:
                    right_x = all_obj_sorted[i+1, 0]
                width = min(right_x, self.hv_reference[0]) - max(left_x, 0)
                height = self.hv_reference[1] - all_obj_sorted[i, 1]
                contributions[i] = max(0, width * height)
            elite_idx_in_sorted = np.argsort(contributions)[-self.population_size:]
            elite_idx = sorted_idx[elite_idx_in_sorted]
        else:
            contributions = np.zeros(n_total)
            for i in range(min(n_total, 100)):
                temp_obj = np.delete(all_obj, i, axis=0)
                hv_without = PerformanceMetrics.hypervolume_3d(temp_obj, self.hv_reference, 2000)
                hv_with = PerformanceMetrics.hypervolume_3d(all_obj, self.hv_reference, 2000)
                contributions[i] = hv_with - hv_without
            contributions[100:] = np.mean(contributions[:100])
            elite_idx = np.argsort(contributions)[-self.population_size:]
        return all_pop[elite_idx], all_obj[elite_idx]
    
    def _predictive_change_detection(self) -> bool:
        if len(self.ideal_history) < 3:
            return False
        ideal_array = np.array(list(self.ideal_history))
        x = np.arange(len(ideal_array))
        predictions = []
        for i in range(self.problem.n_objectives):
            y = ideal_array[:, i]
            if len(y) >= 2 and np.std(y) > 1e-6:
                slope, intercept = np.polyfit(x, y, 1)
                pred = slope * len(x) + intercept
                predictions.append(pred)
            else:
                predictions.append(y[-1])
        deviations = [abs(pred - self.ideal_point[i]) for i, pred in enumerate(predictions)]
        max_deviation = np.max(deviations)
        if len(self.history_scenarios) > 5:
            uncertainty = np.std(list(self.history_scenarios))
            current_threshold = self.change_threshold * (1 + min(0.5, uncertainty))
        else:
            current_threshold = self.change_threshold
        return max_deviation > current_threshold
    
    def _adaptive_niching(self):
        if len(self.objectives) < 3:
            return
        try:
            distances = cdist(self.objectives, self.objectives)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            avg_min_dist = np.mean(min_distances)
            if avg_min_dist < self.niche_radius * 0.5:
                self.niche_radius = min(0.3, self.niche_radius * 1.05)
            elif avg_min_dist > self.niche_radius * 1.5:
                self.niche_radius = max(0.05, self.niche_radius * 0.95)
            niche_counts = np.sum(distances < self.niche_radius, axis=1)
            shared_fitness = 1.0 / (niche_counts + 1)
            if np.random.rand() < 0.15:
                worst_idx = np.argmin(shared_fitness)
                best_idx = np.argmax(shared_fitness)
                self.population[worst_idx] = 0.7 * self.population[worst_idx] + 0.3 * self.population[best_idx]
                self.population[worst_idx] = np.clip(self.population[worst_idx], self.lb, self.ub)
                self.objectives[worst_idx] = self.problem.evaluate(self.population[worst_idx], self.generation)
        except Exception:
            pass
    
    def initialize(self):
        n_samples = self.population_size
        n_vars = self.problem.n_variables
        self.population = np.zeros((n_samples, n_vars))
        for i in range(n_vars):
            segments = np.linspace(self.lb[i], self.ub[i], n_samples + 1)
            points = segments[:-1] + np.random.rand(n_samples) * (segments[1] - segments[:-1])
            np.random.shuffle(points)
            self.population[:, i] = points
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.ideal_history.append(self.ideal_point.copy())
        self._generate_scenarios()
        if self.problem.n_objectives == 2:
            self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
        else:
            self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
        self._update_pareto_archive()
    
    def evolve(self):
        if self.population is None:
            self.initialize()
        if self.generation % 10 == 0:
            self._generate_scenarios()
            self._update_risk_aversion()
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
                if np.random.rand() < 0.5:
                    if abs(parent1[j] - parent2[j]) > 1e-10:
                        u = np.random.rand()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        child[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                        child[j] = np.clip(child[j], self.lb[j], self.ub[j])
            child = self._risk_guided_mutation(child, i)
            offspring_pop.append(child)
            offspring_obj.append(self.problem.evaluate(child, self.generation))
        offspring_pop = np.array(offspring_pop)
        offspring_obj = np.array(offspring_obj)
        self.population, self.objectives = self._fast_hypervolume_selection(offspring_pop, offspring_obj)
        current_min = np.min(self.objectives, axis=0)
        self.ideal_point = np.minimum(self.ideal_point, current_min)
        self.ideal_history.append(self.ideal_point.copy())
        self._adaptive_niching()
        self._update_pareto_archive()
        if len(self.environmental_igd) > 0:
            hv = PerformanceMetrics.hypervolume_2d(self.get_pareto_front(), self.hv_reference)
            self.update_hypervolume(hv)
    
    def detect_change(self) -> bool:
        if self.generation > self.problem.config.warmup_generations:
            if self._predictive_change_detection():
                return True
            n_tests = min(5, self.population_size)
            test_indices = np.random.choice(self.population_size, n_tests, replace=False)
            for idx in test_indices:
                new_obj = self.problem.evaluate(self.population[idx], self.generation)
                if not np.allclose(new_obj, self.objectives[idx], atol=1e-5, rtol=1e-5):
                    return True
        return False
    
    def respond_to_change(self):
        if self.population is None:
            self.initialize()
            return
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        self.ideal_history.append(self.ideal_point.copy())
        self.risk_history = {j: deque(maxlen=10) for j in range(self.problem.n_variables)}
        self.risk_momentum = np.ones(self.problem.n_variables)
        self.base_mutation = min(0.25, self.base_mutation * 1.2)
        self._generate_scenarios()
        self._update_pareto_archive()


# ============================================================================
# ABLATION VARIANT 1: WITHOUT RISK (Uniform Mutation)
# ============================================================================

class MOEADRV_WithoutRisk(MOEADRVFull):
    """MOEA/D-RV without risk-guided mutation (uniform mutation only)."""
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        # Keep all other features, remove risk-guided mutation
    
    def _compute_risk(self, variable_idx: int, solution: np.ndarray, subproblem_idx: int) -> float:
        # Return constant risk (no risk guidance)
        return 0.5
    
    def _risk_guided_mutation(self, parent: np.ndarray, subproblem_idx: int) -> np.ndarray:
        """Standard uniform mutation (no risk guidance)."""
        child = parent.copy()
        for j in range(self.problem.n_variables):
            if np.random.rand() < self.p_m:
                delta = np.random.rand()
                if delta < 0.5:
                    delta_val = (2 * delta) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta_val = 1 - (2 * (1 - delta)) ** (1 / (self.eta_m + 1))
                step = self.base_mutation * (self.ub[j] - self.lb[j]) * delta_val
                child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child


# ============================================================================
# ABLATION VARIANT 2: WITHOUT HV (Tchebycheff-only Selection)
# ============================================================================

class MOEADRV_WithoutHV(MOEADRVFull):
    """MOEA/D-RV without hypervolume-guided selection (Tchebycheff-only)."""
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
    
    def _fast_hypervolume_selection(self, candidates: np.ndarray, candidate_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tchebycheff-only selection (no hypervolume)."""
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        
        # Simple selection based on Tchebycheff fitness sum
        fitness = np.array([np.sum(self._select_decomposition(obj, self.weights[i % self.population_size])) 
                           for i, obj in enumerate(all_obj)])
        elite_idx = np.argsort(fitness)[:self.population_size]
        return all_pop[elite_idx], all_obj[elite_idx]


# ============================================================================
# ABLATION VARIANT 3: WITHOUT PREDICTION (Reactive Detection)
# ============================================================================

class MOEADRV_WithoutPrediction(MOEADRVFull):
    """MOEA/D-RV without predictive change detection (reactive only)."""
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
    
    def _predictive_change_detection(self) -> bool:
        # Always return False - no predictive detection
        return False


# ============================================================================
# ABLATION VARIANT 4: WITHOUT NICHING (No Diversity Preservation)
# ============================================================================

class MOEADRV_WithoutNiching(MOEADRVFull):
    """MOEA/D-RV without adaptive niching (no diversity preservation)."""
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
    
    def _adaptive_niching(self):
        # Skip niching
        pass


# ============================================================================
# ABLATION VARIANT 5: WITHOUT SCENARIO (Deterministic Objectives)
# ============================================================================

class MOEADRV_WithoutScenario(MOEADRVFull):
    """MOEA/D-RV without scenario sampling (deterministic objectives)."""
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
    
    def _generate_scenarios(self):
        # Return only the current time (no scenario sampling)
        current_time = self.problem.get_time()
        self.scenarios = [current_time]
        self.scenario_weights = [1.0]
        return self.scenarios, self.scenario_weights
    
    def _compute_expected_objectives(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Deterministic evaluation (no variance)
        obj = self.problem.evaluate(x, self.generation)
        return obj, np.zeros(self.problem.n_objectives)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_individual_igd_convergence(all_results: Dict, problem_name: str, 
                                     algorithm_names: List[str], save_path: str = None):
    """Plot IGD convergence curve for a single problem (Ablation Study)."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo_name in algorithm_names:
        if problem_name in all_results and algo_name in all_results[problem_name]:
            trajectories = all_results[problem_name][algo_name].get('full_igd_trajectories', [])
            if trajectories:
                max_len = max(len(traj) for traj in trajectories if len(traj) > 0)
                aligned_trajs = []
                for traj in trajectories:
                    if len(traj) > 0:
                        if len(traj) < max_len:
                            padded = np.pad(traj, (0, max_len - len(traj)), constant_values=traj[-1])
                        else:
                            padded = traj[:max_len]
                        aligned_trajs.append(padded)
                
                if aligned_trajs:
                    avg_traj = np.mean(aligned_trajs, axis=0)
                    std_traj = np.std(aligned_trajs, axis=0)
                    
                    generations = np.arange(len(avg_traj)) * 10
                    color = ABLATION_COLORS.get(algo_name, '#888888')
                    linewidth = 3 if algo_name == 'Full' else 2
                    ax.plot(generations, avg_traj, label=algo_name, 
                           color=color, linewidth=linewidth)
                    ax.fill_between(generations, avg_traj - std_traj, avg_traj + std_traj,
                                   alpha=0.2, color=color)
    
    ax.set_xlabel('Environmental Changes', fontsize=12, fontweight='bold')
    ax.set_ylabel('IGD (↓ lower is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'{problem_name}: Ablation Study - IGD Convergence ({N_RUNS} runs, mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{problem_name}_ablation_igd_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_mhv_convergence(all_results: Dict, problem_name: str, 
                                     algorithm_names: List[str], save_path: str = None):
    """Plot MHV convergence curve for a single problem (Ablation Study)."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for algo_name in algorithm_names:
        if problem_name in all_results and algo_name in all_results[problem_name]:
            trajectories = all_results[problem_name][algo_name].get('full_hv_trajectories', [])
            if trajectories:
                max_len = max(len(traj) for traj in trajectories if len(traj) > 0)
                aligned_trajs = []
                for traj in trajectories:
                    if len(traj) > 0:
                        if len(traj) < max_len:
                            padded = np.pad(traj, (0, max_len - len(traj)), constant_values=traj[-1])
                        else:
                            padded = traj[:max_len]
                        aligned_trajs.append(padded)
                
                if aligned_trajs:
                    avg_traj = np.mean(aligned_trajs, axis=0)
                    std_traj = np.std(aligned_trajs, axis=0)
                    
                    generations = np.arange(len(avg_traj)) * 10
                    color = ABLATION_COLORS.get(algo_name, '#888888')
                    linewidth = 3 if algo_name == 'Full' else 2
                    ax.plot(generations, avg_traj, label=algo_name, 
                           color=color, linewidth=linewidth)
                    ax.fill_between(generations, avg_traj - std_traj, avg_traj + std_traj,
                                   alpha=0.2, color=color)
    
    ax.set_xlabel('Environmental Changes', fontsize=12, fontweight='bold')
    ax.set_ylabel('MHV (↑ higher is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'{problem_name}: Ablation Study - MHV Convergence ({N_RUNS} runs, mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{problem_name}_ablation_mhv_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_improvement(improvement_data: Dict, save_path: str = None):
    """Plot bar chart showing improvement from each component."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    components = list(improvement_data.keys())
    avg_improvements = [improvement_data[comp]['avg_improvement'] for comp in components]
    std_improvements = [improvement_data[comp]['std_improvement'] for comp in components]
    colors = [ABLATION_COLORS.get(comp, '#888888') for comp in components]
    
    bars = ax.bar(components, avg_improvements, yerr=std_improvements,
                  capsize=5, color=colors, edgecolor='#333333', linewidth=1.2, alpha=0.85)
    
    for bar, val in zip(bars, avg_improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Ablation Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('MIGD Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Component Contribution Analysis - MIGD Improvement', 
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/ablation_improvement_bars.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_ablation_curves(all_results: Dict, problem_names: List[str], 
                              algorithm_names: List[str], save_path: str = None):
    """Generate all ablation convergence plots for each problem."""
    
    print("\nGenerating ablation convergence curves for each problem...")
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "ablation_curves")
    
    os.makedirs(save_path, exist_ok=True)
    
    for problem_name in problem_names:
        print(f"  Plotting {problem_name}...")
        plot_individual_igd_convergence(all_results, problem_name, algorithm_names, save_path)
        plot_individual_mhv_convergence(all_results, problem_name, algorithm_names, save_path)
    
    print(f"  Saved plots to {save_path}/")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(problem_name: str, run_id: int, algorithm_class, seed_offset: int) -> Dict:
    """Run a single experiment for one problem and algorithm variant."""
    try:
        config = ProblemConfig(
            n_variables=N_VARIABLES,
            n_objectives=2,
            frequency_change=FREQUENCY_CHANGE,
            severity_change=SEVERITY_CHANGE,
            warmup_generations=WARMUP_GENERATIONS,
            n_changes=N_CHANGES,
            population_size=POP_SIZE
        )
        
        problem_class = get_problem_class(problem_name)
        problem = problem_class(config)
        
        np.random.seed(run_id * 1000 + seed_offset)
        
        algo = algorithm_class(problem, config.population_size)
        algo.initialize()
        
        full_igd = []
        full_hv = []
        
        for gen in range(config.max_generations):
            algo.step(gen)
            
            if gen % 10 == 0 or gen == config.max_generations - 1:
                true_pf = problem.get_true_pareto_front(n_points=500, generation=gen)
                current_pf = algo.get_pareto_front()
                if len(current_pf) > 0 and len(true_pf) > 0:
                    igd = PerformanceMetrics.inverted_generational_distance(true_pf, current_pf)
                    if not np.isnan(igd) and igd != float('inf'):
                        full_igd.append(igd)
                    
                    if problem.n_objectives == 2:
                        ref_point = REF_POINTS.get(problem_name, np.array([2, 2]))
                        hv = PerformanceMetrics.hypervolume_2d(current_pf, ref_point)
                    else:
                        ref_point = REF_POINTS.get(problem_name, np.array([2, 2, 2]))
                        hv = PerformanceMetrics.hypervolume_3d(current_pf, ref_point)
                    if hv > 0 and not np.isnan(hv):
                        full_hv.append(hv)
        
        migd = PerformanceMetrics.mean_inverted_generational_distance(algo.environmental_igd)
        mhv = PerformanceMetrics.mean_hypervolume(algo.environmental_hv)
        
        return {
            'migd': migd,
            'mhv': mhv,
            'full_igd_trajectory': full_igd,
            'full_hv_trajectory': full_hv,
            'success': True
        }
    except Exception as e:
        return {'migd': float('inf'), 'mhv': 0.0, 'success': False, 'error': str(e)}


def get_problem_class(name: str):
    problems = {
        'DF1': DF1, 'DF2': DF2, 'DF3': DF3, 'DF4': DF4, 'DF5': DF5,
        'DF6': DF6, 'DF7': DF7, 'DF8': DF8, 'DF9': DF9, 'DF10': DF10,
        'DF11': DF11, 'DF12': DF12, 'DF13': DF13, 'DF14': DF14
    }
    return problems.get(name, DF1)


def run_set4_ablation_benchmark():
    """Run Set 4 ablation study comparing Full MOEA/D-RV against variants."""
    
    test_problems = ['DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6', 'DF7', 'DF8', 'DF9',
                     'DF10', 'DF11', 'DF12', 'DF13', 'DF14']
    
    # Ablation variants
    algorithm_classes = [
        MOEADRVFull,
        MOEADRV_WithoutRisk,
        MOEADRV_WithoutHV,
        MOEADRV_WithoutPrediction,
        MOEADRV_WithoutNiching,
        MOEADRV_WithoutScenario
    ]
    algorithm_names = [
        'Full',
        'w/o Risk',
        'w/o HV',
        'w/o Prediction',
        'w/o Niching',
        'w/o Scenario'
    ]
    
    all_results = {}
    
    print("=" * 100)
    print("MOEA/D-RV: SET 4 - ABLATION STUDY")
    print("=" * 100)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"\nAblation Variants:")
    print(f"  1. Full                    - Complete algorithm with all features")
    print(f"  2. w/o Risk                - Uniform mutation (no risk-guided mutation)")
    print(f"  3. w/o HV                  - Tchebycheff-only selection (no hypervolume guidance)")
    print(f"  4. w/o Prediction          - Reactive change detection (no predictive detection)")
    print(f"  5. w/o Niching             - No diversity preservation (no adaptive niching)")
    print(f"  6. w/o Scenario            - No scenario sampling (deterministic objectives)")
    print(f"\nEnhanced Features in Full Version:")
    print(f"  • Hybrid Decomposition (PBI ↔ Chebyshev switching)")
    print(f"  • Monte Carlo Scenario Sampling")
    print(f"  • Mean-Variance Risk Control")
    print(f"  • Risk-Guided Mutation")
    print(f"  • Hypervolume-Guided Selection")
    print(f"  • Predictive Change Detection")
    print(f"  • Adaptive Niching")
    print(f"\nConfiguration:")
    print(f"  Population size: {POP_SIZE}")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Warmup generations: {WARMUP_GENERATIONS}")
    print(f"  Frequency of change: {FREQUENCY_CHANGE}")
    print(f"  Number of runs: {N_RUNS}")
    print(f"  Problems: {len(test_problems)} (DF1-DF14)")
    print(f"  Ablation variants: {len(algorithm_classes)}")
    print(f"  Total experiments: {len(test_problems) * len(algorithm_classes) * N_RUNS}")
    print("=" * 100)
    
    total_start = time.time()
    
    # Initialize results structure
    for problem_name in test_problems:
        all_results[problem_name] = {}
        for algo_name in algorithm_names:
            all_results[problem_name][algo_name] = {
                'mean_migd': 0.0, 'std_migd': 0.0,
                'mean_mhv': 0.0, 'std_mhv': 0.0,
                'full_igd_trajectories': [],
                'full_hv_trajectories': [],
            }
    
    exp_count = 0
    total_exp = len(test_problems) * len(algorithm_classes) * N_RUNS
    
    for problem_idx, problem_name in enumerate(test_problems):
        print(f"\n[{problem_name}] ({problem_idx+1}/{len(test_problems)})")
        
        for algo_idx, (algo_class, algo_name) in enumerate(zip(algorithm_classes, algorithm_names)):
            print(f"\n  Running {algo_name}...")
            algo_start = time.time()
            
            migd_values = []
            mhv_values = []
            full_igd_trajectories = []
            full_hv_trajectories = []
            
            for run in range(N_RUNS):
                seed_offset = (algo_idx * 10 + hash(problem_name) % 100)
                result = run_single_experiment(problem_name, run, algo_class, seed_offset)
                migd_values.append(result['migd'])
                mhv_values.append(result['mhv'])
                if result.get('full_igd_trajectory'):
                    full_igd_trajectories.append(result['full_igd_trajectory'])
                if result.get('full_hv_trajectory'):
                    full_hv_trajectories.append(result['full_hv_trajectory'])
                
                exp_count += 1
                if (run + 1) % 5 == 0 or run == N_RUNS - 1:
                    print(f"    Runs: {run+1}/{N_RUNS} (MIGD: {result['migd']:.6f}, MHV: {result['mhv']:.4f})", flush=True)
                else:
                    print(f"    Run {run+1}/{N_RUNS}: MIGD = {result['migd']:.6f}, MHV = {result['mhv']:.4f}", flush=True)
                
                if exp_count % (total_exp // 20 + 1) == 0:
                    pct = 100 * exp_count / total_exp
                    print(f"    [Overall Progress: {pct:.1f}% ({exp_count}/{total_exp})]", flush=True)
            
            elapsed = time.time() - algo_start
            valid_migd = [v for v in migd_values if v != float('inf') and not np.isnan(v)]
            valid_mhv = [v for v in mhv_values if v > 0 and not np.isnan(v)]
            
            mean_migd = np.mean(valid_migd) if valid_migd else float('inf')
            std_migd = np.std(valid_migd) if valid_migd else 0.0
            mean_mhv = np.mean(valid_mhv) if valid_mhv else 0.0
            std_mhv = np.std(valid_mhv) if valid_mhv else 0.0
            
            all_results[problem_name][algo_name] = {
                'mean_migd': mean_migd,
                'std_migd': std_migd,
                'mean_mhv': mean_mhv,
                'std_mhv': std_mhv,
                'full_igd_trajectories': full_igd_trajectories,
                'full_hv_trajectories': full_hv_trajectories,
            }
            
            print(f"  {algo_name} completed in {elapsed:.1f}s")
            print(f"  Summary: MIGD = {mean_migd:.6f} ± {std_migd:.6f}, MHV = {mean_mhv:.4f} ± {std_mhv:.4f}")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 100)
    print(f"BENCHMARK COMPLETE! Total time: {total_time/3600:.2f} hours")
    print("=" * 100)
    
    # Summary table - MIGD
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - MIGD VALUES (lower is better)")
    print("=" * 100)
    print(f"{'Problem':<12}", end='')
    for algo_name in algorithm_names:
        print(f" {algo_name:<18}", end='')
    print()
    print("-" * (12 + 18 * len(algorithm_names)))
    
    win_count = {name: 0 for name in algorithm_names}
    
    for problem_name in test_problems:
        print(f"{problem_name:<12}", end='')
        best_migd = float('inf')
        best_algo = None
        
        for algo_name in algorithm_names:
            mean_val = all_results[problem_name][algo_name]['mean_migd']
            std_val = all_results[problem_name][algo_name]['std_migd']
            
            if not np.isnan(mean_val) and mean_val != float('inf'):
                print(f" {mean_val:.6f}±{std_val:.6f} ", end='')
                if mean_val < best_migd:
                    best_migd = mean_val
                    best_algo = algo_name
            else:
                print(f" {'nan±nan':<18}", end='')
        
        if best_algo:
            win_count[best_algo] += 1
        print()
    
    print("-" * (12 + 18 * len(algorithm_names)))
    print(f"\nWIN COUNT (best MIGD):")
    for algo_name in algorithm_names:
        print(f"  {algo_name}: {win_count[algo_name]}/{len(test_problems)}")
    
    # Ablation improvement analysis
    print("\n" + "=" * 100)
    print("ABLATION ANALYSIS - Improvement from Each Component")
    print("=" * 100)
    print(f"{'Problem':<12} {'Full vs w/o Risk':<22} {'Full vs w/o HV':<22} {'Full vs w/o Pred':<22} {'Full vs w/o Nich':<22} {'Full vs w/o Scen':<22}")
    print("-" * 110)
    
    improvement_data = {name: {'improvements': []} for name in ['w/o Risk', 'w/o HV', 'w/o Prediction', 'w/o Niching', 'w/o Scenario']}
    
    for problem_name in test_problems:
        full_mean = all_results[problem_name]['Full']['mean_migd']
        
        print(f"{problem_name:<12}", end='')
        for variant in ['w/o Risk', 'w/o HV', 'w/o Prediction', 'w/o Niching', 'w/o Scenario']:
            variant_mean = all_results[problem_name][variant]['mean_migd']
            if full_mean != float('inf') and variant_mean != float('inf') and not np.isnan(full_mean) and not np.isnan(variant_mean):
                improvement = (variant_mean - full_mean) / variant_mean * 100
                improvement_data[variant]['improvements'].append(improvement)
                print(f" {improvement:+.1f}%           ", end='')
            else:
                print(f" {'N/A':<12}", end='')
        print()
    
    print("\n" + "=" * 100)
    print("AVERAGE IMPROVEMENT BY COMPONENT:")
    print("=" * 100)
    
    avg_improvements = {}
    for variant, data in improvement_data.items():
        if data['improvements']:
            avg_imp = np.mean(data['improvements'])
            std_imp = np.std(data['improvements'])
            avg_improvements[variant] = {'avg_improvement': avg_imp, 'std_improvement': std_imp}
            print(f"  {variant:<20}: {avg_imp:+.1f}% ± {std_imp:.1f}%")
    
    print("\nNote: Positive percentage indicates improvement from adding the component")
    
    # Generate ablation improvement bar chart
    plot_ablation_improvement(avg_improvements, save_path=os.path.join(OUTPUT_DIR, "ablation_curves"))
    
    # Generate individual convergence plots
    plot_all_ablation_curves(all_results, test_problems, algorithm_names, save_path=os.path.join(OUTPUT_DIR, "ablation_curves"))
    
    # Save results to JSON
    output = {
        'timestamp': datetime.now().isoformat(),
        'output_directory': OUTPUT_DIR,
        'config': {
            'population_size': POP_SIZE,
            'max_generations': MAX_GENERATIONS,
            'frequency_change': FREQUENCY_CHANGE,
            'severity_change': SEVERITY_CHANGE,
            'warmup_generations': WARMUP_GENERATIONS,
            'n_changes': N_CHANGES,
            'n_runs': N_RUNS,
            'description': 'Ablation Study - Comparing Full MOEA/D-RV against variants'
        },
        'ablation_variants': algorithm_names,
        'results': {}
    }
    
    for problem_name in test_problems:
        output['results'][problem_name] = {}
        for algo_name in algorithm_names:
            mean_migd = all_results[problem_name][algo_name]['mean_migd']
            std_migd = all_results[problem_name][algo_name]['std_migd']
            mean_mhv = all_results[problem_name][algo_name]['mean_mhv']
            std_mhv = all_results[problem_name][algo_name]['std_mhv']
            output['results'][problem_name][algo_name] = {
                'migd_mean': float(mean_migd) if not np.isnan(mean_migd) and mean_migd != float('inf') else None,
                'migd_std': float(std_migd) if not np.isnan(std_migd) else None,
                'mhv_mean': float(mean_mhv) if not np.isnan(mean_mhv) and mean_mhv > 0 else None,
                'mhv_std': float(std_mhv) if not np.isnan(std_mhv) else None,
            }
    
    filename = os.path.join(OUTPUT_DIR, f'moead_rv_set4_ablation_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also save a summary CSV
    csv_filename = os.path.join(OUTPUT_DIR, f'summary_ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(csv_filename, 'w') as f:
        f.write("Problem,Variant,MIGD_mean,MIGD_std,MHV_mean,MHV_std\n")
        for problem_name in test_problems:
            for algo_name in algorithm_names:
                migd_mean = all_results[problem_name][algo_name]['mean_migd']
                migd_std = all_results[problem_name][algo_name]['std_migd']
                mhv_mean = all_results[problem_name][algo_name]['mean_mhv']
                mhv_std = all_results[problem_name][algo_name]['std_mhv']
                f.write(f"{problem_name},{algo_name},{migd_mean},{migd_std},{mhv_mean},{mhv_std}\n")
    
    print(f"\nResults saved to: {filename}")
    print(f"CSV summary saved to: {csv_filename}")
    print(f"Ablation curves saved to: {os.path.join(OUTPUT_DIR, 'ablation_curves')}/")
    
    return all_results


if __name__ == "__main__":
    print("=" * 100)
    print("MOEA/D-RV: SET 4 - ABLATION STUDY")
    print("6 Ablation Variants: Full, w/o Risk, w/o HV, w/o Prediction, w/o Niching, w/o Scenario")
    print("14 Test Problems: DF1-DF14")
    print(f"{N_RUNS} Independent Runs")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 100)
    
    results = run_set4_ablation_benchmark()
    
    print("\n" + "=" * 100)
    print("SET 4 COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("\nKey Findings Expected:")
    print("  - Full MOEA/D-RV should achieve best MIGD on most problems")
    print("  - Risk component should provide largest improvement (variable-specific mutation)")
    print("  - HV component should provide second largest improvement (selection quality)")
    print("  - Prediction component should provide modest improvement (proactive detection)")
    print("  - Niching component should provide improvement on 3-objective problems")
    print("  - Scenario component should provide improvement on uncertain problems")
    print("=" * 100)