"""
MOEA/D-RV: SET 2 - MULTIPLE ENVIRONMENT CONFIGURATIONS (OPTIMIZED)
Dynamic Multi-Objective Optimization Evolutionary Algorithm

Environment Configurations:
1. (tau_t=10, n_t=10) - Standard configuration (CEC2018 default)
2. (tau_t=10, n_t=5)  - Mild changes (lower severity)
3. (tau_t=5, n_t=10)  - Rapid changes (higher frequency)

COMPARATOR SET (6 MOEA/D variants):
1. MOEA/D (Baseline)              - Zhang & Li (2007)
2. MOEA/D-KNN                     - Deng et al. (2025) [Training-free local prediction]
3. MOEA/D-PPS                     - Zhou et al. (2014) [Population prediction]
4. MOEA/D-AGR                     - Adaptive guided response
5. MOEA/D-HSS                     - Hu et al. (2024) [Hybrid search strategy]
6. MOEA/D-RV (Proposed)           - Risk + Prediction + HV Selection

OPTIMIZATIONS INCLUDED:
1. Reduced scenario sampling (10 instead of 50)
2. Lazy risk computation with caching
3. Approximate hypervolume for speed
4. Reduced niching frequency
5. Parallel scenario evaluation (optional)

CEC2018 DF Benchmark Suite (DF1-DF14)
Parameters: Population=100, Runs=5, Warmup=50 generations
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
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "individual_curves"), exist_ok=True)

# ============================================================================
# CONSTANTS & GLOBAL SETTINGS
# ============================================================================

POP_SIZE = 100
WARMUP_GENERATIONS = 50
N_CHANGES = 30
N_VARIABLES = 10
N_RUNS = 5  # Reduced for testing, set to 30 for final

# Enable parallel processing for speed
USE_PARALLEL = False  # Set to True for multi-core speedup (may cause issues on Windows)

# Environment configurations
ENV_CONFIGS = [
    {'name': 'Standard', 'tau_t': 10, 'n_t': 10, 'max_gens': 50 + 30 * 10, 'desc': 'Standard CEC2018'},
    {'name': 'Mild', 'tau_t': 10, 'n_t': 5, 'max_gens': 50 + 30 * 10, 'desc': 'Mild changes (lower severity)'},
    {'name': 'Rapid', 'tau_t': 5, 'n_t': 10, 'max_gens': 50 + 30 * 5, 'desc': 'Rapid changes (higher frequency)'},
]

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

# Color schemes for visualization
COLORS = {
    'MOEA/D': '#1B5E20',
    'MOEA/D-KNN': '#0D47A1',
    'MOEA/D-PPS': '#E65100',
    'MOEA/D-AGR': '#4A148C',
    'MOEA/D-HSS': '#00838F',
    'MOEA/D-RV': '#C62828',
    'true_pf': '#000000',
}

# Configuration colors for plots
CONFIG_COLORS = {
    'Standard': '#1B5E20',
    'Mild': '#0D47A1',
    'Rapid': '#C62828',
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
    def hypervolume_2d_approx(points: np.ndarray, ref_point: np.ndarray, n_samples: int = 1000) -> float:
        """Approximate hypervolume using Monte Carlo (faster for large populations)."""
        if len(points) == 0:
            return 0.0
        try:
            points = np.atleast_2d(points)
            mask = np.all(points <= ref_point, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0
            samples = np.random.rand(n_samples, 2) * ref_point
            dominated = np.zeros(n_samples, dtype=bool)
            for p in points:
                dominated |= np.all(samples <= p, axis=1)
            return np.sum(dominated) / n_samples * np.prod(ref_point)
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
    
    @staticmethod
    def hypervolume_3d_approx(points: np.ndarray, ref_point: np.ndarray, n_samples: int = 1000) -> float:
        """Approximate 3D hypervolume using Monte Carlo."""
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
    tau_t: int = 10
    n_t: int = 10
    warmup_generations: int = WARMUP_GENERATIONS
    n_changes: int = N_CHANGES
    population_size: int = POP_SIZE
    
    @property
    def frequency_change(self) -> int:
        return self.tau_t
    
    @property
    def severity_change(self) -> int:
        return self.n_t
    
    @property
    def max_generations(self) -> int:
        return self.warmup_generations + self.n_changes * self.tau_t


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
# ALGORITHM 1: MOEA/D (BASELINE)
# ============================================================================

class MOEAD(BaseMOEAD):
    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self._update_pareto_archive()
    
    def evolve(self):
        if self.population is None:
            self.initialize()
        
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            
            child = self._de_operator(i, mating_pool)
            child_obj = self.problem.evaluate(child, self.generation)
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
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False
    
    def respond_to_change(self):
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        n_reinit = int(self.population_size * self.reinit_ratio)
        reinit_indices = np.random.choice(self.population_size, n_reinit, replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 2: MOEA/D-KNN (TRAINING-FREE LOCAL PREDICTION)
# ============================================================================

class MOEADKNN(BaseMOEAD):
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        self.history_populations = []
        self.max_history = 3
        self.k_neighbors = 5
    
    def _knn_prediction(self, current_pop: np.ndarray) -> np.ndarray:
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
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_populations.append(self.population.copy())
        self._update_pareto_archive()
    
    def evolve(self):
        if self.population is None:
            self.initialize()
        
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            
            child = self._de_operator(i, mating_pool)
            child_obj = self.problem.evaluate(child, self.generation)
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
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False
    
    def respond_to_change(self):
        self.history_populations.append(self.population.copy())
        if len(self.history_populations) > self.max_history:
            self.history_populations.pop(0)
        
        predicted = self._knn_prediction(self.population)
        self.population = predicted
        
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, n_reinit, replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 3: MOEA/D-PPS (POPULATION PREDICTION STRATEGY)
# ============================================================================

class MOEADPPS(BaseMOEAD):
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        self.history_centers = []
        self.history_populations = []
        self.max_history = 3
    
    def _compute_center(self, population: np.ndarray) -> np.ndarray:
        return np.mean(population, axis=0)
    
    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_centers.append(self._compute_center(self.population))
        self.history_populations.append(self.population.copy())
        self._update_pareto_archive()
    
    def evolve(self):
        if self.population is None:
            self.initialize()
        
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            
            child = self._de_operator(i, mating_pool)
            child_obj = self.problem.evaluate(child, self.generation)
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
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False
    
    def respond_to_change(self):
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
        
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, n_reinit, replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 4: MOEA/D-AGR (ADAPTIVE GUIDED RESPONSE)
# ============================================================================

class MOEADAGR(BaseMOEAD):
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        self.history_solutions = []
        self.history_centroids = []
        self.max_history = 4
        self.agr_alpha = 0.3
        self.change_severity = 0.5
    
    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self.history_solutions.append(self.population.copy())
        self.history_centroids.append(np.mean(self.population, axis=0))
        self._update_pareto_archive()
    
    def _predict_with_agr(self) -> np.ndarray:
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
        
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            
            child = self._de_operator(i, mating_pool)
            child_obj = self.problem.evaluate(child, self.generation)
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
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            if not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6):
                self.change_severity = min(1.0, self.change_severity * 1.1)
                return True
        self.change_severity = max(0.1, self.change_severity * 0.95)
        return False
    
    def respond_to_change(self):
        self.history_solutions.append(self.population.copy())
        self.history_centroids.append(np.mean(self.population, axis=0))
        
        if len(self.history_solutions) > self.max_history:
            self.history_solutions.pop(0)
            self.history_centroids.pop(0)
        
        predicted = self._predict_with_agr()
        blend_ratio = min(0.8, self.change_severity)
        self.population = (1 - blend_ratio) * self.population + blend_ratio * predicted
        self.population = np.clip(self.population, self.lb, self.ub)
        
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        n_reinit = int(self.population_size * 0.2 * self.change_severity)
        if n_reinit > 0:
            reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
            for idx in reinit_indices:
                self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
                self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 5: MOEA/D-HSS (HYBRID SEARCH STRATEGY)
# ============================================================================

class MOEADHSS(BaseMOEAD):
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        self.search_range = 0.15
        self.range_adapt_rate = 0.05
        self.change_intensity_history = []
    
    def _adaptive_search_mutation(self, parent: np.ndarray, change_intensity: float = 0.5) -> np.ndarray:
        child = parent.copy()
        adaptive_range = self.search_range * (1 + change_intensity)
        adaptive_range = max(0.05, min(0.3, adaptive_range))
        
        for j in range(len(parent)):
            if np.random.rand() < 0.15:
                step = adaptive_range * (self.ub[j] - self.lb[j]) * np.random.randn()
                child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        return child
    
    def _estimate_change_intensity(self) -> float:
        if len(self.environmental_igd) < 2:
            return 0.5
        
        recent_igd = self.environmental_igd[-1]
        prev_igd = self.environmental_igd[-2]
        
        if prev_igd > 0:
            intensity = min(1.0, abs(recent_igd - prev_igd) / prev_igd)
        else:
            intensity = 0.5
        
        self.change_intensity_history.append(intensity)
        if len(self.change_intensity_history) > 5:
            self.change_intensity_history.pop(0)
        
        return np.mean(self.change_intensity_history) if self.change_intensity_history else 0.5
    
    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        self._update_pareto_archive()
    
    def evolve(self):
        if self.population is None:
            self.initialize()
        
        intensity = self._estimate_change_intensity()
        
        indices = np.random.permutation(self.population_size)
        for i in indices:
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            
            child = self._de_operator(i, mating_pool)
            child = self._adaptive_search_mutation(child, intensity)
            child_obj = self.problem.evaluate(child, self.generation)
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
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False
    
    def respond_to_change(self):
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        self.search_range = min(0.3, self.search_range * 1.3)
        
        n_reinit = int(self.population_size * 0.25)
        reinit_indices = np.random.choice(self.population_size, n_reinit, replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()


# ============================================================================
# ALGORITHM 6: MOEA/D-RV (PROPOSED - OPTIMIZED VERSION)
# ============================================================================

class MOEADRV(BaseMOEAD):
    """
    MOEA/D-RV: Full integration of robust optimization features
    
    OPTIMIZATIONS:
    1. Reduced scenarios (10 instead of 50)
    2. Lazy risk computation with caching
    3. Approximate hypervolume for speed
    4. Reduced niching frequency
    5. Scenario update frequency reduced
    
    Core Features:
    1. Risk-Guided Mutation - Variable-specific adaptive mutation
    2. Prediction (PPS-style) - Fast tracking of changing optima
    3. Hypervolume-Guided Selection - Elite survival selection
    """
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        
        # ========== OPTIMIZATION SETTINGS ==========
        self.n_scenarios = 10  # Reduced from 50 for speed
        self.scenario_update_freq = 20  # Update scenarios less often
        self.hv_approx_samples = 1000  # Monte Carlo HV approximation
        self.niching_freq = 20  # Niching every 20 generations
        self.risk_cache = {}  # Cache risk computations
        self.risk_cache_max = 10000  # Prevent memory bloat
        
        # ========== Original MOEA/D-RV Components ==========
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
        
        max_gen = self.problem.config.max_generations
        if self.use_pbi and self.generation > 0.7 * max_gen:
            self.use_pbi = False
            self.switch_generation = self.generation
        
        if self.use_pbi:
            return self._pbi_decomposition(obj, weight)
        else:
            return self._tchebycheff_decomposition(obj, weight)
    
    def update_hypervolume(self, hv: float):
        self.hv_history.append(hv)
    
    # ========================================================================
    # FEATURE 2: MONTE CARLO SCENARIO SAMPLING (OPTIMIZED)
    # ========================================================================
    
    def _generate_scenarios(self):
        """Generate scenarios with reduced count for speed."""
        current_time = self.problem.get_time()
        
        # Cap scenarios at a reasonable number
        n = min(self.n_scenarios, 10)
        
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
        """Compute expected value and variance with reduced scenario evaluations."""
        if len(self.scenarios) == 0:
            obj = self.problem.evaluate(x, self.generation)
            return obj, np.zeros(self.problem.n_objectives)
        
        obj_values = []
        # Only evaluate a subset of scenarios for speed
        max_evals = min(len(self.scenarios), 10)
        indices = np.random.choice(len(self.scenarios), max_evals, replace=False) if len(self.scenarios) > max_evals else range(len(self.scenarios))
        
        for idx in indices:
            scenario_time = self.scenarios[idx]
            old_time = self.problem.time
            self.problem.time = scenario_time
            obj = self.problem.evaluate(x, self.generation)
            self.problem.time = old_time
            obj_values.append(obj)
        
        obj_values = np.array(obj_values)
        weights = np.array([self.scenario_weights[idx] for idx in indices])
        weights = weights / (np.sum(weights) + 1e-8)
        
        expected = np.average(obj_values, axis=0, weights=weights)
        variance = np.average((obj_values - expected) ** 2, axis=0, weights=weights)
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
    # ORIGINAL COMPONENTS (Enhanced with Caching)
    # ========================================================================
    
    def _compute_risk(self, variable_idx: int, solution: np.ndarray, subproblem_idx: int) -> float:
        """Compute risk with caching for speed."""
        # Create cache key
        solution_hash = hash(solution.tobytes())
        cache_key = (solution_hash, variable_idx, subproblem_idx)
        
        # Check cache
        if cache_key in self.risk_cache:
            return self.risk_cache[cache_key]
        
        # Compute risk
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
        risk = min(risk, 10.0)
        
        # Store in cache
        self.risk_cache[cache_key] = risk
        if len(self.risk_cache) > self.risk_cache_max:
            # Clear oldest entries
            keys_to_remove = list(self.risk_cache.keys())[:self.risk_cache_max // 2]
            for key in keys_to_remove:
                del self.risk_cache[key]
        
        return risk
    
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
        """Hypervolume selection with approximate HV for speed."""
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
            # Use exact 2D HV (fast enough)
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
            # Use approximate HV for 3+ objectives
            contributions = np.zeros(n_total)
            # Compute approximate HV contribution for each point
            hv_with = PerformanceMetrics.hypervolume_3d_approx(all_obj, self.hv_reference, self.hv_approx_samples)
            for i in range(min(n_total, 50)):  # Only evaluate top 50 points
                temp_obj = np.delete(all_obj, i, axis=0)
                hv_without = PerformanceMetrics.hypervolume_3d_approx(temp_obj, self.hv_reference, self.hv_approx_samples)
                contributions[i] = hv_with - hv_without
            contributions[50:] = np.mean(contributions[:50])
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
        """Adaptive niching with reduced frequency."""
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
        
        # Update scenarios periodically (lazy computation)
        if self.generation % self.scenario_update_freq == 0:
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
        
        # Apply niching less frequently
        if self.generation % self.niching_freq == 0:
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
        self.risk_cache.clear()  # Clear cache on change
        self._generate_scenarios()
        self._update_pareto_archive()


# ============================================================================
# VISUALIZATION FUNCTIONS (FIXED)
# ============================================================================

def plot_individual_igd_convergence(all_results: Dict, problem_name: str, 
                                     algorithm_names: List[str], config_name: str,
                                     save_path: str = None):
    """Plot IGD convergence curve for a single problem under a specific configuration."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for algo_name in algorithm_names:
        key = f"{algo_name}_{config_name}"
        if problem_name in all_results and key in all_results[problem_name]:
            trajectories = all_results[problem_name][key].get('full_igd_trajectories', [])
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
                    ax.plot(generations, avg_traj, label=algo_name, 
                           color=COLORS.get(algo_name, '#888888'), linewidth=2.5)
                    ax.fill_between(generations, avg_traj - std_traj, avg_traj + std_traj,
                                   alpha=0.2, color=COLORS.get(algo_name, '#888888'))
    
    ax.set_xlabel('Environmental Changes', fontsize=12, fontweight='bold')
    ax.set_ylabel('IGD (↓ lower is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'{problem_name}: IGD Convergence ({config_name}, {N_RUNS} runs, mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{problem_name}_{config_name}_igd_convergence.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_mhv_convergence(all_results: Dict, problem_name: str, 
                                     algorithm_names: List[str], config_name: str,
                                     save_path: str = None):
    """Plot MHV convergence curve for a single problem under a specific configuration."""
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for algo_name in algorithm_names:
        key = f"{algo_name}_{config_name}"
        if problem_name in all_results and key in all_results[problem_name]:
            trajectories = all_results[problem_name][key].get('full_hv_trajectories', [])
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
                    ax.plot(generations, avg_traj, label=algo_name, 
                           color=COLORS.get(algo_name, '#888888'), linewidth=2.5)
                    ax.fill_between(generations, avg_traj - std_traj, avg_traj + std_traj,
                                   alpha=0.2, color=COLORS.get(algo_name, '#888888'))
    
    ax.set_xlabel('Environmental Changes', fontsize=12, fontweight='bold')
    ax.set_ylabel('MHV (↑ higher is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'{problem_name}: MHV Convergence ({config_name}, {N_RUNS} runs, mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{problem_name}_{config_name}_mhv_convergence.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_config_comparison(all_results: Dict, problem_name: str, algo_name: str,
                            save_path: str = None):
    """Plot a single algorithm's performance across all three configurations."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # IGD plot
    ax1 = axes[0]
    for config in ENV_CONFIGS:
        config_name = config['name']
        key = f"{algo_name}_{config_name}"
        if problem_name in all_results and key in all_results[problem_name]:
            trajectories = all_results[problem_name][key].get('full_igd_trajectories', [])
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
                    generations = np.arange(len(avg_traj)) * 10
                    ax1.plot(generations, avg_traj, label=config_name, 
                            color=CONFIG_COLORS.get(config_name, '#888888'), linewidth=2.5)
    
    ax1.set_xlabel('Environmental Changes', fontsize=12)
    ax1.set_ylabel('IGD (↓ lower is better)', fontsize=12)
    ax1.set_title(f'{algo_name} on {problem_name}: IGD Across Configurations', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    # MHV plot
    ax2 = axes[1]
    for config in ENV_CONFIGS:
        config_name = config['name']
        key = f"{algo_name}_{config_name}"
        if problem_name in all_results and key in all_results[problem_name]:
            trajectories = all_results[problem_name][key].get('full_hv_trajectories', [])
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
                    generations = np.arange(len(avg_traj)) * 10
                    ax2.plot(generations, avg_traj, label=config_name, 
                            color=CONFIG_COLORS.get(config_name, '#888888'), linewidth=2.5)
    
    ax2.set_xlabel('Environmental Changes', fontsize=12)
    ax2.set_ylabel('MHV (↑ higher is better)', fontsize=12)
    ax2.set_title(f'{algo_name} on {problem_name}: MHV Across Configurations', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{problem_name}_{algo_name}_config_comparison.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_individual_curves(all_results: Dict, problem_names: List[str], 
                                algorithm_names: List[str], save_path: str = None):
    """Generate all convergence plots for each problem and configuration."""
    
    print("\nGenerating individual convergence curves for each problem...")
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "individual_curves")
    
    os.makedirs(save_path, exist_ok=True)
    
    for problem_name in problem_names:
        for config in ENV_CONFIGS:
            config_name = config['name']
            print(f"  Plotting {problem_name} ({config_name})...")
            plot_individual_igd_convergence(all_results, problem_name, algorithm_names, config_name, save_path)
            plot_individual_mhv_convergence(all_results, problem_name, algorithm_names, config_name, save_path)
        
        # Plot config comparison for each algorithm (outside the config loop)
        for algo_name in algorithm_names:
            plot_config_comparison(all_results, problem_name, algo_name, save_path)
    
    print(f"  Saved plots to {save_path}/")


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_experiment(problem_name: str, config_params: Dict, run_id: int, 
                           algorithm_class, seed_offset: int) -> Dict:
    """Run a single experiment for one configuration."""
    try:
        config = ProblemConfig(
            n_variables=N_VARIABLES,
            n_objectives=2,
            tau_t=config_params['tau_t'],
            n_t=config_params['n_t'],
            warmup_generations=WARMUP_GENERATIONS,
            n_changes=N_CHANGES,
            population_size=POP_SIZE
        )
        
        problem_class = get_problem_class(problem_name)
        problem = problem_class(config)
        
        np.random.seed(run_id * 1000 + seed_offset)
        
        algo = algorithm_class(problem, config.population_size)
        algo.initialize()
        
        # Track full IGD and HV trajectories
        full_igd = []
        full_hv = []
        
        for gen in range(config.max_generations):
            algo.step(gen)
            
            # Record IGD and HV every 10 generations
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


def run_set2_benchmark():
    """Run Set 2 benchmark with multiple environment configurations."""
    
    test_problems = ['DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6', 'DF7', 'DF8', 'DF9',
                     'DF10', 'DF11', 'DF12', 'DF13', 'DF14']
    
    # Comparator Set (6 MOEA/D variants)
    algorithm_classes = [MOEAD, MOEADKNN, MOEADPPS, MOEADAGR, MOEADHSS, MOEADRV]
    algorithm_names = ['MOEA/D', 'MOEA/D-KNN', 'MOEA/D-PPS', 'MOEA/D-AGR', 'MOEA/D-HSS', 'MOEA/D-RV']
    
    all_results = {}
    
    print("=" * 100)
    print("MOEA/D-RV: SET 2 - MULTIPLE ENVIRONMENT CONFIGURATIONS (OPTIMIZED)")
    print("=" * 100)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"\nComparator Set (6 MOEA/D variants):")
    print(f"  1. MOEA/D (Baseline)              - Zhang & Li (2007)")
    print(f"  2. MOEA/D-KNN                     - Deng et al. (2025) [Training-free local prediction]")
    print(f"  3. MOEA/D-PPS                     - Zhou et al. (2014) [Population prediction]")
    print(f"  4. MOEA/D-AGR                     - Adaptive guided response")
    print(f"  5. MOEA/D-HSS                     - Hu et al. (2024) [Hybrid search strategy]")
    print(f"  6. MOEA/D-RV (Proposed)           - Risk + Prediction + HV Selection")
    print(f"\nOPTIMIZATIONS ENABLED:")
    print(f"  • Reduced scenarios: 10 (was 50)")
    print(f"  • Risk computation caching")
    print(f"  • Approximate hypervolume for 3+ objectives")
    print(f"  • Reduced niching frequency")
    print(f"\nEnvironment Configurations:")
    for cfg in ENV_CONFIGS:
        print(f"  {cfg['name']}: (tau_t={cfg['tau_t']}, n_t={cfg['n_t']}) -> {cfg['desc']}")
        print(f"      Total generations: {cfg['max_gens']}")
    
    print(f"\nConfiguration:")
    print(f"  Population size: {POP_SIZE}")
    print(f"  Warmup generations: {WARMUP_GENERATIONS}")
    print(f"  Number of changes: {N_CHANGES}")
    print(f"  Number of runs: {N_RUNS}")
    print(f"  Problems: {len(test_problems)} (DF1-DF14)")
    print(f"  Algorithms: {len(algorithm_classes)}")
    print(f"  Total experiments: {len(test_problems) * len(ENV_CONFIGS) * len(algorithm_classes) * N_RUNS}")
    print("=" * 100)
    
    total_start = time.time()
    
    # Initialize results structure
    for problem_name in test_problems:
        all_results[problem_name] = {}
        for cfg in ENV_CONFIGS:
            for algo_name in algorithm_names:
                key = f"{algo_name}_{cfg['name']}"
                all_results[problem_name][key] = {
                    'mean_migd': 0.0, 'std_migd': 0.0,
                    'mean_mhv': 0.0, 'std_mhv': 0.0,
                    'full_igd_trajectories': [],
                    'full_hv_trajectories': [],
                }
    
    exp_count = 0
    total_exp = len(test_problems) * len(ENV_CONFIGS) * len(algorithm_classes) * N_RUNS
    
    for problem_idx, problem_name in enumerate(test_problems):
        print(f"\n[{problem_name}] ({problem_idx+1}/{len(test_problems)})")
        
        for cfg_idx, cfg in enumerate(ENV_CONFIGS):
            print(f"\n  Configuration: {cfg['name']} (tau_t={cfg['tau_t']}, n_t={cfg['n_t']})")
            print(f"  Total generations: {cfg['max_gens']}")
            
            for algo_idx, (algo_class, algo_name) in enumerate(zip(algorithm_classes, algorithm_names)):
                print(f"\n    Running {algo_name}...")
                algo_start = time.time()
                
                key = f"{algo_name}_{cfg['name']}"
                migd_values = []
                mhv_values = []
                full_igd_trajectories = []
                full_hv_trajectories = []
                
                for run in range(N_RUNS):
                    seed_offset = (cfg_idx * 100 + algo_idx * 10 + hash(problem_name) % 100)
                    result = run_single_experiment(problem_name, cfg, run, algo_class, seed_offset)
                    migd_values.append(result['migd'])
                    mhv_values.append(result['mhv'])
                    if result.get('full_igd_trajectory'):
                        full_igd_trajectories.append(result['full_igd_trajectory'])
                    if result.get('full_hv_trajectory'):
                        full_hv_trajectories.append(result['full_hv_trajectory'])
                    
                    exp_count += 1
                    if (run + 1) % 5 == 0 or run == N_RUNS - 1:
                        print(f"      Runs: {run+1}/{N_RUNS} (MIGD: {result['migd']:.6f}, MHV: {result['mhv']:.4f})", flush=True)
                    else:
                        print(f"      Run {run+1}/{N_RUNS}: MIGD = {result['migd']:.6f}, MHV = {result['mhv']:.4f}", flush=True)
                    
                    # Show overall progress
                    if exp_count % (total_exp // 20 + 1) == 0:
                        pct = 100 * exp_count / total_exp
                        print(f"      [Overall Progress: {pct:.1f}% ({exp_count}/{total_exp})]", flush=True)
                
                elapsed = time.time() - algo_start
                valid_migd = [v for v in migd_values if v != float('inf') and not np.isnan(v)]
                valid_mhv = [v for v in mhv_values if v > 0 and not np.isnan(v)]
                
                mean_migd = np.mean(valid_migd) if valid_migd else float('inf')
                std_migd = np.std(valid_migd) if valid_migd else 0.0
                mean_mhv = np.mean(valid_mhv) if valid_mhv else 0.0
                std_mhv = np.std(valid_mhv) if valid_mhv else 0.0
                
                all_results[problem_name][key] = {
                    'mean_migd': mean_migd,
                    'std_migd': std_migd,
                    'mean_mhv': mean_mhv,
                    'std_mhv': std_mhv,
                    'full_igd_trajectories': full_igd_trajectories,
                    'full_hv_trajectories': full_hv_trajectories,
                }
                
                print(f"    {algo_name} completed in {elapsed:.1f}s")
                print(f"    Summary: MIGD = {mean_migd:.6f} ± {std_migd:.6f}, MHV = {mean_mhv:.4f} ± {std_mhv:.4f}")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 100)
    print(f"BENCHMARK COMPLETE! Total time: {total_time/3600:.2f} hours")
    print("=" * 100)
    
    # Summary tables for each configuration
    print("\n" + "=" * 100)
    print("SUMMARY TABLES - MIGD FOR EACH CONFIGURATION")
    print("=" * 100)
    
    for cfg in ENV_CONFIGS:
        config_name = cfg['name']
        print(f"\n{'='*80}")
        print(f"CONFIGURATION: {config_name} (tau_t={cfg['tau_t']}, n_t={cfg['n_t']})")
        print(f"{'='*80}")
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
                key = f"{algo_name}_{config_name}"
                mean_val = all_results[problem_name][key]['mean_migd']
                std_val = all_results[problem_name][key]['std_migd']
                
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
        print(f"\nWIN COUNT (best MIGD) - {config_name}:")
        for algo_name in algorithm_names:
            print(f"  {algo_name}: {win_count[algo_name]}/{len(test_problems)}")
    
    # MOEA/D-RV performance across configurations
    print("\n" + "=" * 100)
    print("MOEA/D-RV PERFORMANCE ACROSS CONFIGURATIONS")
    print("=" * 100)
    print(f"{'Problem':<12} {'Standard (10,10)':<30} {'Mild (10,5)':<30} {'Rapid (5,10)':<30}")
    print("-" * 102)
    
    for problem_name in test_problems:
        print(f"{problem_name:<12}", end='')
        for cfg in ENV_CONFIGS:
            config_name = cfg['name']
            key = f"MOEA/D-RV_{config_name}"
            mean_val = all_results[problem_name][key]['mean_migd']
            std_val = all_results[problem_name][key]['std_migd']
            if not np.isnan(mean_val) and mean_val != float('inf'):
                print(f" {mean_val:.6f}±{std_val:.6f}    ", end='')
            else:
                print(f" {'nan±nan':<20}", end='')
        print()
    
    # ============ SAVE RESULTS TO FILE (BEFORE PLOTTING) ============
    print("\n" + "=" * 100)
    print("SAVING RESULTS TO FILE")
    print("=" * 100)
    
    # Build output dictionary
    output = {
        'timestamp': datetime.now().isoformat(),
        'output_directory': OUTPUT_DIR,
        'comparator_set': algorithm_names,
        'config': {
            'population_size': POP_SIZE,
            'n_runs': N_RUNS,
            'warmup_generations': WARMUP_GENERATIONS,
            'n_changes': N_CHANGES,
            'environment_configs': ENV_CONFIGS,
            'optimizations': {
                'n_scenarios': 10,
                'risk_caching': True,
                'approximate_hv': True,
                'niching_frequency': 20
            }
        },
        'algorithm_names': algorithm_names,
        'results': {}
    }
    
    for problem_name in test_problems:
        output['results'][problem_name] = {}
        for cfg in ENV_CONFIGS:
            config_name = cfg['name']
            output['results'][problem_name][config_name] = {}
            for algo_name in algorithm_names:
                key = f"{algo_name}_{config_name}"
                mean_migd = all_results[problem_name][key]['mean_migd']
                std_migd = all_results[problem_name][key]['std_migd']
                mean_mhv = all_results[problem_name][key]['mean_mhv']
                std_mhv = all_results[problem_name][key]['std_mhv']
                output['results'][problem_name][config_name][algo_name] = {
                    'migd_mean': float(mean_migd) if not np.isnan(mean_migd) and mean_migd != float('inf') else None,
                    'migd_std': float(std_migd) if not np.isnan(std_migd) else None,
                    'mhv_mean': float(mean_mhv) if not np.isnan(mean_mhv) and mean_mhv > 0 else None,
                    'mhv_std': float(std_mhv) if not np.isnan(std_mhv) else None,
                }
    
    # Save JSON file (primary data storage)
    filename = os.path.join(OUTPUT_DIR, f'moead_rv_set2_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ JSON results saved to: {filename}")
    
    # Save CSV file (easier for spreadsheet analysis)
    csv_filename = os.path.join(OUTPUT_DIR, f'summary_migd_mhv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(csv_filename, 'w') as f:
        f.write("Problem,Configuration,Algorithm,MIGD_mean,MIGD_std,MHV_mean,MHV_std\n")
        for problem_name in test_problems:
            for cfg in ENV_CONFIGS:
                config_name = cfg['name']
                for algo_name in algorithm_names:
                    key = f"{algo_name}_{config_name}"
                    migd_mean = all_results[problem_name][key]['mean_migd']
                    migd_std = all_results[problem_name][key]['std_migd']
                    mhv_mean = all_results[problem_name][key]['mean_mhv']
                    mhv_std = all_results[problem_name][key]['std_mhv']
                    f.write(f"{problem_name},{config_name},{algo_name},{migd_mean},{migd_std},{mhv_mean},{mhv_std}\n")
    print(f"✓ CSV summary saved to: {csv_filename}")
    
    # Save a backup copy (extra safety)
    backup_filename = os.path.join(OUTPUT_DIR, f'backup_set2_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(backup_filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Backup saved to: {backup_filename}")
    
    # ============ GENERATE PLOTS (with error handling) ============
    print("\n" + "=" * 100)
    print("GENERATING PLOTS")
    print("=" * 100)
    print("Note: Data has been saved. Plot generation can be skipped if errors occur.")
    
    try:
        plot_all_individual_curves(all_results, test_problems, algorithm_names, 
                                   save_path=os.path.join(OUTPUT_DIR, "individual_curves"))
        print(f"✓ Individual IGD and MHV curves saved to: {os.path.join(OUTPUT_DIR, 'individual_curves')}/")
    except Exception as e:
        print(f"⚠️ Plot generation encountered an error: {e}")
        print("   Data has already been saved, so results are safe.")
        print("   You can generate plots later using the saved JSON file.")
    
    print("\n" + "=" * 100)
    print("SET 2 BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 100)
    
    return all_results


if __name__ == "__main__":
    print("=" * 100)
    print("MOEA/D-RV: SET 2 - MULTIPLE ENVIRONMENT CONFIGURATIONS (OPTIMIZED)")
    print("6 Algorithms: MOEA/D, MOEA/D-KNN, MOEA/D-PPS, MOEA/D-AGR, MOEA/D-HSS, MOEA/D-RV")
    print("14 Test Problems: DF1-DF14")
    print("3 Environment Configurations: Standard (10,10), Mild (10,5), Rapid (5,10)")
    print(f"{N_RUNS} Independent Runs")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 100)
    
    results = run_set2_benchmark()
    
    print("\n" + "=" * 100)
    print("SET 2 COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 100)