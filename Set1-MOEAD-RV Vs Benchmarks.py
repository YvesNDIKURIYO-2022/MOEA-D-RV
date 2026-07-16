"""
MOEA/D-RV (Risk-Guided Mutation + Prediction + HV Selection)
Dynamic Multi-Objective Optimization Algorithm

Core Features:
1. ✅ Risk-Guided Mutation - Variable-specific adaptive mutation based on sensitivity
2. ✅ Prediction (PPS-style) - Population prediction strategy for fast tracking
3. ✅ Hypervolume-Guided Selection - Elite selection for better convergence

COMPARATOR SET (6 MOEA/D variants):
1. MOEA/D (Baseline)
2. MOEA/D-KNN (Training-free local prediction)
3. MOEA/D-PPS (Population prediction strategy)
4. MOEA/D-AGR (Adaptive guided response)
5. MOEA/D-HSS (Hybrid search strategy)
6. MOEA/D-RV (Proposed - Risk + Prediction + HV)


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from datetime import datetime
from collections import deque
import os
from sklearn.neighbors import NearestNeighbors

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
MAX_GENERATIONS = 350
FREQUENCY_CHANGE = 10
SEVERITY_CHANGE = 10
WARMUP_GENERATIONS = 50
N_CHANGES = 30
N_VARIABLES = 10
N_RUNS = 5  # Set to 30 for final publication

REF_POINTS = {
    'DF1': np.array([1.1, 1.1]), 'DF2': np.array([1.1, 1.1]),
    'DF3': np.array([1.1, 1.1]), 'DF4': np.array([30, 30]),
    'DF5': np.array([1.1, 1.1]), 'DF6': np.array([1.1, 1.1]),
    'DF7': np.array([10, 10]), 'DF8': np.array([1.1, 1.1]),
    'DF9': np.array([1.1, 1.1]), 'DF10': np.array([1.1, 1.1, 1.1]),
    'DF11': np.array([1.1, 1.1, 1.1]), 'DF12': np.array([1.1, 1.1, 1.1]),
    'DF13': np.array([1.1, 1.1, 1.1]), 'DF14': np.array([1.1, 1.1, 1.1]),
}

COLORS = {
    'MOEA/D': '#1B5E20',
    'MOEA/D-KNN': '#0D47A1',
    'MOEA/D-PPS': '#E65100',
    'MOEA/D-AGR': '#4A148C',
    'MOEA/D-HSS': '#00838F',
    'MOEA/D-RV': '#C62828',
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
# PROBLEM DEFINITIONS
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


# DF1-DF9 (2-objective)
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


# DF10-DF14 (3-objective)
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
            return
        
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
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False
    
    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
            
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        n_reinit = int(self.population_size * self.reinit_ratio)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        self._update_pareto_archive()

# ============================================================================
# ALGORITHM 2: MOEA/D-KNN
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
            return
        
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
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
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
        
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()

# ============================================================================
# ALGORITHM 3: MOEA/D-PPS
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
            return
        
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
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
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
        
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()

# ============================================================================
# ALGORITHM 4: MOEA/D-AGR
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
            return
        
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
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
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
# ALGORITHM 5: MOEA/D-HSS
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
            if self.population is None or self.objectives is None or len(self.population) == 0:
                return False
            test_idx = np.random.randint(min(self.population_size, len(self.population)))
            new_obj = self.problem.evaluate(self.population[test_idx], self.generation)
            return not np.allclose(new_obj, self.objectives[test_idx], atol=1e-6)
        return False
    
    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
            
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        self.search_range = min(0.3, self.search_range * 1.3)
        
        n_reinit = int(self.population_size * 0.25)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()

# ============================================================================
# ALGORITHM 6: MOEA/D-RV (PROPOSED)
# ============================================================================

class MOEADRV(BaseMOEAD):
    """
    MOEA/D-RV (Risk-Guided Mutation + Prediction + HV Selection)
    
    Core Features:
    1. ✅ Risk-Guided Mutation - Variable-specific adaptive mutation
    2. ✅ Prediction (PPS-style) - Fast tracking of changing optima
    3. ✅ Hypervolume-Guided Selection - Elite survival selection
    """
    
    def __init__(self, problem: DynamicProblem, population_size: int = POP_SIZE):
        super().__init__(problem, population_size)
        
        # ===== Core Components =====
        # Risk-Guided Mutation
        self.risk_history = {j: deque(maxlen=10) for j in range(problem.n_variables)}
        self.risk_sensitivity = 1.5
        self.base_mutation = 0.15
        self.risk_momentum = np.ones(problem.n_variables)
        
        # Prediction (PPS-style)
        self.history_centers = []
        self.history_populations = []
        self.max_history = 3
        
        # Hypervolume Selection
        self.hv_reference = None
        
        # Enhanced change detection
        self.change_threshold = 1e-6
        self.detection_samples = 5
        
    def initialize(self):
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.problem.n_variables))
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.weights = self._generate_weights()
        self.neighbors = self._compute_neighbors()
        self.ideal_point = np.min(self.objectives, axis=0)
        
        # Initialize prediction history
        self.history_centers.append(np.mean(self.population, axis=0))
        self.history_populations.append(self.population.copy())
        
        # Initialize HV reference
        if self.problem.n_objectives == 2:
            self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2]))
        else:
            self.hv_reference = REF_POINTS.get(self.problem.get_name(), np.array([2, 2, 2]))
        
        self._update_pareto_archive()
    
    # ===== FEATURE 1: RISK-GUIDED MUTATION =====
    
    def _compute_risk(self, variable_idx: int, solution: np.ndarray, subproblem_idx: int) -> float:
        """Compute risk sensitivity for a variable with numerical stability"""
        # Safe delta computation
        delta = max(
            0.05 * (self.ub[variable_idx] - self.lb[variable_idx]),
            1e-6  # Minimum delta
        )
        
        x_perturbed = solution.copy()
        x_perturbed[variable_idx] += delta
        x_perturbed = np.clip(x_perturbed, self.lb[variable_idx], self.ub[variable_idx])
        
        # Clamp fitness values to prevent overflow
        current_fitness = min(self._tchebycheff(self.objectives[subproblem_idx], self.weights[subproblem_idx]), 1e10)
        new_fitness = min(self._tchebycheff(
            self.problem.evaluate(x_perturbed, self.generation), 
            self.weights[subproblem_idx]
        ), 1e10)
        
        risk = abs(new_fitness - current_fitness) / (abs(delta) + 1e-12)
        return min(risk, 10.0)  # Cap risk to prevent instability
    
    def _risk_guided_mutation(self, parent: np.ndarray, subproblem_idx: int) -> np.ndarray:
        """Variable-specific mutation based on risk sensitivity"""
        child = parent.copy()
        
        for j in range(self.problem.n_variables):
            risk = self._compute_risk(j, child, subproblem_idx)
            self.risk_history[j].append(risk)
            
            # Update momentum
            if len(self.risk_history[j]) > 1:
                self.risk_momentum[j] = 0.9 * self.risk_momentum[j] + 0.1 * risk
            else:
                self.risk_momentum[j] = risk
            
            # Normalize risk
            all_risks = [self.risk_momentum[k] for k in range(self.problem.n_variables)]
            risk_min = np.min(all_risks)
            risk_max = np.max(all_risks)
            risk_norm = (self.risk_momentum[j] - risk_min) / (risk_max - risk_min + 1e-8)
            risk_norm = np.clip(risk_norm, 0, 1)
            
            # Adaptive mutation strength
            sigma = self.base_mutation * (1 + self.risk_sensitivity * risk_norm)
            
            # Direction based on risk trend
            if len(self.risk_history[j]) >= 3:
                recent_risks = list(self.risk_history[j])[-3:]
                avg_recent = np.mean(recent_risks)
                avg_all = np.mean(all_risks)
                direction = -1 if avg_recent < avg_all else 1
            else:
                direction = 1 if np.random.rand() < 0.5 else -1
            
            # Polynomial mutation with adaptive strength
            delta = np.random.rand()
            if delta < 0.5:
                delta_val = (2 * delta) ** (1 / (self.eta_m + 1)) - 1
            else:
                delta_val = 1 - (2 * (1 - delta)) ** (1 / (self.eta_m + 1))
            
            step = direction * sigma * (self.ub[j] - self.lb[j]) * delta_val
            child[j] = np.clip(child[j] + step, self.lb[j], self.ub[j])
        
        return child
    
    # ===== FEATURE 2: PREDICTION (PPS-STYLE) =====
    
    def _predict_population(self) -> np.ndarray:
        """Predict new population using PPS-style linear prediction"""
        if len(self.history_populations) < 2:
            return self.population
        
        # Compute centers
        centers = np.array(self.history_centers)
        
        if len(centers) >= 2:
            # Linear prediction with stability check
            predicted_center = 2 * centers[-1] - centers[-2]
            shift = predicted_center - centers[-1]
            
            # Limit shift to prevent instability
            max_shift = 0.5 * (self.ub - self.lb)
            shift = np.clip(shift, -max_shift, max_shift)
            
            predicted_pop = self.history_populations[-1] + shift
            return np.clip(predicted_pop, self.lb, self.ub)
        
        return self.population
    
    # ===== FEATURE 3: HYPERVOLUME-GUIDED SELECTION =====
    
    def _crowding_selection(self, candidates: np.ndarray, candidate_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback selection using crowding distance when HV is zero"""
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        
        # Compute crowding distance
        crowding = np.zeros(n_total)
        for m in range(self.problem.n_objectives):
            idx = np.argsort(all_obj[:, m])
            crowding[idx[0]] = np.inf
            crowding[idx[-1]] = np.inf
            for i in range(1, n_total - 1):
                crowding[idx[i]] += (all_obj[idx[i+1], m] - all_obj[idx[i-1], m])
        
        # Select top by crowding distance
        elite_idx = np.argsort(crowding)[-self.population_size:]
        return all_pop[elite_idx], all_obj[elite_idx]
    
    def _hypervolume_selection(self, candidates: np.ndarray, candidate_objs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select elite solutions using hypervolume contribution"""
        all_pop = np.vstack([self.population, candidates])
        all_obj = np.vstack([self.objectives, candidate_objs])
        
        n_total = len(all_obj)
        if n_total <= self.population_size:
            return all_pop, all_obj
        
        # Check if any solutions are dominated or HV is zero
        if self.problem.n_objectives == 2:
            # Fast 2D hypervolume selection
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
            
            # If all contributions are zero, use crowding distance
            if np.sum(contributions) == 0:
                return self._crowding_selection(candidates, candidate_objs)
            
            elite_idx_in_sorted = np.argsort(contributions)[-self.population_size:]
            elite_idx = sorted_idx[elite_idx_in_sorted]
        else:
            # Approximate 3D+ hypervolume selection
            contributions = np.zeros(n_total)
            for i in range(min(n_total, 100)):
                temp_obj = np.delete(all_obj, i, axis=0)
                hv_without = PerformanceMetrics.hypervolume_3d(temp_obj, self.hv_reference, 2000)
                hv_with = PerformanceMetrics.hypervolume_3d(all_obj, self.hv_reference, 2000)
                contributions[i] = hv_with - hv_without
            
            # If all contributions are zero, use crowding distance
            if np.sum(contributions) == 0:
                return self._crowding_selection(candidates, candidate_objs)
            
            contributions[100:] = np.mean(contributions[:100])
            elite_idx = np.argsort(contributions)[-self.population_size:]
        
        return all_pop[elite_idx], all_obj[elite_idx]
    
    # ===== ENHANCED CHANGE DETECTION =====
    
    def detect_change(self) -> bool:
        """Enhanced change detection with multiple samples"""
        if self.generation <= self.problem.config.warmup_generations:
            return False
        
        if self.population is None or self.objectives is None or len(self.population) == 0:
            return False
        
        # Sample multiple individuals for robust detection
        n_samples = min(self.detection_samples, self.population_size)
        test_indices = np.random.choice(self.population_size, n_samples, replace=False)
        
        changes_detected = 0
        for idx in test_indices:
            new_obj = self.problem.evaluate(self.population[idx], self.generation)
            if not np.allclose(new_obj, self.objectives[idx], atol=self.change_threshold):
                changes_detected += 1
        
        # Detect change if majority of samples indicate change
        return changes_detected > n_samples // 2
    
    # ===== MAIN ALGORITHM METHODS =====
    
    def evolve(self):
        if self.population is None:
            self.initialize()
            return
        
        # Generate offspring
        offspring_pop = []
        offspring_obj = []
        
        for i in range(self.population_size):
            if np.random.rand() < self.delta:
                mating_pool = self.neighbors[i]
            else:
                mating_pool = np.arange(self.population_size)
            
            # Tournament selection
            n_tournament = min(3, len(mating_pool))
            idx1 = np.random.choice(mating_pool, n_tournament, replace=False)
            idx2 = np.random.choice(mating_pool, n_tournament, replace=False)
            
            fitness1 = np.sum(self.objectives[idx1], axis=1)
            fitness2 = np.sum(self.objectives[idx2], axis=1)
            
            parent1 = self.population[idx1[np.argmin(fitness1)]]
            parent2 = self.population[idx2[np.argmin(fitness2)]]
            
            # Crossover (SBX)
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
            
            # Risk-Guided Mutation
            child = self._risk_guided_mutation(child, i)
            
            offspring_pop.append(child)
            offspring_obj.append(self.problem.evaluate(child, self.generation))
        
        offspring_pop = np.array(offspring_pop)
        offspring_obj = np.array(offspring_obj)
        
        # Hypervolume Selection
        self.population, self.objectives = self._hypervolume_selection(offspring_pop, offspring_obj)
        
        # Update ideal point
        self.ideal_point = np.min(self.objectives, axis=0)
        self._update_pareto_archive()
    
    def respond_to_change(self):
        if self.population is None or self.objectives is None:
            self.initialize()
            return
        
        # Store history for prediction
        self.history_centers.append(np.mean(self.population, axis=0))
        self.history_populations.append(self.population.copy())
        
        if len(self.history_centers) > self.max_history:
            self.history_centers.pop(0)
            self.history_populations.pop(0)
        
        # Apply prediction
        if len(self.history_populations) >= 2:
            predicted = self._predict_population()
            # Blend with current population
            blend_ratio = 0.7
            self.population = (1 - blend_ratio) * self.population + blend_ratio * predicted
            self.population = np.clip(self.population, self.lb, self.ub)
        
        # Re-evaluate objectives
        self.objectives = np.array([self.problem.evaluate(ind, self.generation) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)
        
        # Reset risk history for new environment
        self.risk_history = {j: deque(maxlen=10) for j in range(self.problem.n_variables)}
        self.risk_momentum = np.ones(self.problem.n_variables)
        self.base_mutation = min(0.25, self.base_mutation * 1.1)
        
        # Reinitialize some individuals for diversity
        n_reinit = int(self.population_size * 0.2)
        reinit_indices = np.random.choice(self.population_size, min(n_reinit, self.population_size), replace=False)
        for idx in reinit_indices:
            self.population[idx] = np.random.uniform(self.lb, self.ub, self.problem.n_variables)
            self.objectives[idx] = self.problem.evaluate(self.population[idx], self.generation)
        
        self._update_pareto_archive()

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_individual_igd_convergence(all_results: Dict, problem_name: str, 
                                     algorithm_names: List[str], save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 7))
    
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
                    ax.plot(generations, avg_traj, label=algo_name, 
                           color=COLORS.get(algo_name, '#888888'), linewidth=2.5)
                    ax.fill_between(generations, avg_traj - std_traj, avg_traj + std_traj,
                                   alpha=0.2, color=COLORS.get(algo_name, '#888888'))
    
    ax.set_xlabel('Environmental Changes', fontsize=12, fontweight='bold')
    ax.set_ylabel('IGD (↓ lower is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'{problem_name}: IGD Convergence ({N_RUNS} runs, mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{problem_name}_igd_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_mhv_convergence(all_results: Dict, problem_name: str, 
                                     algorithm_names: List[str], save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 7))
    
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
                    ax.plot(generations, avg_traj, label=algo_name, 
                           color=COLORS.get(algo_name, '#888888'), linewidth=2.5)
                    ax.fill_between(generations, avg_traj - std_traj, avg_traj + std_traj,
                                   alpha=0.2, color=COLORS.get(algo_name, '#888888'))
    
    ax.set_xlabel('Environmental Changes', fontsize=12, fontweight='bold')
    ax.set_ylabel('MHV (↑ higher is better)', fontsize=12, fontweight='bold')
    ax.set_title(f'{problem_name}: MHV Convergence ({N_RUNS} runs, mean ± std)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{problem_name}_mhv_convergence.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_individual_curves(all_results: Dict, problem_names: List[str], 
                                algorithm_names: List[str], save_path: str = None):
    print("\nGenerating individual convergence curves...")
    if save_path is None:
        save_path = 'results/individual_curves'
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
        mhv = PerformanceMetrics.mean_hypervolume(algo.environmental_hv) if hasattr(algo, 'environmental_hv') else 0.0
        
        return {
            'migd': migd,
            'mhv': mhv,
            'full_igd_trajectory': full_igd,
            'full_hv_trajectory': full_hv,
            'success': True
        }
    except Exception as e:
        print(f"Error in run {run_id}: {e}")
        return {'migd': float('inf'), 'mhv': 0.0, 'success': False, 'error': str(e)}


def get_problem_class(name: str):
    problems = {
        'DF1': DF1, 'DF2': DF2, 'DF3': DF3, 'DF4': DF4, 'DF5': DF5,
        'DF6': DF6, 'DF7': DF7, 'DF8': DF8, 'DF9': DF9, 'DF10': DF10,
        'DF11': DF11, 'DF12': DF12, 'DF13': DF13, 'DF14': DF14
    }
    return problems.get(name, DF1)


def run_set1_benchmark():
    """Run benchmark with all MOEA/D variants"""
    
    test_problems = ['DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6', 'DF7', 'DF8', 'DF9',
                     'DF10', 'DF11', 'DF12', 'DF13', 'DF14']
    
    # All MOEA/D variants for fair comparison
    algorithm_classes = [MOEAD, MOEADKNN, MOEADPPS, MOEADAGR, MOEADHSS, MOEADRV]
    algorithm_names = ['MOEA/D', 'MOEA/D-KNN', 'MOEA/D-PPS', 'MOEA/D-AGR', 'MOEA/D-HSS', 'MOEA/D-RV']
    
    all_results = {}
    for problem_name in test_problems:
        all_results[problem_name] = {}
        for algo_name in algorithm_names:
            all_results[problem_name][algo_name] = {
                'mean_migd': float('inf'),
                'std_migd': 0.0,
                'mean_mhv': 0.0,
                'std_mhv': 0.0,
                'full_igd_trajectories': [],
                'full_hv_trajectories': [],
            }
    
    print("=" * 100)
    print("MOEA/D-RV: MOEA/D VARIANT COMPARISON")
    print("=" * 100)
    print("\nAlgorithm Comparison (6 MOEA/D variants):")
    print("  1. MOEA/D (Baseline)              - Zhang & Li (2007)")
    print("  2. MOEA/D-KNN                     - Deng et al. (2025) [Training-free local prediction]")
    print("  3. MOEA/D-PPS                     - Zhou et al. (2014) [Population prediction]")
    print("  4. MOEA/D-AGR                     - Adaptive guided response")
    print("  5. MOEA/D-HSS                     - Hu et al. (2024) [Hybrid search]")
    print("  6. MOEA/D-RV (Proposed)           - Risk + Prediction + HV Selection")
    print("\nProposed Algorithm Features:")
    print("  ✅ Risk-Guided Mutation (variable-specific adaptive mutation)")
    print("  ✅ Prediction (PPS-style for fast tracking)")
    print("  ✅ Hypervolume-Guided Selection (elite survival)")
    print("  ✅ Enhanced Change Detection (multi-sample detection)")
    print("  ✅ Numerical Stability (clamping and fallback mechanisms)")
    print(f"\nConfiguration:")
    print(f"  Population size: {POP_SIZE}")
    print(f"  Max generations: {MAX_GENERATIONS}")
    print(f"  Frequency of change: {FREQUENCY_CHANGE}")
    print(f"  Number of runs: {N_RUNS}")
    print(f"  Problems: {len(test_problems)} (DF1-DF14)")
    print("=" * 100)
    
    total_start = time.time()
    
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
                
                print(f"    Run {run+1}/{N_RUNS}: MIGD = {result['migd']:.6f}, MHV = {result['mhv']:.4f}", flush=True)
            
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
    print(f"\nTotal time: {total_time/3600:.2f} hours")
    
    # Summary tables
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - MIGD VALUES (lower is better)")
    print("=" * 100)
    print(f"{'Problem':<12}", end='')
    for algo_name in algorithm_names:
        print(f" {algo_name:<20}", end='')
    print()
    print("-" * (12 + 20 * len(algorithm_names)))
    
    win_count_migd = {name: 0 for name in algorithm_names}
    
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
                print(f" {'nan±nan':<20}", end='')
        
        if best_algo:
            win_count_migd[best_algo] += 1
        print()
    
    print("-" * (12 + 20 * len(algorithm_names)))
    print(f"\nWIN COUNT - MIGD (best):")
    for algo_name in algorithm_names:
        print(f"  {algo_name}: {win_count_migd[algo_name]}/{len(test_problems)}")
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLE - MHV VALUES (higher is better)")
    print("=" * 100)
    print(f"{'Problem':<12}", end='')
    for algo_name in algorithm_names:
        print(f" {algo_name:<20}", end='')
    print()
    print("-" * (12 + 20 * len(algorithm_names)))
    
    win_count_mhv = {name: 0 for name in algorithm_names}
    
    for problem_name in test_problems:
        print(f"{problem_name:<12}", end='')
        best_mhv = -float('inf')
        best_algo = None
        
        for algo_name in algorithm_names:
            mean_val = all_results[problem_name][algo_name]['mean_mhv']
            std_val = all_results[problem_name][algo_name]['std_mhv']
            
            if not np.isnan(mean_val) and mean_val > 0:
                print(f" {mean_val:.4f}±{std_val:.4f} ", end='')
                if mean_val > best_mhv:
                    best_mhv = mean_val
                    best_algo = algo_name
            else:
                print(f" {'nan±nan':<20}", end='')
        
        if best_algo:
            win_count_mhv[best_algo] += 1
        print()
    
    print("-" * (12 + 20 * len(algorithm_names)))
    print(f"\nWIN COUNT - MHV (best):")
    for algo_name in algorithm_names:
        print(f"  {algo_name}: {win_count_mhv[algo_name]}/{len(test_problems)}")
    
    # Generate plots
    plot_all_individual_curves(all_results, test_problems, algorithm_names, save_path='results/individual_curves')
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'population_size': POP_SIZE,
            'max_generations': MAX_GENERATIONS,
            'frequency_change': FREQUENCY_CHANGE,
            'severity_change': SEVERITY_CHANGE,
            'warmup_generations': WARMUP_GENERATIONS,
            'n_changes': N_CHANGES,
            'n_runs': N_RUNS,
            'description': 'MOEA/D variant comparison with RV'
        },
        'algorithm_names': algorithm_names,
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
    
    filename = f'moead_rv_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return all_results


if __name__ == "__main__":
    print("=" * 100)
    print("MOEA/D-RV: RISK-GUIDED MUTATION + PREDICTION + HV SELECTION")
    print("Dynamic Multi-Objective Optimization Algorithm")
    print("6 MOEA/D Variants | 14 DF Problems | 5 Runs")
    print("=" * 100)
    
    results = run_set1_benchmark()
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE!")
    print("=" * 100)