"""
微分方程动态分类模型
实现二阶常微分方程组、龙格-库塔法求解和分类边界自适应调整
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ODEParameters:
    """二阶常微分方程参数定义"""
    alpha: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    beta: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.1]))
    delta: np.ndarray = field(default_factory=lambda: np.array([0.05]))
    epsilon: float = 0.01
    zeta: float = 0.001
    
    damping: float = 0.1
    stiffness: float = 1.0
    coupling_strength: float = 0.5
    
    def validate(self) -> bool:
        if np.any(self.alpha < 0) or np.any(self.beta < 0):
            return False
        if np.any(self.gamma < 0) or np.any(self.delta < 0):
            return False
        if self.epsilon < 0 or self.zeta < 0:
            return False
        if self.damping < 0 or self.stiffness <= 0:
            return False
        return True


@dataclass
class AdaptiveStepConfig:
    """自适应步长配置"""
    initial_step: float = 0.01
    min_step: float = 1e-6
    max_step: float = 0.1
    tolerance: float = 1e-6
    safety_factor: float = 0.9
    max_iterations: int = 10000


class BoundaryAdjustmentType(Enum):
    """边界调整类型"""
    THRESHOLD_TRIGGER = "threshold_trigger"
    GRADIENT_BASED = "gradient_based"
    HYBRID = "hybrid"


@dataclass
class ClassificationBoundary:
    """分类边界"""
    center: np.ndarray
    radius: np.ndarray
    normal_vectors: np.ndarray
    last_adjustment: float = 0.0
    adjustment_history: List[float] = field(default_factory=list)


class SecondOrderODESystem:
    """
    二阶常微分方程组
    特征维度变化率方程: dx/dt = v
    加速度方程: dv/dt = f(x, v, t)
    """
    
    def __init__(self, n_dimensions: int, params: Optional[ODEParameters] = None):
        self.n_dimensions = n_dimensions
        self.params = params or ODEParameters()
        
        self._init_parameters()
        
        self.state: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None
        self.time = 0.0
        
        self._state_history: List[Tuple[float, np.ndarray, np.ndarray]] = []
        
    def _init_parameters(self):
        n = self.n_dimensions
        
        if len(self.params.alpha) != n:
            self.params.alpha = np.ones(n) * self.params.alpha[0] if len(self.params.alpha) == 1 else np.ones(n)
        if len(self.params.beta) != n:
            self.params.beta = np.ones(n) * self.params.beta[0] if len(self.params.beta) == 1 else np.ones(n) * 0.5
        if len(self.params.gamma) != n:
            self.params.gamma = np.ones(n) * self.params.gamma[0] if len(self.params.gamma) == 1 else np.ones(n) * 0.1
        if len(self.params.delta) != n:
            self.params.delta = np.ones(n) * self.params.delta[0] if len(self.params.delta) == 1 else np.ones(n) * 0.05
            
        self.coupling_matrix = self._build_coupling_matrix()
        
    def _build_coupling_matrix(self) -> np.ndarray:
        n = self.n_dimensions
        matrix = np.eye(n) * (-2.0)
        
        for i in range(n - 1):
            matrix[i, i + 1] = 1.0
            matrix[i + 1, i] = 1.0
            
        if n > 2:
            matrix[0, n - 1] = 0.5
            matrix[n - 1, 0] = 0.5
            
        return matrix * self.params.coupling_strength
    
    def initialize_state(self, initial_features: np.ndarray, initial_velocity: Optional[np.ndarray] = None):
        if len(initial_features) != self.n_dimensions:
            raise ValueError(f"Initial features dimension {len(initial_features)} does not match system dimension {self.n_dimensions}")
            
        self.state = initial_features.copy().astype(np.float64)
        
        if initial_velocity is not None:
            self.velocity = initial_velocity.copy().astype(np.float64)
        else:
            self.velocity = np.zeros(self.n_dimensions, dtype=np.float64)
            
        self.time = 0.0
        self._state_history = [(0.0, self.state.copy(), self.velocity.copy())]
        
    def compute_acceleration(self, state: np.ndarray, velocity: np.ndarray, 
                            external_force: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算加速度 (二阶导数)
        方程: d²x/dt² = -damping*v - stiffness*K*x - α*v|v| - β*x|x| - γ*sign(v)*v² - δ*tanh(x)*||x||
        """
        n = self.n_dimensions
        
        acceleration = np.zeros(n, dtype=np.float64)
        
        acceleration -= self.params.damping * velocity
        
        acceleration -= self.params.stiffness * self.coupling_matrix @ state
        
        acceleration -= self.params.alpha * velocity * np.abs(velocity)
        
        acceleration -= self.params.beta * state * np.abs(state)
        
        acceleration -= self.params.gamma * np.sign(velocity) * velocity**2
        
        acceleration -= self.params.delta * np.tanh(state) * np.linalg.norm(state)
        
        if external_force is not None:
            acceleration += external_force
            
        return acceleration
    
    def compute_feature_change_rate(self, state: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """计算特征变化率 (一阶导数)"""
        return velocity.copy()
    
    def compute_second_derivative(self, state: np.ndarray, velocity: np.ndarray,
                                  external_force: Optional[np.ndarray] = None) -> np.ndarray:
        """计算二阶导数"""
        return self.compute_acceleration(state, velocity, external_force)
    
    def get_state_derivatives(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.state is None or self.velocity is None:
            raise RuntimeError("State not initialized. Call initialize_state first.")
        return self.state.copy(), self.velocity.copy()


class RungeKuttaSolver:
    """
    龙格-库塔法迭代求解器
    实现RK4方法和步长自适应
    """
    
    def __init__(self, ode_system: SecondOrderODESystem, config: Optional[AdaptiveStepConfig] = None):
        self.ode_system = ode_system
        self.config = config or AdaptiveStepConfig()
        
        self.current_step = self.config.initial_step
        self.step_history: List[Tuple[float, float]] = []
        
    def _rk4_step(self, state: np.ndarray, velocity: np.ndarray, 
                  dt: float, external_force: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """RK4单步积分"""
        k1_v = self.ode_system.compute_acceleration(state, velocity, external_force)
        k1_x = velocity
        
        k2_v = self.ode_system.compute_acceleration(
            state + 0.5 * dt * k1_x,
            velocity + 0.5 * dt * k1_v,
            external_force
        )
        k2_x = velocity + 0.5 * dt * k1_v
        
        k3_v = self.ode_system.compute_acceleration(
            state + 0.5 * dt * k2_x,
            velocity + 0.5 * dt * k2_v,
            external_force
        )
        k3_x = velocity + 0.5 * dt * k2_v
        
        k4_v = self.ode_system.compute_acceleration(
            state + dt * k3_x,
            velocity + dt * k3_v,
            external_force
        )
        k4_x = velocity + dt * k3_v
        
        new_state = state + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        new_velocity = velocity + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return new_state, new_velocity
    
    def _estimate_error(self, state: np.ndarray, velocity: np.ndarray,
                       dt: float, external_force: Optional[np.ndarray] = None) -> float:
        """估计局部截断误差"""
        state_full, vel_full = self._rk4_step(state, velocity, dt, external_force)
        
        state_half, vel_half = self._rk4_step(state, velocity, dt / 2, external_force)
        state_half2, vel_half2 = self._rk4_step(state_half, vel_half, dt / 2, external_force)
        
        state_error = np.linalg.norm(state_full - state_half2)
        vel_error = np.linalg.norm(vel_full - vel_half2)
        return max(state_error, vel_error)
    
    def _adapt_step_size(self, error: float) -> float:
        """自适应调整步长"""
        if error < 1e-15:
            return min(self.current_step * 2.0, self.config.max_step)
            
        ratio = self.config.tolerance / error
        new_step = self.config.safety_factor * self.current_step * (ratio ** 0.2)
        
        new_step = max(self.config.min_step, min(new_step, self.config.max_step))
        
        return new_step
    
    def solve(self, target_time: float, external_force: Optional[np.ndarray] = None,
              force_func: Optional[Callable[[float, np.ndarray, np.ndarray], np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """求解到目标时间"""
        if self.ode_system.state is None or self.ode_system.velocity is None:
            raise RuntimeError("ODE system not initialized")
            
        state = self.ode_system.state.copy()
        velocity = self.ode_system.velocity.copy()
        current_time = self.ode_system.time
        
        iterations = 0
        while current_time < target_time and iterations < self.config.max_iterations:
            remaining = target_time - current_time
            dt = min(self.current_step, remaining)
            
            current_force = external_force
            if force_func is not None:
                current_force = force_func(current_time, state, velocity)
            
            error = self._estimate_error(state, velocity, dt, current_force)
            
            if error > self.config.tolerance * 10:
                self.current_step = self._adapt_step_size(error)
                continue
            
            state, velocity = self._rk4_step(state, velocity, dt, current_force)
            current_time += dt
            
            self.current_step = self._adapt_step_size(error)
            
            iterations += 1
            
        self.ode_system.state = state
        self.ode_system.velocity = velocity
        self.ode_system.time = current_time
        self.ode_system._state_history.append((current_time, state.copy(), velocity.copy()))
        
        self.step_history.append((current_time, self.current_step))
        
        return state, velocity, current_time
    
    def solve_to_steady_state(self, convergence_threshold: float = 1e-8,
                              max_time: float = 100.0,
                              external_force: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """求解到稳态"""
        if self.ode_system.state is None or self.ode_system.velocity is None:
            raise RuntimeError("ODE system not initialized")
            
        state = self.ode_system.state.copy()
        velocity = self.ode_system.velocity.copy()
        current_time = self.ode_system.time
        
        prev_state = state.copy()
        iterations = 0
        converged = False
        
        while current_time < max_time and iterations < self.config.max_iterations:
            state, velocity, current_time = self.solve(
                current_time + self.current_step * 10,
                external_force
            )
            
            state_change = np.linalg.norm(state - prev_state)
            velocity_norm = np.linalg.norm(velocity)
            
            if state_change < convergence_threshold and velocity_norm < convergence_threshold:
                converged = True
                break
                
            prev_state = state.copy()
            iterations += 1
            
        return state, velocity, current_time, converged


class AdaptiveBoundaryController:
    """
    自适应边界控制器
    实现特征变化率阈值触发分类边界自适应调整
    """
    
    def __init__(self, ode_system: SecondOrderODESystem, 
                 solver: RungeKuttaSolver,
                 threshold: float = 0.1,
                 adjustment_type: BoundaryAdjustmentType = BoundaryAdjustmentType.HYBRID):
        self.ode_system = ode_system
        self.solver = solver
        self.threshold = threshold
        self.adjustment_type = adjustment_type
        
        self.boundaries: Dict[str, ClassificationBoundary] = {}
        self.change_rate_history: List[Tuple[float, np.ndarray]] = []
        self.adjustment_events: List[Dict] = []
        
    def initialize_boundary(self, category: str, center: np.ndarray, 
                           radius: Optional[np.ndarray] = None):
        n = len(center)
        
        if radius is None:
            radius = np.ones(n) * 0.5
            
        normal_vectors = np.eye(n)
        
        self.boundaries[category] = ClassificationBoundary(
            center=center,
            radius=radius,
            normal_vectors=normal_vectors
        )
        
    def compute_feature_change_rate(self) -> np.ndarray:
        if self.ode_system.state is None or self.ode_system.velocity is None:
            raise RuntimeError("ODE system not initialized")
            
        return self.ode_system.velocity.copy()
    
    def check_threshold_trigger(self, change_rate: np.ndarray) -> Tuple[bool, np.ndarray]:
        """检查是否触发阈值"""
        magnitude = np.linalg.norm(change_rate)
        
        triggered = magnitude > self.threshold
        
        direction = np.zeros_like(change_rate)
        if triggered and magnitude > 1e-10:
            direction = change_rate / magnitude
            
        return triggered, direction
    
    def adjust_boundary(self, category: str, change_rate: np.ndarray,
                       adjustment_magnitude: Optional[float] = None) -> Dict:
        """调整分类边界"""
        if category not in self.boundaries:
            raise ValueError(f"Boundary for category '{category}' not initialized")
            
        boundary = self.boundaries[category]
        triggered, direction = self.check_threshold_trigger(change_rate)
        
        adjustment_info = {
            'category': category,
            'triggered': triggered,
            'change_rate': change_rate.copy(),
            'direction': direction.copy(),
            'adjustment_type': self.adjustment_type.value,
            'old_center': boundary.center.copy(),
            'old_radius': boundary.radius.copy()
        }
        
        if not triggered:
            adjustment_info['adjusted'] = False
            return adjustment_info
            
        if adjustment_magnitude is None:
            adjustment_magnitude = np.linalg.norm(change_rate) * 0.1
            
        if self.adjustment_type == BoundaryAdjustmentType.THRESHOLD_TRIGGER:
            new_center = self._threshold_adjustment(boundary, direction, adjustment_magnitude)
        elif self.adjustment_type == BoundaryAdjustmentType.GRADIENT_BASED:
            new_center = self._gradient_adjustment(boundary, change_rate)
        else:
            new_center = self._hybrid_adjustment(boundary, change_rate, adjustment_magnitude)
            
        boundary.center = new_center
        boundary.last_adjustment = self.ode_system.time
        boundary.adjustment_history.append(np.linalg.norm(new_center - adjustment_info['old_center']))
        
        adjustment_info['new_center'] = boundary.center.copy()
        adjustment_info['new_radius'] = boundary.radius.copy()
        adjustment_info['adjusted'] = True
        adjustment_info['adjustment_magnitude'] = np.linalg.norm(
            boundary.center - adjustment_info['old_center']
        )
        
        self.adjustment_events.append(adjustment_info)
        
        return adjustment_info
    
    def _threshold_adjustment(self, boundary: ClassificationBoundary, 
                             direction: np.ndarray, magnitude: float) -> np.ndarray:
        """阈值触发调整"""
        return boundary.center + magnitude * direction
    
    def _gradient_adjustment(self, boundary: ClassificationBoundary,
                            change_rate: np.ndarray) -> np.ndarray:
        """梯度驱动调整"""
        gradient = change_rate / (1 + np.abs(change_rate))
        
        learning_rate = 0.1
        return boundary.center + learning_rate * gradient
    
    def _hybrid_adjustment(self, boundary: ClassificationBoundary,
                          change_rate: np.ndarray, magnitude: float) -> np.ndarray:
        """混合调整策略"""
        change_norm = np.linalg.norm(change_rate)
        if change_norm < 1e-10:
            return boundary.center.copy()
            
        direction = change_rate / change_norm
        
        gradient_factor = np.tanh(change_norm)
        
        adjustment = magnitude * direction * (1 + gradient_factor)
        
        return boundary.center + adjustment
    
    def update_all_boundaries(self, change_rates: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        results = {}
        
        for category, rate in change_rates.items():
            if category in self.boundaries:
                results[category] = self.adjust_boundary(category, rate)
                
        return results
    
    def get_boundary_for_category(self, category: str) -> Optional[ClassificationBoundary]:
        return self.boundaries.get(category)
    
    def get_all_boundaries(self) -> Dict[str, ClassificationBoundary]:
        return self.boundaries.copy()


class DynamicClassificationModel:
    """
    动态分类模型
    整合ODE系统、求解器和边界控制器
    """
    
    def __init__(self, n_features: int, n_categories: int,
                 ode_params: Optional[ODEParameters] = None,
                 step_config: Optional[AdaptiveStepConfig] = None,
                 threshold: float = 0.1):
        self.n_features = n_features
        self.n_categories = n_categories
        
        self.ode_system = SecondOrderODESystem(n_features, ode_params)
        self.solver = RungeKuttaSolver(self.ode_system, step_config)
        self.boundary_controller = AdaptiveBoundaryController(
            self.ode_system, self.solver, threshold
        )
        
        self.category_names: List[str] = []
        self.feature_weights: np.ndarray = np.ones(n_features)
        
    def initialize(self, category_centers: Dict[str, np.ndarray],
                   initial_features: Optional[np.ndarray] = None):
        self.category_names = list(category_centers.keys())
        
        for category, center in category_centers.items():
            self.boundary_controller.initialize_boundary(category, center)
            
        if initial_features is None:
            initial_features = np.zeros(self.n_features)
            
        self.ode_system.initialize_state(initial_features)
        
    def update_features(self, new_features: np.ndarray, 
                       time_delta: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        if self.ode_system.state is None:
            self.ode_system.initialize_state(new_features)
            return new_features, np.zeros(self.n_features)
            
        target_state = new_features
        
        external_force = (target_state - self.ode_system.state) * 10.0
        
        state, velocity, _ = self.solver.solve(
            self.ode_system.time + time_delta,
            external_force=external_force
        )
        
        return state, velocity
    
    def classify(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        distances = {}
        
        for category, boundary in self.boundary_controller.boundaries.items():
            distance = np.linalg.norm(features - boundary.center)
            distances[category] = distance
            
        sorted_categories = sorted(distances.items(), key=lambda x: x[1])
        best_category = sorted_categories[0][0]
        best_distance = sorted_categories[0][1]
        
        max_dist = max(distances.values()) if distances else 1.0
        confidence = 1.0 - (best_distance / max_dist) if max_dist > 0 else 1.0
        
        return best_category, confidence, distances
    
    def adapt_to_sample(self, features: np.ndarray, true_category: str,
                       time_delta: float = 0.1) -> Dict:
        state, velocity = self.update_features(features, time_delta)
        
        change_rate = self.boundary_controller.compute_feature_change_rate()
        
        adjustment_result = self.boundary_controller.adjust_boundary(
            true_category, change_rate
        )
        
        predicted_category, confidence, distances = self.classify(features)
        
        return {
            'state': state,
            'velocity': velocity,
            'change_rate': change_rate,
            'adjustment': adjustment_result,
            'predicted_category': predicted_category,
            'true_category': true_category,
            'confidence': confidence,
            'correct': predicted_category == true_category,
            'distances': distances
        }
    
    def get_steady_state(self, max_time: float = 100.0) -> Tuple[np.ndarray, bool]:
        state, velocity, time, converged = self.solver.solve_to_steady_state(max_time=max_time)
        return state, converged
    
    def get_boundary_centers(self) -> Dict[str, np.ndarray]:
        return {
            cat: boundary.center.copy() 
            for cat, boundary in self.boundary_controller.boundaries.items()
        }
    
    def get_adjustment_history(self) -> List[Dict]:
        return self.boundary_controller.adjustment_events.copy()


class NumericalStabilityChecker:
    """数值稳定性检查器"""
    
    @staticmethod
    def check_state_stability(state: np.ndarray, max_value: float = 1e6) -> Tuple[bool, str]:
        if np.any(np.isnan(state)):
            return False, "State contains NaN values"
        if np.any(np.isinf(state)):
            return False, "State contains infinite values"
        if np.any(np.abs(state) > max_value):
            return False, f"State values exceed maximum threshold {max_value}"
        return True, "State is stable"
    
    @staticmethod
    def check_velocity_stability(velocity: np.ndarray, max_value: float = 1e6) -> Tuple[bool, str]:
        if np.any(np.isnan(velocity)):
            return False, "Velocity contains NaN values"
        if np.any(np.isinf(velocity)):
            return False, "Velocity contains infinite values"
        if np.any(np.abs(velocity) > max_value):
            return False, f"Velocity values exceed maximum threshold {max_value}"
        return True, "Velocity is stable"
    
    @staticmethod
    def check_matrix_stability(matrix: np.ndarray, name: str = "Matrix") -> Tuple[bool, str]:
        if np.any(np.isnan(matrix)):
            return False, f"{name} contains NaN values"
        if np.any(np.isinf(matrix)):
            return False, f"{name} contains infinite values"
        
        cond = np.linalg.cond(matrix)
        if cond > 1e12:
            return False, f"{name} is ill-conditioned (condition number: {cond})"
            
        return True, f"{name} is stable"
    
    @staticmethod
    def stabilize_state(state: np.ndarray, max_value: float = 1e3) -> np.ndarray:
        stabilized = np.clip(state, -max_value, max_value)
        stabilized = np.nan_to_num(stabilized, nan=0.0, posinf=max_value, neginf=-max_value)
        return stabilized
    
    @staticmethod
    def stabilize_velocity(velocity: np.ndarray, max_value: float = 1e3) -> np.ndarray:
        stabilized = np.clip(velocity, -max_value, max_value)
        stabilized = np.nan_to_num(stabilized, nan=0.0, posinf=max_value, neginf=-max_value)
        return stabilized


def create_dynamic_classifier(n_features: int, 
                             category_centers: Dict[str, np.ndarray],
                             threshold: float = 0.1,
                             damping: float = 0.1,
                             stiffness: float = 1.0) -> DynamicClassificationModel:
    """创建动态分类器的便捷函数"""
    params = ODEParameters(
        damping=damping,
        stiffness=stiffness
    )
    
    step_config = AdaptiveStepConfig(
        initial_step=0.01,
        min_step=1e-6,
        max_step=0.1,
        tolerance=1e-6
    )
    
    model = DynamicClassificationModel(
        n_features=n_features,
        n_categories=len(category_centers),
        ode_params=params,
        step_config=step_config,
        threshold=threshold
    )
    
    model.initialize(category_centers)
    
    return model


import itertools
import math
from collections import defaultdict


class PermutationCombinationDecisionTree:
    def __init__(self, features: List[str], max_depth: int = 5):
        self.features = features
        self.max_depth = max_depth
        self.tree = None
        self.permutation_cache = {}
        self.combination_cache = {}
    
    def factorial(self, n: int) -> int:
        if n < 0:
            return 0
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def permutation_count(self, n: int, r: int) -> int:
        if r > n or n < 0 or r < 0:
            return 0
        cache_key = (n, r)
        if cache_key in self.permutation_cache:
            return self.permutation_cache[cache_key]
        result = self.factorial(n) // self.factorial(n - r)
        self.permutation_cache[cache_key] = result
        return result
    
    def combination_count(self, n: int, r: int) -> int:
        if r > n or n < 0 or r < 0:
            return 0
        cache_key = (n, r)
        if cache_key in self.combination_cache:
            return self.combination_cache[cache_key]
        result = math.comb(n, r)
        self.combination_cache[cache_key] = result
        return result
    
    def generate_feature_permutations(self, feature_subset: List[str], r: int) -> List[Tuple[str, ...]]:
        return list(itertools.permutations(feature_subset, min(r, len(feature_subset))))
    
    def generate_feature_combinations(self, feature_subset: List[str], r: int) -> List[Tuple[str, ...]]:
        return list(itertools.combinations(feature_subset, min(r, len(feature_subset))))
    
    def calculate_permutation_constraint(self, feature_values: Dict[str, Any], 
                                         constraint_rules: Dict[str, Any]) -> float:
        valid_permutations = 0
        total_permutations = 0
        
        for feature in self.features:
            if feature in feature_values and feature in constraint_rules:
                rule = constraint_rules[feature]
                value = feature_values[feature]
                
                if isinstance(rule, list):
                    valid_values = rule
                    if value in valid_values:
                        valid_permutations += 1
                elif isinstance(rule, dict):
                    if 'range' in rule:
                        min_val, max_val = rule['range']
                        if min_val <= value <= max_val:
                            valid_permutations += 1
                    elif 'set' in rule:
                        if value in rule['set']:
                            valid_permutations += 1
                
                total_permutations += 1
        
        if total_permutations == 0:
            return 1.0
        
        return valid_permutations / total_permutations
    
    def calculate_combination_constraint(self, selected_features: List[str], 
                                         target_count: int,
                                         feature_weights: Dict[str, float]) -> float:
        n = len(selected_features)
        if n < target_count:
            return 0.0
        
        total_combinations = self.combination_count(n, target_count)
        if total_combinations == 0:
            return 0.0
        
        weighted_sum = sum(feature_weights.get(f, 1.0) for f in selected_features)
        max_weighted_sum = max(feature_weights.values()) * target_count if feature_weights else target_count
        
        if max_weighted_sum == 0:
            return 0.0
        
        return min(1.0, weighted_sum / max_weighted_sum)
    
    def build_decision_tree(self, samples: List[Dict[str, Any]], 
                           labels: List[int],
                           depth: int = 0,
                           used_features: set = None) -> Dict:
        if used_features is None:
            used_features = set()
        
        if depth >= self.max_depth:
            return self._create_leaf_node(labels)
        
        if len(set(labels)) <= 1:
            return self._create_leaf_node(labels)
        
        if len(samples) == 0:
            return self._create_leaf_node(labels)
        
        available_features = [f for f in self.features if f not in used_features]
        if not available_features:
            return self._create_leaf_node(labels)
        
        best_feature, best_threshold, best_gain = self._find_best_split(
            samples, labels, available_features
        )
        
        if best_gain <= 0:
            return self._create_leaf_node(labels)
        
        left_samples, left_labels, right_samples, right_labels = self._split_samples(
            samples, labels, best_feature, best_threshold
        )
        
        new_used = used_features | {best_feature}
        
        node = {
            'type': 'decision',
            'feature': best_feature,
            'threshold': best_threshold,
            'permutation_weight': self.permutation_count(len(available_features), 1) / max(1, len(self.features)),
            'combination_weight': self.combination_count(len(available_features), 1) / max(1, len(self.features)),
            'left': self.build_decision_tree(left_samples, left_labels, depth + 1, new_used),
            'right': self.build_decision_tree(right_samples, right_labels, depth + 1, new_used)
        }
        
        return node
    
    def _create_leaf_node(self, labels: List[int]) -> Dict:
        if not labels:
            return {'type': 'leaf', 'prediction': 0, 'confidence': 0.0}
        
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        total = len(labels)
        best_label = max(label_counts.keys(), key=lambda x: label_counts[x])
        confidence = label_counts[best_label] / total
        
        return {
            'type': 'leaf',
            'prediction': best_label,
            'confidence': confidence,
            'distribution': {k: v / total for k, v in label_counts.items()}
        }
    
    def _find_best_split(self, samples: List[Dict], labels: List[int], 
                         available_features: List[str]) -> Tuple[str, float, float]:
        best_feature = None
        best_threshold = 0.0
        best_gain = -float('inf')
        
        current_entropy = self._calculate_entropy(labels)
        
        for feature in available_features:
            values = [s.get(feature, 0) for s in samples if feature in s]
            if not values:
                continue
            
            numeric_values = [v if isinstance(v, (int, float)) else hash(v) % 1000 for v in values]
            thresholds = list(set(numeric_values))
            
            for threshold in thresholds:
                left_labels = []
                right_labels = []
                
                for i, sample in enumerate(samples):
                    value = sample.get(feature, 0)
                    numeric_value = value if isinstance(value, (int, float)) else hash(value) % 1000
                    
                    if numeric_value <= threshold:
                        left_labels.append(labels[i])
                    else:
                        right_labels.append(labels[i])
                
                if not left_labels or not right_labels:
                    continue
                
                n = len(labels)
                n_left = len(left_labels)
                n_right = len(right_labels)
                
                left_entropy = self._calculate_entropy(left_labels)
                right_entropy = self._calculate_entropy(right_labels)
                
                gain = current_entropy - (n_left / n) * left_entropy - (n_right / n) * right_entropy
                
                permutation_bonus = self.permutation_count(len(available_features), 1) / max(1, len(self.features) * 10)
                gain += permutation_bonus * 0.1
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _calculate_entropy(self, labels: List[int]) -> float:
        if not labels:
            return 0.0
        
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        total = len(labels)
        entropy = 0.0
        
        for count in label_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _split_samples(self, samples: List[Dict], labels: List[int], 
                       feature: str, threshold: float) -> Tuple[List, List, List, List]:
        left_samples = []
        left_labels = []
        right_samples = []
        right_labels = []
        
        for i, sample in enumerate(samples):
            value = sample.get(feature, 0)
            numeric_value = value if isinstance(value, (int, float)) else hash(value) % 1000
            
            if numeric_value <= threshold:
                left_samples.append(sample)
                left_labels.append(labels[i])
            else:
                right_samples.append(sample)
                right_labels.append(labels[i])
        
        return left_samples, left_labels, right_samples, right_labels
    
    def predict(self, sample: Dict[str, Any]) -> Tuple[int, float]:
        if self.tree is None:
            return 0, 0.0
        
        return self._traverse_tree(sample, self.tree)
    
    def _traverse_tree(self, sample: Dict, node: Dict) -> Tuple[int, float]:
        if node['type'] == 'leaf':
            return node['prediction'], node['confidence']
        
        feature = node['feature']
        threshold = node['threshold']
        
        value = sample.get(feature, 0)
        numeric_value = value if isinstance(value, (int, float)) else hash(value) % 1000
        
        if numeric_value <= threshold:
            return self._traverse_tree(sample, node['left'])
        else:
            return self._traverse_tree(sample, node['right'])
    
    def fit(self, samples: List[Dict[str, Any]], labels: List[int]):
        self.tree = self.build_decision_tree(samples, labels)


class InclusionExclusionBoundaryCorrector:
    def __init__(self):
        self.sets = {}
        self.boundaries = {}
        self.correction_cache = {}
    
    def add_set(self, name: str, elements: set):
        self.sets[name] = elements.copy()
        self.correction_cache.clear()
    
    def set_boundary(self, name: str, lower: float, upper: float):
        self.boundaries[name] = (lower, upper)
    
    def calculate_union_size(self, set_names: List[str]) -> int:
        if not set_names:
            return 0
        
        cache_key = tuple(sorted(set_names))
        if cache_key in self.correction_cache:
            return self.correction_cache[cache_key]
        
        n = len(set_names)
        total = 0
        
        for k in range(1, n + 1):
            for combo in itertools.combinations(set_names, k):
                intersection = self.sets[combo[0]].copy()
                for name in combo[1:]:
                    intersection &= self.sets.get(name, set())
                
                size = len(intersection)
                if k % 2 == 1:
                    total += size
                else:
                    total -= size
        
        self.correction_cache[cache_key] = total
        return total
    
    def calculate_intersection_size(self, set_names: List[str]) -> int:
        if not set_names:
            return 0
        
        result = self.sets.get(set_names[0], set()).copy()
        for name in set_names[1:]:
            result &= self.sets.get(name, set())
        
        return len(result)
    
    def correct_classification_boundary(self, category_sets: Dict[str, set], 
                                        overlap_penalty: float = 0.5) -> Dict[str, Tuple[float, float]]:
        corrected_boundaries = {}
        
        all_categories = list(category_sets.keys())
        total_elements = self.calculate_union_size(all_categories)
        
        if total_elements == 0:
            return {cat: (0.0, 1.0) for cat in all_categories}
        
        for category, elements in category_sets.items():
            other_categories = [c for c in all_categories if c != category]
            
            exclusive_elements = elements.copy()
            for other in other_categories:
                exclusive_elements -= category_sets.get(other, set())
            
            exclusive_ratio = len(exclusive_elements) / max(1, len(elements))
            
            overlap_count = 0
            for other in other_categories:
                overlap_count += len(elements & category_sets.get(other, set()))
            
            overlap_ratio = overlap_count / max(1, len(elements) * len(other_categories))
            
            lower_correction = exclusive_ratio * (1 - overlap_penalty * overlap_ratio)
            upper_correction = 1.0 - overlap_penalty * overlap_ratio * 0.5
            
            corrected_boundaries[category] = (
                max(0.0, min(1.0, lower_correction)),
                max(0.0, min(1.0, upper_correction))
            )
        
        return corrected_boundaries
    
    def calculate_boundary_adjustment(self, category: str, 
                                      all_categories: List[str],
                                      sample_distribution: Dict[str, int]) -> float:
        if category not in self.sets:
            return 1.0
        
        category_size = len(self.sets[category])
        if category_size == 0:
            return 1.0
        
        overlaps = []
        for other in all_categories:
            if other != category and other in self.sets:
                intersection = self.sets[category] & self.sets[other]
                overlap_ratio = len(intersection) / category_size
                overlaps.append(overlap_ratio)
        
        if not overlaps:
            return 1.0
        
        avg_overlap = sum(overlaps) / len(overlaps)
        max_overlap = max(overlaps)
        
        adjustment = 1.0 - avg_overlap * 0.3 - max_overlap * 0.2
        
        return max(0.5, min(1.5, adjustment))
    
    def get_exclusive_elements(self, category: str, all_categories: List[str]) -> set:
        if category not in self.sets:
            return set()
        
        exclusive = self.sets[category].copy()
        for other in all_categories:
            if other != category and other in self.sets:
                exclusive -= self.sets[other]
        
        return exclusive
    
    def calculate_membership_probability(self, element: Any, 
                                         category_sets: Dict[str, set]) -> Dict[str, float]:
        probabilities = {}
        
        containing_categories = []
        for category, elements in category_sets.items():
            if element in elements:
                containing_categories.append(category)
        
        if not containing_categories:
            return {cat: 0.0 for cat in category_sets}
        
        if len(containing_categories) == 1:
            probabilities = {cat: 0.0 for cat in category_sets}
            probabilities[containing_categories[0]] = 1.0
            return probabilities
        
        union_size = self.calculate_union_size(containing_categories)
        if union_size == 0:
            return {cat: 1.0 / len(category_sets) for cat in category_sets}
        
        for category in containing_categories:
            exclusive = self.get_exclusive_elements(category, containing_categories)
            exclusive_ratio = len(exclusive) / max(1, union_size)
            
            overlap_contribution = 1.0 / len(containing_categories)
            
            probabilities[category] = exclusive_ratio * 0.7 + overlap_contribution * 0.3
        
        for category in category_sets:
            if category not in containing_categories:
                probabilities[category] = 0.0
        
        return probabilities


class PigeonholeBucketAdjuster:
    def __init__(self, min_bucket_size: int = 1, max_buckets: int = 20):
        self.min_bucket_size = min_bucket_size
        self.max_buckets = max_buckets
        self.buckets = {}
        self.bucket_stats = defaultdict(lambda: {'count': 0, 'overflow': 0})
    
    def calculate_min_buckets(self, n_items: int, capacity: int) -> int:
        if capacity <= 0:
            return self.max_buckets
        
        min_buckets = math.ceil(n_items / capacity)
        return min(min_buckets, self.max_buckets)
    
    def calculate_max_items_per_bucket(self, n_items: int, n_buckets: int) -> int:
        if n_buckets <= 0:
            return n_items
        
        return math.ceil(n_items / n_buckets)
    
    def assign_to_buckets(self, items: List[Any], 
                          key_func: callable,
                          n_buckets: int = None) -> Dict[int, List[Any]]:
        if not items:
            return {}
        
        if n_buckets is None:
            n_buckets = min(len(items), self.max_buckets)
        
        n_buckets = max(1, min(n_buckets, self.max_buckets))
        
        buckets = defaultdict(list)
        
        for item in items:
            key = key_func(item)
            bucket_id = hash(key) % n_buckets
            buckets[bucket_id].append(item)
        
        return dict(buckets)
    
    def dynamic_adjust_buckets(self, items: List[Any],
                               key_func: callable,
                               target_items_per_bucket: int = None) -> Dict[int, List[Any]]:
        if not items:
            return {}
        
        n_items = len(items)
        
        if target_items_per_bucket is None:
            target_items_per_bucket = max(self.min_bucket_size, 
                                          int(math.sqrt(n_items)))
        
        n_buckets = self.calculate_min_buckets(n_items, target_items_per_bucket)
        
        buckets = self.assign_to_buckets(items, key_func, n_buckets)
        
        max_iterations = 10
        for iteration in range(max_iterations):
            overloaded = []
            underloaded = []
            
            for bucket_id, bucket_items in buckets.items():
                if len(bucket_items) > target_items_per_bucket * 1.5:
                    overloaded.append(bucket_id)
                elif len(bucket_items) < target_items_per_bucket * 0.5:
                    underloaded.append(bucket_id)
            
            if not overloaded:
                break
            
            for bucket_id in overloaded:
                if bucket_id in buckets:
                    excess = buckets[bucket_id][target_items_per_bucket:]
                    buckets[bucket_id] = buckets[bucket_id][:target_items_per_bucket]
                    
                    for item in excess:
                        new_bucket_id = (bucket_id + 1) % n_buckets
                        attempts = 0
                        while len(buckets.get(new_bucket_id, [])) >= target_items_per_bucket and attempts < n_buckets:
                            new_bucket_id = (new_bucket_id + 1) % n_buckets
                            attempts += 1
                        
                        if new_bucket_id not in buckets:
                            buckets[new_bucket_id] = []
                        buckets[new_bucket_id].append(item)
        
        self.buckets = buckets
        return dict(buckets)
    
    def balance_buckets_by_pigeonhole(self, buckets: Dict[int, List[Any]]) -> Dict[int, List[Any]]:
        if not buckets:
            return {}
        
        all_items = []
        for bucket_items in buckets.values():
            all_items.extend(bucket_items)
        
        n_items = len(all_items)
        n_buckets = len(buckets)
        
        if n_buckets == 0:
            return {}
        
        target_per_bucket = self.calculate_max_items_per_bucket(n_items, n_buckets)
        
        balanced = {i: [] for i in range(n_buckets)}
        
        item_idx = 0
        for bucket_id in range(n_buckets):
            take = min(target_per_bucket, n_items - item_idx)
            balanced[bucket_id] = all_items[item_idx:item_idx + take]
            item_idx += take
        
        while item_idx < n_items:
            for bucket_id in range(n_buckets):
                if item_idx >= n_items:
                    break
                balanced[bucket_id].append(all_items[item_idx])
                item_idx += 1
        
        return balanced
    
    def detect_pigeonhole_violation(self, buckets: Dict[int, List[Any]], 
                                    capacity: int) -> List[int]:
        violations = []
        
        for bucket_id, items in buckets.items():
            if len(items) > capacity:
                violations.append(bucket_id)
        
        return violations
    
    def get_bucket_distribution_stats(self, buckets: Dict[int, List[Any]]) -> Dict[str, Any]:
        if not buckets:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'total': 0}
        
        sizes = [len(items) for items in buckets.values()]
        
        return {
            'mean': np.mean(sizes),
            'std': np.std(sizes),
            'min': min(sizes),
            'max': max(sizes),
            'total': sum(sizes),
            'bucket_count': len(buckets),
            'variance': np.var(sizes)
        }
    
    def optimize_bucket_count(self, n_items: int, 
                              target_variance: float = 1.0) -> int:
        if n_items <= 0:
            return 1
        
        best_bucket_count = 1
        best_variance = float('inf')
        
        for n_buckets in range(1, min(n_items + 1, self.max_buckets + 1)):
            base_size = n_items // n_buckets
            remainder = n_items % n_buckets
            
            sizes = [base_size + (1 if i < remainder else 0) for i in range(n_buckets)]
            variance = np.var(sizes)
            
            if variance < best_variance:
                best_variance = variance
                best_bucket_count = n_buckets
            
            if variance <= target_variance:
                return n_buckets
        
        return best_bucket_count


class CombinationCountValidator:
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.validation_history = []
    
    def combination_count(self, n: int, r: int) -> int:
        if r > n or n < 0 or r < 0:
            return 0
        return math.comb(n, r)
    
    def permutation_count(self, n: int, r: int) -> int:
        if r > n or n < 0 or r < 0:
            return 0
        result = 1
        for i in range(n, n - r, -1):
            result *= i
        return result
    
    def calculate_expected_distribution(self, total_items: int, 
                                        categories: Dict[str, int],
                                        sample_size: int) -> Dict[str, float]:
        expected = {}
        total_category_items = sum(categories.values())
        
        if total_category_items == 0 or sample_size == 0:
            return {cat: 0.0 for cat in categories}
        
        for category, count in categories.items():
            probability = count / total_category_items
            expected_count = sample_size * probability
            expected[category] = expected_count
        
        return expected
    
    def calculate_combination_probability(self, n: int, k: int, 
                                          favorable: int) -> float:
        if k > n or favorable > n or k <= 0:
            return 0.0
        
        total_combinations = self.combination_count(n, k)
        if total_combinations == 0:
            return 0.0
        
        favorable_combinations = self.combination_count(favorable, k)
        
        return favorable_combinations / total_combinations
    
    def validate_distribution(self, observed: Dict[str, int], 
                              expected: Dict[str, float]) -> Dict[str, Any]:
        if not observed or not expected:
            return {'valid': False, 'reason': 'Empty distribution'}
        
        all_categories = set(observed.keys()) | set(expected.keys())
        
        total_observed = sum(observed.values())
        total_expected = sum(expected.values())
        
        if total_observed == 0 or total_expected == 0:
            return {'valid': False, 'reason': 'Zero total'}
        
        chi_square = 0.0
        deviations = {}
        
        for category in all_categories:
            obs = observed.get(category, 0)
            exp = expected.get(category, 0.0)
            
            if exp > 0:
                deviation = (obs - exp) ** 2 / exp
                chi_square += deviation
                deviations[category] = abs(obs - exp) / max(1, exp)
            else:
                deviations[category] = float('inf') if obs > 0 else 0.0
        
        df = len(all_categories) - 1
        if df <= 0:
            df = 1
        
        critical_value = self._get_chi_square_critical(df)
        
        is_valid = chi_square <= critical_value
        
        result = {
            'valid': is_valid,
            'chi_square': chi_square,
            'critical_value': critical_value,
            'degrees_of_freedom': df,
            'deviations': deviations,
            'max_deviation': max(deviations.values()) if deviations else 0.0
        }
        
        self.validation_history.append(result)
        return result
    
    def _get_chi_square_critical(self, df: int, alpha: float = 0.05) -> float:
        critical_values = {
            1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
            6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
            11: 19.675, 12: 21.026, 13: 22.362, 14: 23.685, 15: 24.996,
            16: 26.296, 17: 27.587, 18: 28.869, 19: 30.144, 20: 31.410
        }
        
        if df in critical_values:
            return critical_values[df]
        
        return 3.841 + df * 2.0
    
    def iterative_validation(self, observed: Dict[str, int],
                             expected: Dict[str, float],
                             max_iterations: int = 10) -> Dict[str, Any]:
        current_observed = observed.copy()
        current_expected = expected.copy()
        
        iteration_results = []
        
        for iteration in range(max_iterations):
            validation = self.validate_distribution(current_observed, current_expected)
            validation['iteration'] = iteration
            iteration_results.append(validation)
            
            if validation['valid']:
                break
            
            for category in current_expected:
                if category in current_observed:
                    obs = current_observed[category]
                    exp = current_expected[category]
                    
                    if abs(obs - exp) > self.tolerance * max(1, exp):
                        adjustment = (exp - obs) * 0.3
                        current_expected[category] = exp + adjustment
        
        return {
            'final_valid': iteration_results[-1]['valid'] if iteration_results else False,
            'iterations': len(iteration_results),
            'iteration_results': iteration_results
        }
    
    def calculate_multinomial_probability(self, n: int, 
                                          counts: Dict[str, int],
                                          probabilities: Dict[str, float]) -> float:
        if n <= 0:
            return 0.0
        
        total_count = sum(counts.values())
        if total_count != n:
            return 0.0
        
        numerator = math.factorial(n)
        denominator = 1.0
        prob_product = 1.0
        
        for category, count in counts.items():
            denominator *= math.factorial(count)
            prob = probabilities.get(category, 0.0)
            if prob > 0:
                prob_product *= prob ** count
            elif count > 0:
                return 0.0
        
        return (numerator / denominator) * prob_product
    
    def validate_sample_combination(self, sample: List[Any],
                                    population_categories: Dict[str, set],
                                    sample_size: int) -> Dict[str, Any]:
        sample_set = set(sample)
        
        observed_counts = {}
        for category, elements in population_categories.items():
            observed_counts[category] = len(sample_set & elements)
        
        total_population = sum(len(elements) for elements in population_categories.values())
        
        expected_counts = {}
        for category, elements in population_categories.items():
            if total_population > 0:
                expected_counts[category] = sample_size * len(elements) / total_population
            else:
                expected_counts[category] = 0.0
        
        validation = self.validate_distribution(observed_counts, expected_counts)
        
        validation['observed_counts'] = observed_counts
        validation['expected_counts'] = expected_counts
        
        return validation
    
    def get_validation_summary(self) -> Dict[str, Any]:
        if not self.validation_history:
            return {'total_validations': 0}
        
        valid_count = sum(1 for v in self.validation_history if v['valid'])
        
        return {
            'total_validations': len(self.validation_history),
            'valid_count': valid_count,
            'valid_rate': valid_count / len(self.validation_history),
            'average_chi_square': np.mean([v['chi_square'] for v in self.validation_history]),
            'average_max_deviation': np.mean([v['max_deviation'] for v in self.validation_history])
        }


class CombinatorialDecisionModel:
    def __init__(self, features: List[str] = None):
        self.features = features or []
        self.decision_tree = PermutationCombinationDecisionTree(self.features)
        self.boundary_corrector = InclusionExclusionBoundaryCorrector()
        self.bucket_adjuster = PigeonholeBucketAdjuster()
        self.validator = CombinationCountValidator()
        self.is_fitted = False
    
    def fit(self, samples: List[Dict[str, Any]], labels: List[int]):
        if not self.features:
            all_features = set()
            for sample in samples:
                all_features.update(sample.keys())
            self.features = list(all_features)
            self.decision_tree.features = self.features
        
        self.decision_tree.fit(samples, labels)
        
        category_sets = defaultdict(set)
        for i, sample in enumerate(samples):
            category = labels[i]
            sample_id = sample.get('id', i)
            category_sets[category].add(sample_id)
        
        for category, elements in category_sets.items():
            self.boundary_corrector.add_set(str(category), elements)
        
        self.is_fitted = True
    
    def predict(self, sample: Dict[str, Any]) -> Tuple[int, float]:
        if not self.is_fitted:
            return 0, 0.0
        
        return self.decision_tree.predict(sample)
    
    def correct_boundaries(self, category_sets: Dict[str, set]) -> Dict[str, Tuple[float, float]]:
        return self.boundary_corrector.correct_classification_boundary(category_sets)
    
    def adjust_buckets(self, items: List[Any], key_func: callable) -> Dict[int, List[Any]]:
        return self.bucket_adjuster.dynamic_adjust_buckets(items, key_func)
    
    def validate_distribution(self, observed: Dict[str, int], 
                              expected: Dict[str, float]) -> Dict[str, Any]:
        return self.validator.validate_distribution(observed, expected)
    
    def get_combination_count(self, n: int, r: int) -> int:
        return self.validator.combination_count(n, r)
    
    def get_permutation_count(self, n: int, r: int) -> int:
        return self.decision_tree.permutation_count(n, r)


def validate_combinatorial_operations():
    print("=" * 50)
    print("组合数学决策模型验证")
    print("=" * 50)
    
    print("\n1. 排列组合决策树验证")
    tree = PermutationCombinationDecisionTree(['f1', 'f2', 'f3'])
    print(f"   P(5,3) = {tree.permutation_count(5, 3)} (期望: 60)")
    print(f"   C(5,3) = {tree.combination_count(5, 3)} (期望: 10)")
    
    perms = tree.generate_feature_permutations(['a', 'b', 'c'], 2)
    print(f"   特征排列数: {len(perms)} (期望: 6)")
    
    print("\n2. 容斥原理边界修正验证")
    corrector = InclusionExclusionBoundaryCorrector()
    corrector.add_set('A', {1, 2, 3, 4, 5})
    corrector.add_set('B', {4, 5, 6, 7, 8})
    corrector.add_set('C', {5, 6, 9, 10})
    
    union_size = corrector.calculate_union_size(['A', 'B', 'C'])
    print(f"   并集大小: {union_size} (期望: 10)")
    
    intersection_size = corrector.calculate_intersection_size(['A', 'B'])
    print(f"   A与B交集大小: {intersection_size} (期望: 2)")
    
    print("\n3. 鸽巢原理分桶验证")
    adjuster = PigeonholeBucketAdjuster()
    min_buckets = adjuster.calculate_min_buckets(100, 10)
    print(f"   最小桶数(100项,容量10): {min_buckets} (期望: 10)")
    
    items = list(range(50))
    buckets = adjuster.dynamic_adjust_buckets(items, lambda x: x)
    print(f"   分桶数量: {len(buckets)}")
    
    print("\n4. 组合计数验证")
    validator = CombinationCountValidator()
    prob = validator.calculate_combination_probability(10, 3, 5)
    print(f"   组合概率: {prob:.6f}")
    
    observed = {'A': 30, 'B': 40, 'C': 30}
    expected = {'A': 33.3, 'B': 33.3, 'C': 33.3}
    result = validator.validate_distribution(observed, expected)
    print(f"   分布验证: valid={result['valid']}, chi_square={result['chi_square']:.4f}")
    
    print("\n" + "=" * 50)
    print("所有验证完成!")
    print("=" * 50)


class PrimeSieve:
    """素数筛法生成器"""
    
    def __init__(self, max_prime: int = 10000):
        self.max_prime = max_prime
        self._primes = []
        self._sieve = []
        self._build_sieve()
    
    def _build_sieve(self):
        self._sieve = [True] * (self.max_prime + 1)
        self._sieve[0] = self._sieve[1] = False
        
        for i in range(2, int(math.sqrt(self.max_prime)) + 1):
            if self._sieve[i]:
                for j in range(i * i, self.max_prime + 1, i):
                    self._sieve[j] = False
        
        self._primes = [i for i in range(2, self.max_prime + 1) if self._sieve[i]]
    
    def get_primes(self, count: int = None) -> List[int]:
        if count is None:
            return self._primes.copy()
        return self._primes[:count]
    
    def is_prime(self, n: int) -> bool:
        if n < 0 or n > self.max_prime:
            return self._is_prime_large(n)
        return self._sieve[n]
    
    def _is_prime_large(self, n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def get_prime_at_index(self, index: int) -> int:
        if index < len(self._primes):
            return self._primes[index]
        return self._primes[-1]


class PrimeFactorization:
    """素数分解器"""
    
    def __init__(self, prime_sieve: PrimeSieve = None):
        self.prime_sieve = prime_sieve or PrimeSieve()
    
    def factorize(self, n: int) -> Dict[int, int]:
        if n <= 1:
            return {}
        
        factors = {}
        primes = self.prime_sieve.get_primes()
        
        for prime in primes:
            if prime * prime > n:
                break
            while n % prime == 0:
                factors[prime] = factors.get(prime, 0) + 1
                n //= prime
        
        if n > 1:
            factors[n] = 1
        
        return factors
    
    def get_prime_factors(self, n: int) -> List[int]:
        factors = self.factorize(n)
        return list(factors.keys())
    
    def get_factor_signature(self, n: int) -> int:
        factors = self.factorize(n)
        signature = 0
        for prime, exp in factors.items():
            signature ^= (prime * exp)
        return signature


class EulerTotientFunction:
    """欧拉函数计算器"""
    
    def __init__(self, prime_sieve: PrimeSieve = None):
        self.prime_sieve = prime_sieve or PrimeSieve()
        self.factorizer = PrimeFactorization(self.prime_sieve)
        self._cache = {}
    
    def compute(self, n: int) -> int:
        if n in self._cache:
            return self._cache[n]
        
        if n <= 0:
            return 0
        if n == 1:
            return 1
        
        factors = self.factorizer.factorize(n)
        
        result = n
        for prime in factors:
            result = result // prime * (prime - 1)
        
        self._cache[n] = result
        return result
    
    def compute_batch(self, max_n: int) -> np.ndarray:
        phi = np.arange(max_n + 1, dtype=np.int64)
        
        for i in range(2, max_n + 1):
            if phi[i] == i:
                for j in range(i, max_n + 1, i):
                    phi[j] = phi[j] // i * (i - 1)
        
        return phi
    
    def get_coprime_count(self, n: int, m: int) -> int:
        from math import gcd
        return self.compute(n) * self.compute(m) // self.compute(gcd(n, m))


class ModularArithmetic:
    """模运算工具类"""
    
    @staticmethod
    def mod_pow(base: int, exp: int, mod: int) -> int:
        result = 1
        base = base % mod
        
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod
            exp //= 2
            base = (base * base) % mod
        
        return result
    
    @staticmethod
    def mod_inverse(a: int, mod: int) -> int:
        return ModularArithmetic._extended_gcd(a, mod)[1] % mod
    
    @staticmethod
    def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if b == 0:
            return a, 1, 0
        gcd, x, y = ModularArithmetic._extended_gcd(b, a % b)
        return gcd, y, x - (a // b) * y
    
    @staticmethod
    def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
        if len(remainders) != len(moduli):
            raise ValueError("Remainders and moduli must have same length")
        
        result = 0
        M = 1
        for m in moduli:
            M *= m
        
        for r, m in zip(remainders, moduli):
            Mi = M // m
            yi = ModularArithmetic.mod_inverse(Mi, m)
            result += r * Mi * yi
        
        return result % M


class NumberTheoryRandomEncoder:
    """数论驱动随机编码模型"""
    
    def __init__(self, seed: int = None, prime_count: int = 100):
        self.prime_sieve = PrimeSieve(max_prime=prime_count * 20)
        self.euler = EulerTotientFunction(self.prime_sieve)
        self.factorizer = PrimeFactorization(self.prime_sieve)
        
        self.primes = self.prime_sieve.get_primes(prime_count)
        
        if seed is None:
            seed = self.primes[0] * self.primes[1] * self.primes[2]
        
        self.seed = seed
        self.state = seed
        
        self._moduli = self._initialize_moduli()
        self._current_modulus_index = 0
        
        self._period_adjuster = 1
        self._counter = 0
        
        self._feature_modulus_map = {}
    
    def _initialize_moduli(self) -> List[int]:
        moduli = []
        
        for i in range(0, min(10, len(self.primes)), 2):
            p1 = self.primes[i]
            p2 = self.primes[i + 1] if i + 1 < len(self.primes) else self.primes[i]
            moduli.append(p1 * p2)
        
        moduli.append(self.primes[0] ** 2)
        moduli.append(self.primes[1] ** 3)
        
        return moduli
    
    def _generate_prime_based_random(self) -> int:
        if self.state == 0:
            self.state = self.primes[0] * self.primes[1]
        
        factors = self.factorizer.factorize(self.state)
        
        if not factors:
            factors = {self.primes[0]: 1}
        
        prime_product = 1
        for prime, exp in factors.items():
            prime_product = (prime_product * ModularArithmetic.mod_pow(prime, exp, 2**31 - 1)) % (2**31 - 1)
        
        large_prime = self.primes[self._counter % len(self.primes)]
        
        self.state = (self.state * large_prime + prime_product + self._counter + 1) % (2**31 - 1)
        
        if self.state == 0:
            self.state = large_prime
        
        return self.state
    
    def generate_random(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        raw_random = self._generate_prime_based_random()
        
        modulus = self._moduli[self._current_modulus_index]
        period = self.euler.compute(modulus)
        
        adjusted_random = (raw_random * self._period_adjuster) % modulus
        
        normalized = adjusted_random / modulus
        
        self._counter += 1
        self._update_period_adjuster()
        
        return min_val + normalized * (max_val - min_val)
    
    def generate_random_int(self, min_val: int, max_val: int) -> int:
        return int(self.generate_random(min_val, max_val + 1))
    
    def _update_period_adjuster(self):
        if self._counter % 100 == 0:
            self._period_adjuster = self.euler.compute(
                self.primes[self._counter % len(self.primes)]
            )
    
    def set_feature_modulus_mapping(self, feature_name: str, modulus_index: int):
        self._feature_modulus_map[feature_name] = modulus_index % len(self._moduli)
    
    def map_feature_to_random(self, feature_value: float, feature_name: str = None) -> float:
        if feature_name and feature_name in self._feature_modulus_map:
            modulus_index = self._feature_modulus_map[feature_name]
        else:
            modulus_index = self._current_modulus_index
        
        modulus = self._moduli[modulus_index]
        
        feature_int = int(abs(feature_value * 1e6)) % modulus
        
        random_base = self.generate_random()
        
        mapped_value = (feature_int + int(random_base * modulus)) % modulus
        
        phi_n = self.euler.compute(modulus)
        period_enhanced = ModularArithmetic.mod_pow(
            mapped_value, phi_n, modulus
        )
        
        return period_enhanced / modulus
    
    def dynamic_modulus_switch(self, feature_vector: np.ndarray) -> int:
        feature_hash = 0
        for val in feature_vector:
            feature_hash = (feature_hash * 31 + int(abs(val * 1000))) % (2**31 - 1)
        
        factors = self.factorizer.factorize(feature_hash % 1000 + 2)
        
        if factors:
            prime_sum = sum(factors.keys())
            new_index = prime_sum % len(self._moduli)
        else:
            new_index = self._counter % len(self._moduli)
        
        self._current_modulus_index = new_index
        return new_index
    
    def multi_modulus_mapping(self, feature_values: Dict[str, float]) -> Dict[str, float]:
        results = {}
        
        for feature_name, value in feature_values.items():
            if feature_name in self._feature_modulus_map:
                modulus_index = self._feature_modulus_map[feature_name]
            else:
                modulus_index = hash(feature_name) % len(self._moduli)
            
            modulus = self._moduli[modulus_index]
            phi_n = self.euler.compute(modulus)
            
            feature_int = int(abs(value * 1e6)) % modulus
            
            random_component = self.generate_random()
            
            mapped = (feature_int * int(random_component * phi_n)) % modulus
            
            period_factor = ModularArithmetic.mod_pow(mapped, phi_n, modulus)
            
            results[feature_name] = period_factor / modulus
        
        return results
    
    def chinese_remainder_encode(self, values: List[float]) -> float:
        if not values:
            return 0.0
        
        remainders = []
        moduli = []
        
        for i, val in enumerate(values):
            mod_idx = i % len(self._moduli)
            modulus = self._moduli[mod_idx]
            
            remainder = int(abs(val * 1e6)) % modulus
            remainders.append(remainder)
            moduli.append(modulus)
        
        try:
            combined = ModularArithmetic.chinese_remainder_theorem(remainders, moduli)
            total_modulus = 1
            for m in moduli:
                total_modulus *= m
            
            return combined / total_modulus
        except:
            return self.generate_random()
    
    def euler_period_adjusted_random(self, base_period: int = None) -> float:
        if base_period is None:
            base_period = self._moduli[self._current_modulus_index]
        
        phi_period = self.euler.compute(base_period)
        
        raw_random = self.generate_random()
        
        period_multiplier = 1 + (raw_random * phi_period) / base_period
        
        adjusted = (raw_random * period_multiplier) % 1.0
        
        return adjusted
    
    def reset_state(self, new_seed: int = None):
        if new_seed is not None:
            self.seed = new_seed
        self.state = self.seed
        self._counter = 0
        self._period_adjuster = 1
        self._current_modulus_index = 0
    
    def get_state_info(self) -> Dict[str, Any]:
        return {
            'seed': self.seed,
            'current_state': self.state,
            'counter': self._counter,
            'current_modulus': self._moduli[self._current_modulus_index],
            'period_adjuster': self._period_adjuster,
            'moduli_count': len(self._moduli)
        }
    
    def validate_randomness_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        samples = []
        for _ in range(n_samples):
            samples.append(self.generate_random())
        
        samples = np.array(samples)
        
        mean = np.mean(samples)
        variance = np.var(samples)
        
        expected_mean = 0.5
        expected_variance = 1.0 / 12.0
        
        runs = 0
        for i in range(1, len(samples)):
            if (samples[i] > mean) != (samples[i-1] > mean):
                runs += 1
        
        expected_runs = 2 * n_samples * (1 - mean) * mean
        runs_ratio = runs / expected_runs if expected_runs > 0 else 0
        
        chi_square = 0.0
        n_bins = 10
        hist, _ = np.histogram(samples, bins=n_bins, range=(0, 1))
        expected_count = n_samples / n_bins
        for count in hist:
            chi_square += (count - expected_count) ** 2 / expected_count
        
        return {
            'mean': mean,
            'expected_mean': expected_mean,
            'mean_deviation': abs(mean - expected_mean),
            'variance': variance,
            'expected_variance': expected_variance,
            'variance_deviation': abs(variance - expected_variance),
            'runs_count': runs,
            'expected_runs': expected_runs,
            'runs_ratio': runs_ratio,
            'chi_square': chi_square,
            'quality_score': 1.0 - (abs(mean - expected_mean) + abs(variance - expected_variance) / 2)
        }


class NumberTheoryQuestionEncoder:
    """基于数论的题目编码器"""
    
    def __init__(self, seed: int = None):
        self.random_encoder = NumberTheoryRandomEncoder(seed)
        self.feature_weights = {}
        self.encoding_history = []
    
    def encode_question_features(self, features: Dict[str, float]) -> np.ndarray:
        feature_names = list(features.keys())
        feature_values = np.array(list(features.values()))
        
        self.random_encoder.dynamic_modulus_switch(feature_values)
        
        mapped_features = self.random_encoder.multi_modulus_mapping(features)
        
        encoded = np.array([mapped_features.get(name, 0.0) for name in feature_names])
        
        return encoded
    
    def generate_question_variant_seed(self, question_id: int, 
                                        difficulty: float,
                                        topic_hash: int) -> int:
        primes = self.random_encoder.primes
        
        p1 = primes[question_id % len(primes)]
        p2 = primes[(question_id + 1) % len(primes)]
        p3 = primes[(question_id + 2) % len(primes)]
        
        base_seed = p1 * p2 * p3
        
        difficulty_factor = int(difficulty * 1000) % p1
        topic_factor = topic_hash % p2
        
        variant_seed = (base_seed + difficulty_factor * p3 + topic_factor) % (2**31 - 1)
        
        return variant_seed
    
    def randomize_question_options(self, options: List[str], 
                                    correct_index: int,
                                    question_seed: int) -> Tuple[List[str], int]:
        self.random_encoder.reset_state(question_seed)
        
        n = len(options)
        permutation = list(range(n))
        
        for i in range(n - 1, 0, -1):
            j = self.random_encoder.generate_random_int(0, i)
            permutation[i], permutation[j] = permutation[j], permutation[i]
        
        new_options = [options[i] for i in permutation]
        new_correct_index = permutation.index(correct_index)
        
        return new_options, new_correct_index
    
    def generate_numerical_parameters(self, param_ranges: Dict[str, Tuple[float, float]],
                                       question_seed: int) -> Dict[str, float]:
        self.random_encoder.reset_state(question_seed)
        
        parameters = {}
        for name, (min_val, max_val) in param_ranges.items():
            self.random_encoder.set_feature_modulus_mapping(name, hash(name) % 10)
            parameters[name] = self.random_encoder.generate_random(min_val, max_val)
        
        return parameters
    
    def compute_question_hash(self, question_data: Dict[str, Any]) -> int:
        feature_vector = []
        
        for key, value in question_data.items():
            if isinstance(value, (int, float)):
                feature_vector.append(float(value))
            elif isinstance(value, str):
                feature_vector.append(float(hash(value) % 10000) / 10000)
        
        combined = self.random_encoder.chinese_remainder_encode(feature_vector)
        
        hash_value = int(combined * (2**31 - 1))
        
        return hash_value
    
    def validate_randomness_quality(self, n_samples: int = 1000) -> Dict[str, float]:
        samples = []
        for _ in range(n_samples):
            samples.append(self.random_encoder.generate_random())
        
        samples = np.array(samples)
        
        mean = np.mean(samples)
        variance = np.var(samples)
        
        expected_mean = 0.5
        expected_variance = 1.0 / 12.0
        
        runs = 0
        for i in range(1, len(samples)):
            if (samples[i] > mean) != (samples[i-1] > mean):
                runs += 1
        
        expected_runs = 2 * n_samples * (1 - mean) * mean
        runs_ratio = runs / expected_runs if expected_runs > 0 else 0
        
        chi_square = 0.0
        n_bins = 10
        hist, _ = np.histogram(samples, bins=n_bins, range=(0, 1))
        expected_count = n_samples / n_bins
        for count in hist:
            chi_square += (count - expected_count) ** 2 / expected_count
        
        return {
            'mean': mean,
            'expected_mean': expected_mean,
            'mean_deviation': abs(mean - expected_mean),
            'variance': variance,
            'expected_variance': expected_variance,
            'variance_deviation': abs(variance - expected_variance),
            'runs_count': runs,
            'expected_runs': expected_runs,
            'runs_ratio': runs_ratio,
            'chi_square': chi_square,
            'quality_score': 1.0 - (abs(mean - expected_mean) + abs(variance - expected_variance) / 2)
        }


def validate_number_theory_random_encoder():
    print("=" * 60)
    print("数论驱动随机编码模型验证")
    print("=" * 60)
    
    print("\n1. 素数筛法验证")
    sieve = PrimeSieve(100)
    primes = sieve.get_primes(10)
    print(f"   前10个素数: {primes}")
    print(f"   验证: 2是素数? {sieve.is_prime(2)}, 4是素数? {sieve.is_prime(4)}")
    
    print("\n2. 素数分解验证")
    factorizer = PrimeFactorization(sieve)
    n = 120
    factors = factorizer.factorize(n)
    print(f"   {n} = {factors}")
    print(f"   因子签名: {factorizer.get_factor_signature(n)}")
    
    print("\n3. 欧拉函数验证")
    euler = EulerTotientFunction(sieve)
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(f"   φ({n}) = {euler.compute(n)}", end="")
    print()
    
    print("\n4. 模运算验证")
    print(f"   3^100 mod 7 = {ModularArithmetic.mod_pow(3, 100, 7)}")
    crt_result = ModularArithmetic.chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
    print(f"   中国剩余定理: x ≡ 2(mod 3), x ≡ 3(mod 5), x ≡ 2(mod 7) => x = {crt_result}")
    
    print("\n5. 数论随机编码器验证")
    encoder = NumberTheoryRandomEncoder(seed=42)
    
    print("   生成10个随机数:")
    randoms = [encoder.generate_random() for _ in range(10)]
    print(f"   {randoms}")
    
    print("\n   特征映射测试:")
    features = {'difficulty': 0.7, 'complexity': 0.5, 'novelty': 0.3}
    mapped = encoder.multi_modulus_mapping(features)
    print(f"   原始特征: {features}")
    print(f"   映射后: {mapped}")
    
    print("\n   动态模数切换测试:")
    feature_vec = np.array([0.5, 0.3, 0.8, 0.2])
    mod_idx = encoder.dynamic_modulus_switch(feature_vec)
    print(f"   特征向量: {feature_vec}")
    print(f"   选择的模数索引: {mod_idx}")
    
    print("\n6. 随机性质量测试")
    quality = encoder.validate_randomness_quality(1000)
    print(f"   均值: {quality['mean']:.6f} (期望: {quality['expected_mean']:.6f})")
    print(f"   方差: {quality['variance']:.6f} (期望: {quality['expected_variance']:.6f})")
    print(f"   游程比: {quality['runs_ratio']:.6f}")
    print(f"   卡方值: {quality['chi_square']:.4f}")
    print(f"   质量评分: {quality['quality_score']:.6f}")
    
    print("\n7. 题目编码器验证")
    question_encoder = NumberTheoryQuestionEncoder(seed=123)
    
    question_features = {
        'difficulty': 0.6,
        'knowledge_depth': 0.7,
        'computation_complexity': 0.4,
        'abstract_level': 0.5
    }
    encoded = question_encoder.encode_question_features(question_features)
    print(f"   原始特征: {question_features}")
    print(f"   编码后: {encoded}")
    
    variant_seed = question_encoder.generate_question_variant_seed(1, 0.5, 100)
    print(f"   题目变体种子: {variant_seed}")
    
    print("\n" + "=" * 60)
    print("所有验证完成!")
    print("=" * 60)


from scipy import stats
from scipy.special import logsumexp, gammaln, comb
from scipy.optimize import minimize


class MultinomialDistribution:
    """多项分布模型"""
    
    def __init__(self, n_trials: int, probabilities: np.ndarray):
        self.n_trials = n_trials
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self._validate_parameters()
        
    def _validate_parameters(self):
        if self.n_trials < 0:
            raise ValueError("试验次数必须非负")
        if np.any(self.probabilities < 0):
            raise ValueError("概率必须非负")
        prob_sum = np.sum(self.probabilities)
        if prob_sum > 0:
            self.probabilities = self.probabilities / prob_sum
        else:
            self.probabilities = np.ones_like(self.probabilities) / len(self.probabilities)
    
    def pmf(self, counts: np.ndarray) -> float:
        if np.sum(counts) != self.n_trials:
            return 0.0
        if len(counts) != len(self.probabilities):
            raise ValueError("计数向量维度与概率向量不匹配")
        
        log_prob = gammaln(self.n_trials + 1)
        log_prob -= np.sum(gammaln(counts + 1))
        valid_mask = self.probabilities > 0
        log_prob += np.sum(counts[valid_mask] * np.log(self.probabilities[valid_mask] + 1e-300))
        
        return np.exp(log_prob)
    
    def log_pmf(self, counts: np.ndarray) -> float:
        if np.sum(counts) != self.n_trials:
            return -np.inf
        if len(counts) != len(self.probabilities):
            return -np.inf
        
        log_prob = gammaln(self.n_trials + 1)
        log_prob -= np.sum(gammaln(counts + 1))
        valid_mask = self.probabilities > 0
        log_prob += np.sum(counts[valid_mask] * np.log(self.probabilities[valid_mask] + 1e-300))
        
        return log_prob
    
    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.multinomial(self.n_trials, self.probabilities, size=size)
    
    def mean(self) -> np.ndarray:
        return self.n_trials * self.probabilities
    
    def variance(self) -> np.ndarray:
        return self.n_trials * self.probabilities * (1 - self.probabilities)


class NegativeBinomialDistribution:
    """负二项分布模型"""
    
    def __init__(self, r: float, p: float):
        self.r = float(r)
        self.p = float(p)
        self._validate_parameters()
        
    def _validate_parameters(self):
        if self.r <= 0:
            raise ValueError("参数r必须为正数")
        if self.p <= 0 or self.p > 1:
            raise ValueError("参数p必须在(0,1]范围内")
    
    def pmf(self, k: int) -> float:
        if k < 0:
            return 0.0
        log_prob = gammaln(k + self.r) - gammaln(k + 1) - gammaln(self.r)
        log_prob += self.r * np.log(self.p) + k * np.log(1 - self.p + 1e-300)
        return np.exp(log_prob)
    
    def log_pmf(self, k: int) -> float:
        if k < 0:
            return -np.inf
        log_prob = gammaln(k + self.r) - gammaln(k + 1) - gammaln(self.r)
        log_prob += self.r * np.log(self.p) + k * np.log(1 - self.p + 1e-300)
        return log_prob
    
    def pmf_array(self, k_max: int) -> np.ndarray:
        k_values = np.arange(k_max + 1)
        log_probs = gammaln(k_values + self.r) - gammaln(k_values + 1) - gammaln(self.r)
        log_probs += self.r * np.log(self.p) + k_values * np.log(1 - self.p + 1e-300)
        return np.exp(log_probs)
    
    def mean(self) -> float:
        return self.r * (1 - self.p) / self.p
    
    def variance(self) -> float:
        return self.r * (1 - self.p) / (self.p ** 2)
    
    def sample(self, size: int = 1) -> np.ndarray:
        return np.random.negative_binomial(self.r, self.p, size=size)


class HypergeometricDistribution:
    """超几何分布模型"""
    
    def __init__(self, M: int, n: int, N: int):
        self.M = M
        self.n = n
        self.N = N
        self._validate_parameters()
        
    def _validate_parameters(self):
        if self.M < 0 or self.n < 0 or self.N < 0:
            raise ValueError("参数必须非负")
        if self.n > self.M:
            raise ValueError("成功元素数不能超过总体大小")
        if self.N > self.M:
            raise ValueError("抽样数不能超过总体大小")
    
    def pmf(self, k: int) -> float:
        if k < max(0, self.N - self.M + self.n) or k > min(self.n, self.N):
            return 0.0
        
        log_prob = (
            gammaln(self.n + 1) - gammaln(k + 1) - gammaln(self.n - k + 1) +
            gammaln(self.M - self.n + 1) - gammaln(self.N - k + 1) - 
            gammaln(self.M - self.n - self.N + k + 1) -
            (gammaln(self.M + 1) - gammaln(self.N + 1) - gammaln(self.M - self.N + 1))
        )
        
        return np.exp(log_prob)
    
    def log_pmf(self, k: int) -> float:
        if k < max(0, self.N - self.M + self.n) or k > min(self.n, self.N):
            return -np.inf
        
        log_prob = (
            gammaln(self.n + 1) - gammaln(k + 1) - gammaln(self.n - k + 1) +
            gammaln(self.M - self.n + 1) - gammaln(self.N - k + 1) - 
            gammaln(self.M - self.n - self.N + k + 1) -
            (gammaln(self.M + 1) - gammaln(self.N + 1) - gammaln(self.M - self.N + 1))
        )
        
        return log_prob
    
    def pmf_array(self) -> np.ndarray:
        k_min = max(0, self.N - self.M + self.n)
        k_max = min(self.n, self.N)
        k_values = np.arange(k_min, k_max + 1)
        
        log_probs = np.array([self.log_pmf(k) for k in k_values])
        pmf_values = np.zeros(k_max + 1)
        pmf_values[k_min:k_max + 1] = np.exp(log_probs)
        
        return pmf_values
    
    def mean(self) -> float:
        return self.N * self.n / self.M
    
    def variance(self) -> float:
        return self.N * self.n * (self.M - self.n) * (self.M - self.N) / (self.M ** 2 * (self.M - 1))


class MixedDistributionModel:
    """混合概率分布模型"""
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.weights = np.ones(n_components) / n_components
        self.distributions: List[Dict] = []
        self._initialize_distributions()
        
    def _initialize_distributions(self):
        self.distributions = [
            {'type': 'multinomial', 'params': {'n_trials': 10, 'probabilities': np.array([0.3, 0.4, 0.3])}},
            {'type': 'negative_binomial', 'params': {'r': 5.0, 'p': 0.5}},
            {'type': 'hypergeometric', 'params': {'M': 100, 'n': 30, 'N': 20}}
        ]
    
    def _compute_log_prob_multinomial(self, x: np.ndarray, params: Dict) -> float:
        dist = MultinomialDistribution(params['n_trials'], params['probabilities'])
        return dist.log_pmf(x)
    
    def _compute_log_prob_negative_binomial(self, x: int, params: Dict) -> float:
        dist = NegativeBinomialDistribution(params['r'], params['p'])
        return dist.log_pmf(x)
    
    def _compute_log_prob_hypergeometric(self, x: int, params: Dict) -> float:
        dist = HypergeometricDistribution(params['M'], params['n'], params['N'])
        return dist.log_pmf(x)
    
    def mixed_pmf(self, x: Union[np.ndarray, int], distribution_types: List[str] = None) -> float:
        if distribution_types is None:
            distribution_types = [d['type'] for d in self.distributions]
        
        log_probs = []
        for i, dist_info in enumerate(self.distributions):
            if dist_info['type'] not in distribution_types:
                continue
                
            if dist_info['type'] == 'multinomial':
                log_p = self._compute_log_prob_multinomial(x, dist_info['params'])
            elif dist_info['type'] == 'negative_binomial':
                log_p = self._compute_log_prob_negative_binomial(int(x), dist_info['params'])
            elif dist_info['type'] == 'hypergeometric':
                log_p = self._compute_log_prob_hypergeometric(int(x), dist_info['params'])
            else:
                log_p = -np.inf
            
            log_probs.append(np.log(self.weights[i] + 1e-300) + log_p)
        
        if not log_probs:
            return 0.0
        
        return np.exp(logsumexp(np.array(log_probs)))
    
    def mixed_log_pmf(self, x: Union[np.ndarray, int], distribution_types: List[str] = None) -> float:
        if distribution_types is None:
            distribution_types = [d['type'] for d in self.distributions]
        
        log_probs = []
        for i, dist_info in enumerate(self.distributions):
            if dist_info['type'] not in distribution_types:
                continue
                
            if dist_info['type'] == 'multinomial':
                log_p = self._compute_log_prob_multinomial(x, dist_info['params'])
            elif dist_info['type'] == 'negative_binomial':
                log_p = self._compute_log_prob_negative_binomial(int(x), dist_info['params'])
            elif dist_info['type'] == 'hypergeometric':
                log_p = self._compute_log_prob_hypergeometric(int(x), dist_info['params'])
            else:
                log_p = -np.inf
            
            log_probs.append(np.log(self.weights[i] + 1e-300) + log_p)
        
        if not log_probs:
            return -np.inf
        
        return logsumexp(np.array(log_probs))
    
    def update_weights(self, new_weights: np.ndarray):
        new_weights = np.asarray(new_weights, dtype=np.float64)
        new_weights = np.clip(new_weights, 1e-10, 1.0)
        self.weights = new_weights / np.sum(new_weights)
    
    def update_distribution_params(self, component_idx: int, params: Dict):
        if 0 <= component_idx < len(self.distributions):
            self.distributions[component_idx]['params'].update(params)
    
    def sample_mixed(self, n_samples: int = 1) -> List[Tuple[int, str]]:
        samples = []
        component_indices = np.random.choice(
            len(self.weights), size=n_samples, p=self.weights
        )
        
        for idx in component_indices:
            dist_info = self.distributions[idx]
            if dist_info['type'] == 'multinomial':
                dist = MultinomialDistribution(
                    dist_info['params']['n_trials'],
                    dist_info['params']['probabilities']
                )
                sample = tuple(dist.sample(1)[0])
            elif dist_info['type'] == 'negative_binomial':
                dist = NegativeBinomialDistribution(
                    dist_info['params']['r'],
                    dist_info['params']['p']
                )
                sample = int(dist.sample(1)[0])
            elif dist_info['type'] == 'hypergeometric':
                dist = HypergeometricDistribution(
                    dist_info['params']['M'],
                    dist_info['params']['n'],
                    dist_info['params']['N']
                )
                pmf = dist.pmf_array()
                sample = int(np.random.choice(len(pmf), p=pmf / np.sum(pmf)))
            else:
                sample = 0
            
            samples.append((sample, dist_info['type']))
        
        return samples


class KLDivergenceOptimizer:
    """KL散度驱动的分布参数自适应迭代优化器"""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 100,
                 tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.optimization_history: List[Dict] = []
    
    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        p = np.clip(p, 1e-300, 1.0)
        q = np.clip(q, 1e-300, 1.0)
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        kl = np.sum(p * np.log(p / q))
        return max(0.0, kl)
    
    def kl_divergence_multivariate(self, p: np.ndarray, q: np.ndarray) -> float:
        return self.kl_divergence(p, q)
    
    def compute_entropy(self, distribution: np.ndarray) -> float:
        distribution = np.asarray(distribution, dtype=np.float64)
        distribution = np.clip(distribution, 1e-300, 1.0)
        distribution = distribution / np.sum(distribution)
        
        entropy = -np.sum(distribution * np.log(distribution + 1e-300))
        return entropy
    
    def compute_cross_entropy(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        p = np.clip(p, 1e-300, 1.0)
        q = np.clip(q, 1e-300, 1.0)
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return -np.sum(p * np.log(q + 1e-300))
    
    def adaptive_iteration(self, current_weights: np.ndarray, 
                          target_distribution: np.ndarray,
                          sample_distribution: np.ndarray) -> Tuple[np.ndarray, float]:
        current_weights = np.asarray(current_weights, dtype=np.float64)
        target_distribution = np.asarray(target_distribution, dtype=np.float64)
        sample_distribution = np.asarray(sample_distribution, dtype=np.float64)
        
        target_distribution = np.clip(target_distribution, 1e-300, 1.0)
        target_distribution = target_distribution / np.sum(target_distribution)
        
        sample_distribution = np.clip(sample_distribution, 1e-300, 1.0)
        sample_distribution = sample_distribution / np.sum(sample_distribution)
        
        kl_target_sample = self.kl_divergence(target_distribution, sample_distribution)
        
        target_entropy = self.compute_entropy(target_distribution)
        sample_entropy = self.compute_entropy(sample_distribution)
        
        entropy_adjustment = np.sign(sample_entropy - target_entropy) * 0.1
        
        distribution_diff = np.sum(np.abs(target_distribution - sample_distribution))
        weight_adjustment = distribution_diff * self.learning_rate
        
        n_weights = len(current_weights)
        adjustment = np.random.uniform(-weight_adjustment, weight_adjustment, n_weights)
        adjustment = adjustment - np.mean(adjustment)
        
        new_weights = current_weights + adjustment
        new_weights = np.clip(new_weights, 1e-10, 1.0)
        new_weights = new_weights / np.sum(new_weights)
        
        return new_weights, kl_target_sample
    
    def optimize_distribution_params(self, 
                                     mixed_model: MixedDistributionModel,
                                     observed_distribution: np.ndarray,
                                     n_categories: int) -> Dict[str, Any]:
        observed = np.asarray(observed_distribution, dtype=np.float64)
        observed = np.clip(observed, 1e-300, 1.0)
        observed = observed / np.sum(observed)
        
        best_kl = float('inf')
        best_weights = mixed_model.weights.copy()
        iteration_records = []
        
        for iteration in range(self.max_iterations):
            current_dist = np.zeros(n_categories)
            
            for i, dist_info in enumerate(mixed_model.distributions):
                if dist_info['type'] == 'multinomial':
                    probs = dist_info['params']['probabilities']
                    if len(probs) >= n_categories:
                        current_dist += mixed_model.weights[i] * probs[:n_categories]
                elif dist_info['type'] == 'negative_binomial':
                    dist = NegativeBinomialDistribution(
                        dist_info['params']['r'],
                        dist_info['params']['p']
                    )
                    pmf = dist.pmf_array(n_categories - 1)
                    if len(pmf) < n_categories:
                        pmf = np.pad(pmf, (0, n_categories - len(pmf)))
                    current_dist += mixed_model.weights[i] * pmf[:n_categories]
                elif dist_info['type'] == 'hypergeometric':
                    dist = HypergeometricDistribution(
                        dist_info['params']['M'],
                        dist_info['params']['n'],
                        dist_info['params']['N']
                    )
                    pmf = dist.pmf_array()
                    if len(pmf) < n_categories:
                        pmf = np.pad(pmf, (0, n_categories - len(pmf)))
                    current_dist += mixed_model.weights[i] * pmf[:n_categories]
            
            current_dist = np.clip(current_dist, 1e-300, 1.0)
            current_dist = current_dist / np.sum(current_dist)
            
            kl = self.kl_divergence(observed, current_dist)
            
            iteration_records.append({
                'iteration': iteration,
                'kl_divergence': kl,
                'weights': mixed_model.weights.copy(),
                'current_distribution': current_dist.copy()
            })
            
            if kl < best_kl:
                best_kl = kl
                best_weights = mixed_model.weights.copy()
            
            if kl < self.tolerance:
                break
            
            new_weights, _ = self.adaptive_iteration(
                mixed_model.weights, observed, current_dist
            )
            mixed_model.update_weights(new_weights)
        
        mixed_model.update_weights(best_weights)
        
        result = {
            'final_kl_divergence': best_kl,
            'iterations': len(iteration_records),
            'best_weights': best_weights,
            'converged': best_kl < self.tolerance,
            'history': iteration_records
        }
        
        self.optimization_history.append(result)
        return result
    
    def entropy_driven_adjustment(self, 
                                  current_params: Dict[str, float],
                                  observed_entropy: float,
                                  target_entropy: float) -> Dict[str, float]:
        entropy_diff = observed_entropy - target_entropy
        
        adjustment_factor = np.tanh(entropy_diff * 0.5)
        
        new_params = {}
        for key, value in current_params.items():
            if key == 'p':
                new_value = value * (1 - adjustment_factor * 0.1)
                new_value = np.clip(new_value, 0.01, 0.99)
            elif key == 'r':
                new_value = value * (1 + adjustment_factor * 0.1)
                new_value = max(0.1, new_value)
            elif key in ['n_trials', 'M', 'n', 'N']:
                new_value = int(max(1, value * (1 + adjustment_factor * 0.05)))
            else:
                new_value = value
            
            new_params[key] = new_value
        
        return new_params


class ConvolutionProbabilityConstraint:
    """卷积运算融合难度/知识点/题型的联合概率约束"""
    
    def __init__(self, n_difficulty_levels: int = 5, 
                 n_knowledge_points: int = 10,
                 n_question_types: int = 4):
        self.n_difficulty = n_difficulty_levels
        self.n_knowledge = n_knowledge_points
        self.n_types = n_question_types
        
        self.difficulty_dist = np.ones(n_difficulty_levels) / n_difficulty_levels
        self.knowledge_dist = np.ones(n_knowledge_points) / n_knowledge_points
        self.type_dist = np.ones(n_question_types) / n_question_types
        
        self.joint_distribution: Optional[np.ndarray] = None
        self._build_joint_distribution()
    
    def _build_joint_distribution(self):
        outer = np.outer(self.difficulty_dist, self.knowledge_dist)
        self.joint_distribution = np.outer(outer.flatten(), self.type_dist)
        self.joint_distribution = self.joint_distribution.reshape(
            self.n_difficulty, self.n_knowledge, self.n_types
        )
        self._normalize_joint_distribution()
    
    def _normalize_joint_distribution(self):
        total = np.sum(self.joint_distribution)
        if total > 0:
            self.joint_distribution = self.joint_distribution / total
        else:
            self.joint_distribution = np.ones(
                (self.n_difficulty, self.n_knowledge, self.n_types)
            ) / (self.n_difficulty * self.n_knowledge * self.n_types)
    
    def set_difficulty_distribution(self, dist: np.ndarray):
        dist = np.asarray(dist, dtype=np.float64)
        dist = np.clip(dist, 1e-300, 1.0)
        self.difficulty_dist = dist / np.sum(dist)
        self._build_joint_distribution()
    
    def set_knowledge_distribution(self, dist: np.ndarray):
        dist = np.asarray(dist, dtype=np.float64)
        dist = np.clip(dist, 1e-300, 1.0)
        self.knowledge_dist = dist / np.sum(dist)
        self._build_joint_distribution()
    
    def set_type_distribution(self, dist: np.ndarray):
        dist = np.asarray(dist, dtype=np.float64)
        dist = np.clip(dist, 1e-300, 1.0)
        self.type_dist = dist / np.sum(dist)
        self._build_joint_distribution()
    
    def discrete_convolution_1d(self, dist1: np.ndarray, dist2: np.ndarray) -> np.ndarray:
        dist1 = np.asarray(dist1, dtype=np.float64)
        dist2 = np.asarray(dist2, dtype=np.float64)
        
        dist1 = np.clip(dist1, 1e-300, 1.0)
        dist2 = np.clip(dist2, 1e-300, 1.0)
        dist1 = dist1 / np.sum(dist1)
        dist2 = dist2 / np.sum(dist2)
        
        result = np.convolve(dist1, dist2, mode='full')
        result = np.clip(result, 1e-300, 1.0)
        return result / np.sum(result)
    
    def discrete_convolution_2d(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        mat1 = np.asarray(mat1, dtype=np.float64)
        mat2 = np.asarray(mat2, dtype=np.float64)
        
        mat1 = np.clip(mat1, 1e-300, 1.0)
        mat2 = np.clip(mat2, 1e-300, 1.0)
        mat1 = mat1 / np.sum(mat1)
        mat2 = mat2 / np.sum(mat2)
        
        from scipy.signal import convolve2d
        result = convolve2d(mat1, mat2, mode='full')
        result = np.clip(result, 1e-300, 1.0)
        return result / np.sum(result)
    
    def compute_joint_probability(self, 
                                  difficulty: int, 
                                  knowledge: int, 
                                  qtype: int) -> float:
        if not (0 <= difficulty < self.n_difficulty and
                0 <= knowledge < self.n_knowledge and
                0 <= qtype < self.n_types):
            return 0.0
        
        return self.joint_distribution[difficulty, knowledge, qtype]
    
    def compute_marginal_difficulty(self) -> np.ndarray:
        return np.sum(self.joint_distribution, axis=(1, 2))
    
    def compute_marginal_knowledge(self) -> np.ndarray:
        return np.sum(self.joint_distribution, axis=(0, 2))
    
    def compute_marginal_type(self) -> np.ndarray:
        return np.sum(self.joint_distribution, axis=(0, 1))
    
    def compute_conditional_probability(self,
                                        condition_on: str,
                                        condition_value: int,
                                        target: str) -> np.ndarray:
        if condition_on == 'difficulty':
            marginal = np.sum(self.joint_distribution[condition_value, :, :])
            if marginal < 1e-300:
                return np.zeros(self.n_knowledge if target == 'knowledge' else self.n_types)
            if target == 'knowledge':
                return np.sum(self.joint_distribution[condition_value, :, :], axis=1) / marginal
            elif target == 'type':
                return np.sum(self.joint_distribution[condition_value, :, :], axis=0) / marginal
        
        elif condition_on == 'knowledge':
            marginal = np.sum(self.joint_distribution[:, condition_value, :])
            if marginal < 1e-300:
                return np.zeros(self.n_difficulty if target == 'difficulty' else self.n_types)
            if target == 'difficulty':
                return np.sum(self.joint_distribution[:, condition_value, :], axis=1) / marginal
            elif target == 'type':
                return np.sum(self.joint_distribution[:, condition_value, :], axis=0) / marginal
        
        elif condition_on == 'type':
            marginal = np.sum(self.joint_distribution[:, :, condition_value])
            if marginal < 1e-300:
                return np.zeros(self.n_difficulty if target == 'difficulty' else self.n_knowledge)
            if target == 'difficulty':
                return np.sum(self.joint_distribution[:, :, condition_value], axis=1) / marginal
            elif target == 'knowledge':
                return np.sum(self.joint_distribution[:, :, condition_value], axis=0) / marginal
        
        return np.array([])
    
    def apply_constraint_convolution(self,
                                     constraint_matrix: np.ndarray,
                                     constraint_type: str = 'additive') -> np.ndarray:
        constraint_matrix = np.asarray(constraint_matrix, dtype=np.float64)
        constraint_matrix = np.clip(constraint_matrix, 0.0, 1.0)
        
        if constraint_type == 'additive':
            result = self.joint_distribution + constraint_matrix
        elif constraint_type == 'multiplicative':
            result = self.joint_distribution * (1 + constraint_matrix)
        elif constraint_type == 'convolution':
            if constraint_matrix.ndim == 1:
                result = np.zeros_like(self.joint_distribution)
                for i in range(self.n_types):
                    result[:, :, i] = self.discrete_convolution_1d(
                        self.joint_distribution[:, :, i].flatten(),
                        constraint_matrix
                    ).reshape(self.n_difficulty, self.n_knowledge)
            else:
                result = np.zeros_like(self.joint_distribution)
                for i in range(self.n_types):
                    result[:, :, i] = self.discrete_convolution_2d(
                        self.joint_distribution[:, :, i],
                        constraint_matrix
                    )
        else:
            result = self.joint_distribution.copy()
        
        result = np.clip(result, 1e-300, 1.0)
        return result / np.sum(result)
    
    def update_from_observed_data(self, 
                                  observations: List[Tuple[int, int, int]],
                                  smoothing: float = 0.1):
        counts = np.zeros((self.n_difficulty, self.n_knowledge, self.n_types))
        
        for diff, know, qtype in observations:
            if (0 <= diff < self.n_difficulty and
                0 <= know < self.n_knowledge and
                0 <= qtype < self.n_types):
                counts[diff, know, qtype] += 1
        
        counts += smoothing
        
        self.joint_distribution = counts / np.sum(counts)
        
        self.difficulty_dist = self.compute_marginal_difficulty()
        self.knowledge_dist = self.compute_marginal_knowledge()
        self.type_dist = self.compute_marginal_type()
    
    def sample_joint(self, n_samples: int = 1) -> List[Tuple[int, int, int]]:
        flat_dist = self.joint_distribution.flatten()
        flat_dist = np.clip(flat_dist, 1e-300, 1.0)
        flat_dist = flat_dist / np.sum(flat_dist)
        
        indices = np.random.choice(len(flat_dist), size=n_samples, p=flat_dist)
        
        samples = []
        for idx in indices:
            diff = idx // (self.n_knowledge * self.n_types)
            remaining = idx % (self.n_knowledge * self.n_types)
            know = remaining // self.n_types
            qtype = remaining % self.n_types
            samples.append((diff, know, qtype))
        
        return samples
    
    def compute_mutual_information(self, var1: str, var2: str) -> float:
        if var1 == 'difficulty' and var2 == 'knowledge':
            joint = np.sum(self.joint_distribution, axis=2)
            marg1 = self.difficulty_dist
            marg2 = self.knowledge_dist
        elif var1 == 'difficulty' and var2 == 'type':
            joint = np.sum(self.joint_distribution, axis=1)
            marg1 = self.difficulty_dist
            marg2 = self.type_dist
        elif var1 == 'knowledge' and var2 == 'type':
            joint = np.sum(self.joint_distribution, axis=0)
            marg1 = self.knowledge_dist
            marg2 = self.type_dist
        else:
            return 0.0
        
        joint = np.clip(joint, 1e-300, 1.0)
        marg1 = np.clip(marg1, 1e-300, 1.0)
        marg2 = np.clip(marg2, 1e-300, 1.0)
        
        outer = np.outer(marg1, marg2)
        outer = np.clip(outer, 1e-300, 1.0)
        
        mi = np.sum(joint * np.log(joint / outer))
        return max(0.0, mi)


class AdvancedProbabilityModel:
    """高阶概率分布混合模型整合类"""
    
    def __init__(self, 
                 n_difficulty_levels: int = 5,
                 n_knowledge_points: int = 10,
                 n_question_types: int = 4):
        self.mixed_distribution = MixedDistributionModel(n_components=3)
        self.kl_optimizer = KLDivergenceOptimizer()
        self.convolution_constraint = ConvolutionProbabilityConstraint(
            n_difficulty_levels, n_knowledge_points, n_question_types
        )
        
        self.n_difficulty = n_difficulty_levels
        self.n_knowledge = n_knowledge_points
        self.n_types = n_question_types
        
        self.adaptation_history: List[Dict] = []
    
    def compute_mixed_probability(self, 
                                  x: Union[np.ndarray, int],
                                  distribution_types: List[str] = None) -> float:
        return self.mixed_distribution.mixed_pmf(x, distribution_types)
    
    def adapt_to_sample_distribution(self,
                                     observed_counts: np.ndarray,
                                     max_iterations: int = 50) -> Dict[str, Any]:
        observed = np.asarray(observed_counts, dtype=np.float64)
        observed = np.clip(observed, 1e-300, 1.0)
        observed = observed / np.sum(observed)
        
        n_categories = len(observed)
        
        self.kl_optimizer.max_iterations = max_iterations
        result = self.kl_optimizer.optimize_distribution_params(
            self.mixed_distribution, observed, n_categories
        )
        
        self.adaptation_history.append({
            'type': 'distribution_adaptation',
            'observed_distribution': observed.copy(),
            'result': result
        })
        
        return result
    
    def update_joint_constraints(self,
                                 difficulty_dist: np.ndarray = None,
                                 knowledge_dist: np.ndarray = None,
                                 type_dist: np.ndarray = None):
        if difficulty_dist is not None:
            self.convolution_constraint.set_difficulty_distribution(difficulty_dist)
        if knowledge_dist is not None:
            self.convolution_constraint.set_knowledge_distribution(knowledge_dist)
        if type_dist is not None:
            self.convolution_constraint.set_type_distribution(type_dist)
    
    def compute_question_probability(self,
                                     difficulty: int,
                                     knowledge: int,
                                     qtype: int) -> float:
        joint_prob = self.convolution_constraint.compute_joint_probability(
            difficulty, knowledge, qtype
        )
        
        mixed_prob = self.mixed_distribution.mixed_pmf(difficulty)
        
        combined_prob = 0.7 * joint_prob + 0.3 * mixed_prob
        return np.clip(combined_prob, 1e-300, 1.0)
    
    def sample_questions(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        joint_samples = self.convolution_constraint.sample_joint(n_samples)
        
        results = []
        for diff, know, qtype in joint_samples:
            prob = self.compute_question_probability(diff, know, qtype)
            results.append({
                'difficulty': diff,
                'knowledge_point': know,
                'question_type': qtype,
                'probability': prob
            })
        
        return results
    
    def get_distribution_statistics(self) -> Dict[str, Any]:
        return {
            'mixed_weights': self.mixed_distribution.weights.copy(),
            'difficulty_marginal': self.convolution_constraint.compute_marginal_difficulty(),
            'knowledge_marginal': self.convolution_constraint.compute_marginal_knowledge(),
            'type_marginal': self.convolution_constraint.compute_marginal_type(),
            'mutual_information_diff_know': self.convolution_constraint.compute_mutual_information(
                'difficulty', 'knowledge'
            ),
            'mutual_information_diff_type': self.convolution_constraint.compute_mutual_information(
                'difficulty', 'type'
            ),
            'mutual_information_know_type': self.convolution_constraint.compute_mutual_information(
                'knowledge', 'type'
            )
        }


def validate_probability_models():
    print("=" * 60)
    print("高阶概率分布混合模型验证")
    print("=" * 60)
    
    print("\n1. 多项分布验证")
    multi_dist = MultinomialDistribution(10, np.array([0.3, 0.4, 0.3]))
    counts = np.array([3, 4, 3])
    print(f"   P(X=[3,4,3]) = {multi_dist.pmf(counts):.6f}")
    print(f"   期望: {multi_dist.mean()}")
    print(f"   方差: {multi_dist.variance()}")
    
    print("\n2. 负二项分布验证")
    nb_dist = NegativeBinomialDistribution(r=5.0, p=0.5)
    print(f"   P(X=3) = {nb_dist.pmf(3):.6f}")
    print(f"   期望: {nb_dist.mean():.4f}")
    print(f"   方差: {nb_dist.variance():.4f}")
    
    print("\n3. 超几何分布验证")
    hg_dist = HypergeometricDistribution(M=100, n=30, N=20)
    print(f"   P(X=5) = {hg_dist.pmf(5):.6f}")
    print(f"   期望: {hg_dist.mean():.4f}")
    print(f"   方差: {hg_dist.variance():.4f}")
    
    print("\n4. 混合分布验证")
    mixed = MixedDistributionModel(n_components=3)
    print(f"   混合权重: {mixed.weights}")
    print(f"   混合PMF(3): {mixed.mixed_pmf(3):.6f}")
    samples = mixed.sample_mixed(5)
    print(f"   采样结果: {samples[:3]}...")
    
    print("\n5. KL散度优化验证")
    kl_opt = KLDivergenceOptimizer(learning_rate=0.1, max_iterations=20)
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    kl = kl_opt.kl_divergence(p, q)
    print(f"   KL(P||Q) = {kl:.6f}")
    print(f"   熵(P) = {kl_opt.compute_entropy(p):.6f}")
    
    print("\n6. 卷积概率约束验证")
    conv = ConvolutionProbabilityConstraint(n_difficulty_levels=5, n_knowledge_points=10, n_question_types=4)
    joint_prob = conv.compute_joint_probability(2, 5, 1)
    print(f"   联合概率 P(D=2, K=5, T=1) = {joint_prob:.6f}")
    print(f"   难度边缘分布: {conv.compute_marginal_difficulty()}")
    print(f"   D-K互信息: {conv.compute_mutual_information('difficulty', 'knowledge'):.6f}")
    
    print("\n7. 高阶概率模型整合验证")
    adv_model = AdvancedProbabilityModel()
    observed = np.array([10, 20, 15, 25, 30])
    result = adv_model.adapt_to_sample_distribution(observed, max_iterations=10)
    print(f"   最终KL散度: {result['final_kl_divergence']:.6f}")
    print(f"   收敛状态: {result['converged']}")
    
    samples = adv_model.sample_questions(3)
    print(f"   采样题目: {samples}")
    
    stats = adv_model.get_distribution_statistics()
    print(f"   D-K互信息: {stats['mutual_information_diff_know']:.6f}")
    
    print("\n" + "=" * 60)
    print("所有验证完成!")
    print("=" * 60)


class ThirdOrderHMM:
    """
    三阶隐马尔可夫模型
    状态转移依赖于前三个状态，实现更复杂的题目调度模式
    """
    
    def __init__(self, n_states: int, n_observations: int):
        self.n_states = n_states
        self.n_observations = n_observations
        
        self.transition_matrix = self._init_third_order_transition()
        self.emission_matrix = self._init_emission_matrix()
        self.initial_probs = self._init_initial_probs()
        
        self.state_history: List[int] = []
        self.observation_history: List[int] = []
        
    def _init_third_order_transition(self) -> np.ndarray:
        shape = (self.n_states, self.n_states, self.n_states, self.n_states)
        matrix = np.random.dirichlet(np.ones(self.n_states), size=(self.n_states, self.n_states, self.n_states))
        return matrix.astype(np.float64)
    
    def _init_emission_matrix(self) -> np.ndarray:
        matrix = np.random.dirichlet(np.ones(self.n_observations), size=self.n_states)
        return matrix.astype(np.float64)
    
    def _init_initial_probs(self) -> np.ndarray:
        probs = np.random.dirichlet(np.ones(self.n_states))
        return probs.astype(np.float64)
    
    def set_transition_matrix(self, matrix: np.ndarray):
        if matrix.shape != (self.n_states, self.n_states, self.n_states, self.n_states):
            raise ValueError(f"Transition matrix shape must be ({self.n_states}, {self.n_states}, {self.n_states}, {self.n_states})")
        self.transition_matrix = matrix.copy()
        
    def set_emission_matrix(self, matrix: np.ndarray):
        if matrix.shape != (self.n_states, self.n_observations):
            raise ValueError(f"Emission matrix shape must be ({self.n_states}, {self.n_observations})")
        self.emission_matrix = matrix.copy()
        
    def normalize_transition_matrix(self):
        for i in range(self.n_states):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    total = np.sum(self.transition_matrix[i, j, k, :])
                    if total > 0:
                        self.transition_matrix[i, j, k, :] /= total
                        
    def get_transition_probability(self, s1: int, s2: int, s3: int, s4: int) -> float:
        return self.transition_matrix[s1, s2, s3, s4]
    
    def sample_next_state(self, s1: int, s2: int, s3: int) -> int:
        probs = self.transition_matrix[s1, s2, s3, :]
        return np.random.choice(self.n_states, p=probs)
    
    def sample_observation(self, state: int) -> int:
        probs = self.emission_matrix[state, :]
        return np.random.choice(self.n_observations, p=probs)
    
    def initialize_sequence(self) -> Tuple[int, int, int]:
        first = np.random.choice(self.n_states, p=self.initial_probs)
        second = np.random.choice(self.n_states, p=self.transition_matrix[0, 0, first, :])
        third = np.random.choice(self.n_states, p=self.transition_matrix[0, first, second, :])
        
        self.state_history = [first, second, third]
        return first, second, third
    
    def generate_sequence(self, length: int) -> Tuple[List[int], List[int]]:
        if len(self.state_history) < 3:
            self.initialize_sequence()
        
        states = self.state_history[-3:].copy()
        observations = []
        
        for _ in range(length):
            s1, s2, s3 = states[-3], states[-2], states[-1]
            next_state = self.sample_next_state(s1, s2, s3)
            states.append(next_state)
            
            observation = self.sample_observation(next_state)
            observations.append(observation)
        
        self.state_history = states
        self.observation_history.extend(observations)
        
        return states, observations
    
    def forward_algorithm(self, observations: List[int]) -> np.ndarray:
        T = len(observations)
        alpha = np.zeros((T, self.n_states, self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    alpha[0, i, j, k] = (self.initial_probs[i] * 
                                         self.transition_matrix[0, 0, i, j] * 
                                         self.transition_matrix[0, i, j, k] * 
                                         self.emission_matrix[k, observations[0]])
        
        for t in range(1, T):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    for k in range(self.n_states):
                        for l in range(self.n_states):
                            alpha[t, i, j, k] += (alpha[t-1, l, i, j] * 
                                                  self.transition_matrix[l, i, j, k] * 
                                                  self.emission_matrix[k, observations[t]])
        
        return alpha
    
    def viterbi_algorithm(self, observations: List[int]) -> List[int]:
        T = len(observations)
        delta = np.zeros((T, self.n_states, self.n_states, self.n_states))
        psi = np.zeros((T, self.n_states, self.n_states, self.n_states), dtype=int)
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    delta[0, i, j, k] = (self.initial_probs[i] * 
                                         self.transition_matrix[0, 0, i, j] * 
                                         self.transition_matrix[0, i, j, k] * 
                                         self.emission_matrix[k, observations[0]])
        
        for t in range(1, T):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    for k in range(self.n_states):
                        max_prob = 0.0
                        best_l = 0
                        for l in range(self.n_states):
                            prob = (delta[t-1, l, i, j] * 
                                   self.transition_matrix[l, i, j, k] * 
                                   self.emission_matrix[k, observations[t]])
                            if prob > max_prob:
                                max_prob = prob
                                best_l = l
                        delta[t, i, j, k] = max_prob
                        psi[t, i, j, k] = best_l
        
        path = np.zeros(T, dtype=int)
        max_prob = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    if delta[T-1, i, j, k] > max_prob:
                        max_prob = delta[T-1, i, j, k]
                        path[T-1] = k
        
        return path.tolist()
    
    def baum_welch_step(self, observations: List[int], learning_rate: float = 0.1):
        T = len(observations)
        alpha = self.forward_algorithm(observations)
        
        beta = np.zeros((T, self.n_states, self.n_states, self.n_states))
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    beta[T-1, i, j, k] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    for k in range(self.n_states):
                        for l in range(self.n_states):
                            beta[t, i, j, k] += (beta[t+1, i, j, l] * 
                                                 self.transition_matrix[i, j, k, l] * 
                                                 self.emission_matrix[l, observations[t+1]])
        
        xi = np.zeros((T-1, self.n_states, self.n_states, self.n_states, self.n_states))
        for t in range(T-1):
            denominator = 0.0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    for k in range(self.n_states):
                        for l in range(self.n_states):
                            xi[t, i, j, k, l] = (alpha[t, i, j, k] * 
                                                 self.transition_matrix[i, j, k, l] * 
                                                 self.emission_matrix[l, observations[t+1]] * 
                                                 beta[t+1, j, k, l])
                            denominator += xi[t, i, j, k, l]
            
            if denominator > 0:
                xi[t, :, :, :, :] /= denominator
        
        new_transition = self.transition_matrix.copy()
        for i in range(self.n_states):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    for l in range(self.n_states):
                        xi_sum = np.sum(xi[:, i, j, k, l])
                        gamma_sum = np.sum(xi[:, i, j, k, :])
                        if gamma_sum > 0:
                            new_transition[i, j, k, l] = xi_sum / gamma_sum
        
        self.transition_matrix = ((1 - learning_rate) * self.transition_matrix + 
                                  learning_rate * new_transition)
        self.normalize_transition_matrix()
        
    def get_state_distribution(self) -> np.ndarray:
        if len(self.state_history) < 3:
            return self.initial_probs.copy()
        
        s1, s2, s3 = self.state_history[-3], self.state_history[-2], self.state_history[-1]
        return self.transition_matrix[s1, s2, s3, :]


class AdaptivePoissonProcess:
    """
    自适应泊松过程
    实现题目抽取间隔的随机化，到达率根据历史数据自适应调整
    """
    
    def __init__(self, initial_rate: float = 1.0, 
                 min_rate: float = 0.1, 
                 max_rate: float = 10.0,
                 adaptation_rate: float = 0.1):
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_rate = adaptation_rate
        
        self.event_times: List[float] = []
        self.intervals: List[float] = []
        self.rate_history: List[Tuple[float, float]] = []
        
        self._time = 0.0
        self._event_count = 0
        
    def set_rate(self, rate: float):
        self.current_rate = np.clip(rate, self.min_rate, self.max_rate)
        
    def sample_interval(self) -> float:
        interval = np.random.exponential(1.0 / self.current_rate)
        return interval
    
    def generate_event(self) -> float:
        interval = self.sample_interval()
        self._time += interval
        self._event_count += 1
        
        self.event_times.append(self._time)
        self.intervals.append(interval)
        self.rate_history.append((self._time, self.current_rate))
        
        return self._time
    
    def generate_multiple_events(self, n: int) -> List[float]:
        events = []
        for _ in range(n):
            events.append(self.generate_event())
        return events
    
    def adapt_rate(self, feedback: float):
        adjustment = self.adaptation_rate * feedback
        new_rate = self.current_rate * np.exp(adjustment)
        self.current_rate = np.clip(new_rate, self.min_rate, self.max_rate)
        
    def adapt_by_performance(self, correct_rate: float, target_rate: float = 0.7):
        error = target_rate - correct_rate
        self.adapt_rate(error)
        
    def adapt_by_intensity(self, recent_events: int, target_events: int):
        if recent_events > target_events:
            self.current_rate = np.clip(
                self.current_rate * (target_events / recent_events),
                self.min_rate, self.max_rate
            )
        elif recent_events < target_events * 0.5:
            self.current_rate = np.clip(
                self.current_rate * 1.2,
                self.min_rate, self.max_rate
            )
            
    def estimate_rate_from_history(self, window_size: int = 10) -> float:
        if len(self.intervals) < 2:
            return self.initial_rate
        
        recent_intervals = self.intervals[-window_size:]
        mean_interval = np.mean(recent_intervals)
        
        if mean_interval > 0:
            return 1.0 / mean_interval
        return self.current_rate
    
    def compound_poisson_sample(self, jump_distribution: str = 'exponential',
                                jump_params: Dict = None) -> Tuple[float, float]:
        interval = self.sample_interval()
        
        if jump_distribution == 'exponential':
            scale = jump_params.get('scale', 1.0) if jump_params else 1.0
            jump = np.random.exponential(scale)
        elif jump_distribution == 'gamma':
            shape = jump_params.get('shape', 1.0) if jump_params else 1.0
            scale = jump_params.get('scale', 1.0) if jump_params else 1.0
            jump = np.random.gamma(shape, scale)
        elif jump_distribution == 'normal':
            mean = jump_params.get('mean', 0.0) if jump_params else 0.0
            std = jump_params.get('std', 1.0) if jump_params else 1.0
            jump = abs(np.random.normal(mean, std))
        else:
            jump = 1.0
            
        return interval, jump
    
    def non_homogeneous_intensity(self, t: float, 
                                  base_rate: float = None,
                                  periodic_amplitude: float = 0.3,
                                  periodic_frequency: float = 0.1) -> float:
        rate = base_rate if base_rate is not None else self.current_rate
        periodic_factor = 1 + periodic_amplitude * np.sin(2 * np.pi * periodic_frequency * t)
        return rate * periodic_factor
    
    def thinning_sample(self, max_rate: float = None) -> float:
        if max_rate is None:
            max_rate = self.max_rate
            
        while True:
            candidate_time = self._time + np.random.exponential(1.0 / max_rate)
            intensity = self.non_homogeneous_intensity(candidate_time)
            
            if np.random.random() < intensity / max_rate:
                self._time = candidate_time
                self.event_times.append(self._time)
                return self._time
    
    def get_statistics(self) -> Dict[str, float]:
        if not self.intervals:
            return {
                'mean_interval': 0.0,
                'std_interval': 0.0,
                'current_rate': self.current_rate,
                'total_events': 0,
                'estimated_rate': self.initial_rate
            }
        
        return {
            'mean_interval': np.mean(self.intervals),
            'std_interval': np.std(self.intervals),
            'current_rate': self.current_rate,
            'total_events': self._event_count,
            'estimated_rate': self.estimate_rate_from_history(),
            'total_time': self._time
        }
    
    def reset(self, keep_rate: bool = False):
        self.event_times = []
        self.intervals = []
        self.rate_history = []
        self._time = 0.0
        self._event_count = 0
        if not keep_rate:
            self.current_rate = self.initial_rate


class WienerProcessPerturbation:
    """
    维纳过程随机扰动
    实现题目权重的连续波动
    """
    
    def __init__(self, n_weights: int,
                 drift: float = 0.0,
                 volatility: float = 0.1,
                 mean_reversion: float = 0.5,
                 target_mean: float = 1.0):
        self.n_weights = n_weights
        self.drift = drift
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        self.target_mean = target_mean
        
        self.weights = np.ones(n_weights, dtype=np.float64)
        self.time = 0.0
        
        self.weight_history: List[Tuple[float, np.ndarray]] = [(0.0, self.weights.copy())]
        self.perturbation_history: List[np.ndarray] = []
        
    def initialize_weights(self, initial_weights: np.ndarray):
        if len(initial_weights) != self.n_weights:
            raise ValueError(f"Initial weights length {len(initial_weights)} must match n_weights {self.n_weights}")
        self.weights = initial_weights.astype(np.float64).copy()
        self.weight_history = [(0.0, self.weights.copy())]
        
    def standard_wiener_increment(self, dt: float) -> np.ndarray:
        return np.random.normal(0, np.sqrt(dt), self.n_weights)
    
    def geometric_brownian_step(self, dt: float) -> np.ndarray:
        dW = self.standard_wiener_increment(dt)
        
        d_weights = (self.drift * self.weights * dt + 
                    self.volatility * self.weights * dW)
        
        self.weights = self.weights + d_weights
        self.weights = np.maximum(self.weights, 0.01)
        
        return self.weights.copy()
    
    def ornstein_uhlenbeck_step(self, dt: float) -> np.ndarray:
        dW = self.standard_wiener_increment(dt)
        
        d_weights = (self.mean_reversion * (self.target_mean - self.weights) * dt + 
                    self.volatility * dW)
        
        self.weights = self.weights + d_weights
        self.weights = np.maximum(self.weights, 0.01)
        
        return self.weights.copy()
    
    def mean_reverting_step(self, dt: float, 
                           reversion_speed: float = None,
                           long_term_mean: np.ndarray = None) -> np.ndarray:
        if reversion_speed is None:
            reversion_speed = self.mean_reversion
        if long_term_mean is None:
            long_term_mean = np.ones(self.n_weights) * self.target_mean
            
        dW = self.standard_wiener_increment(dt)
        
        d_weights = (reversion_speed * (long_term_mean - self.weights) * dt + 
                    self.volatility * dW)
        
        self.weights = self.weights + d_weights
        self.weights = np.clip(self.weights, 0.01, 10.0)
        
        return self.weights.copy()
    
    def jump_diffusion_step(self, dt: float, 
                           jump_intensity: float = 0.1,
                           jump_mean: float = 0.0,
                           jump_std: float = 0.2) -> np.ndarray:
        dW = self.standard_wiener_increment(dt)
        
        d_weights = self.drift * self.weights * dt + self.volatility * self.weights * dW
        
        n_jumps = np.random.poisson(jump_intensity * dt)
        for _ in range(n_jumps):
            jump_size = np.random.normal(jump_mean, jump_std, self.n_weights)
            d_weights += self.weights * jump_size
        
        self.weights = self.weights + d_weights
        self.weights = np.maximum(self.weights, 0.01)
        
        return self.weights.copy()
    
    def correlated_perturbation(self, dt: float, 
                               correlation_matrix: np.ndarray = None) -> np.ndarray:
        if correlation_matrix is None:
            correlation_matrix = np.eye(self.n_weights)
            
        L = np.linalg.cholesky(correlation_matrix)
        
        independent_increments = np.random.normal(0, np.sqrt(dt), self.n_weights)
        correlated_increments = L @ independent_increments
        
        d_weights = (self.drift * self.weights * dt + 
                    self.volatility * self.weights * correlated_increments)
        
        self.weights = self.weights + d_weights
        self.weights = np.maximum(self.weights, 0.01)
        
        return self.weights.copy()
    
    def multi_factor_perturbation(self, dt: float, 
                                  factors: np.ndarray,
                                  factor_loadings: np.ndarray) -> np.ndarray:
        n_factors = len(factors)
        
        dW_factors = np.random.normal(0, np.sqrt(dt), n_factors)
        
        idiosyncratic = np.random.normal(0, np.sqrt(dt), self.n_weights)
        
        systematic_perturbation = factor_loadings @ dW_factors
        idiosyncratic_perturbation = 0.3 * idiosyncratic
        
        d_weights = (self.drift * self.weights * dt + 
                    self.volatility * self.weights * 
                    (systematic_perturbation + idiosyncratic_perturbation))
        
        self.weights = self.weights + d_weights
        self.weights = np.maximum(self.weights, 0.01)
        
        return self.weights.copy()
    
    def evolve(self, dt: float, method: str = 'ornstein_uhlenbeck', **kwargs) -> np.ndarray:
        self.time += dt
        
        if method == 'standard':
            dW = self.standard_wiener_increment(dt)
            self.weights = self.weights + self.volatility * dW
        elif method == 'geometric':
            self.geometric_brownian_step(dt)
        elif method == 'ornstein_uhlenbeck':
            self.ornstein_uhlenbeck_step(dt)
        elif method == 'mean_reverting':
            self.mean_reverting_step(dt, **kwargs)
        elif method == 'jump_diffusion':
            self.jump_diffusion_step(dt, **kwargs)
        elif method == 'correlated':
            self.correlated_perturbation(dt, **kwargs)
        elif method == 'multi_factor':
            self.multi_factor_perturbation(dt, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.weight_history.append((self.time, self.weights.copy()))
        
        return self.weights.copy()
    
    def get_normalized_weights(self) -> np.ndarray:
        total = np.sum(self.weights)
        if total > 0:
            return self.weights / total
        return np.ones(self.n_weights) / self.n_weights
    
    def get_softmax_weights(self, temperature: float = 1.0) -> np.ndarray:
        exp_weights = np.exp(self.weights / temperature)
        return exp_weights / np.sum(exp_weights)
    
    def apply_constraint(self, min_weight: float = 0.01, max_weight: float = 5.0):
        self.weights = np.clip(self.weights, min_weight, max_weight)
        
    def reset_weights(self, value: float = 1.0):
        self.weights = np.ones(self.n_weights, dtype=np.float64) * value
        self.time = 0.0
        self.weight_history = [(0.0, self.weights.copy())]
        
    def get_weight_statistics(self) -> Dict[str, float]:
        return {
            'mean': np.mean(self.weights),
            'std': np.std(self.weights),
            'min': np.min(self.weights),
            'max': np.max(self.weights),
            'total': np.sum(self.weights),
            'time': self.time
        }
    
    def get_volatility_estimate(self, window: int = 20) -> float:
        if len(self.weight_history) < window + 1:
            return self.volatility
        
        recent_weights = np.array([w for _, w in self.weight_history[-window-1:]])
        returns = np.diff(np.log(recent_weights), axis=0)
        
        return np.std(returns) * np.sqrt(len(returns))


class StochasticSchedulingModel:
    """
    随机过程调度模型
    整合HMM、泊松过程和维纳过程
    """
    
    def __init__(self, n_states: int, n_questions: int, n_weights: int):
        self.n_states = n_states
        self.n_questions = n_questions
        self.n_weights = n_weights
        
        self.hmm = ThirdOrderHMM(n_states, n_questions)
        self.poisson = AdaptivePoissonProcess(initial_rate=1.0)
        self.wiener = WienerProcessPerturbation(n_weights)
        
        self.schedule_history: List[Dict] = []
        self.current_state = 0
        self.current_time = 0.0
        
    def initialize(self, initial_weights: np.ndarray = None):
        if initial_weights is None:
            initial_weights = np.ones(self.n_weights)
        self.wiener.initialize_weights(initial_weights)
        self.hmm.initialize_sequence()
        self.current_state = self.hmm.state_history[-1]
        
    def schedule_next_question(self, dt: float = 0.1) -> Dict:
        next_time = self.poisson.generate_event()
        
        self.wiener.evolve(dt, method='ornstein_uhlenbeck')
        
        if len(self.hmm.state_history) >= 3:
            s1, s2, s3 = (self.hmm.state_history[-3], 
                         self.hmm.state_history[-2], 
                         self.hmm.state_history[-1])
            next_state = self.hmm.sample_next_state(s1, s2, s3)
        else:
            next_state = np.random.choice(self.n_states)
        
        self.hmm.state_history.append(next_state)
        
        weights = self.wiener.get_normalized_weights()
        
        observation = self.hmm.sample_observation(next_state)
        
        schedule_info = {
            'time': next_time,
            'state': next_state,
            'observation': observation,
            'weights': weights.copy(),
            'poisson_rate': self.poisson.current_rate,
            'interval': self.poisson.intervals[-1] if self.poisson.intervals else 0.0
        }
        
        self.schedule_history.append(schedule_info)
        self.current_state = next_state
        self.current_time = next_time
        
        return schedule_info
    
    def generate_schedule(self, n_questions: int, dt: float = 0.1) -> List[Dict]:
        schedule = []
        for _ in range(n_questions):
            info = self.schedule_next_question(dt)
            schedule.append(info)
        return schedule
    
    def adapt_by_performance(self, correct: bool, difficulty: float = 0.5):
        feedback = 0.1 if correct else -0.1
        self.poisson.adapt_rate(feedback * difficulty)
        
        if correct:
            self.wiener.weights *= (1 + 0.05 * difficulty)
        else:
            self.wiener.weights *= (1 - 0.03 * difficulty)
        
        self.wiener.weights = np.clip(self.wiener.weights, 0.01, 10.0)
        
    def get_state_sequence(self) -> List[int]:
        return self.hmm.state_history.copy()
    
    def get_observation_sequence(self) -> List[int]:
        return self.hmm.observation_history.copy()
    
    def get_current_weights(self) -> np.ndarray:
        return self.wiener.weights.copy()
    
    def get_statistics(self) -> Dict:
        return {
            'hmm_states': len(self.hmm.state_history),
            'poisson_events': self.poisson._event_count,
            'wiener_time': self.wiener.time,
            'current_poisson_rate': self.poisson.current_rate,
            'weight_stats': self.wiener.get_weight_statistics(),
            'poisson_stats': self.poisson.get_statistics()
        }


def validate_stochastic_processes():
    print("=" * 60)
    print("随机过程调度模型验证")
    print("=" * 60)
    
    print("\n1. 三阶隐马尔可夫模型验证")
    hmm = ThirdOrderHMM(n_states=3, n_observations=5)
    
    transition_sum = np.sum(hmm.transition_matrix, axis=3)
    print(f"   状态转移矩阵行和（应全为1）: {transition_sum[0, 0, :]}")
    
    states, obs = hmm.generate_sequence(10)
    print(f"   生成状态序列长度: {len(states)}")
    print(f"   生成观测序列长度: {len(obs)}")
    
    viterbi_path = hmm.viterbi_algorithm(obs)
    print(f"   Viterbi解码路径长度: {len(viterbi_path)}")
    
    print("\n2. 自适应泊松过程验证")
    poisson = AdaptivePoissonProcess(initial_rate=2.0, min_rate=0.5, max_rate=5.0)
    
    events = poisson.generate_multiple_events(100)
    print(f"   生成事件数: {len(events)}")
    print(f"   平均间隔: {np.mean(poisson.intervals):.4f} (期望: {1/2.0:.4f})")
    
    poisson.adapt_by_performance(0.5, target_rate=0.7)
    print(f"   自适应后到达率: {poisson.current_rate:.4f}")
    
    stats = poisson.get_statistics()
    print(f"   统计信息: mean_interval={stats['mean_interval']:.4f}")
    
    print("\n3. 维纳过程随机扰动验证")
    wiener = WienerProcessPerturbation(n_weights=5, volatility=0.1, mean_reversion=0.5)
    
    initial_weights = wiener.weights.copy()
    print(f"   初始权重: {initial_weights}")
    
    for _ in range(100):
        wiener.evolve(0.01, method='ornstein_uhlenbeck')
    
    final_weights = wiener.weights.copy()
    print(f"   演化后权重: {final_weights}")
    print(f"   权重变化: {np.linalg.norm(final_weights - initial_weights):.4f}")
    
    normalized = wiener.get_normalized_weights()
    print(f"   归一化权重和: {np.sum(normalized):.4f} (期望: 1.0)")
    
    print("\n4. 随机过程调度模型整合验证")
    model = StochasticSchedulingModel(n_states=3, n_questions=10, n_weights=5)
    model.initialize()
    
    schedule = model.generate_schedule(20)
    print(f"   生成调度数量: {len(schedule)}")
    
    stats = model.get_statistics()
    print(f"   HMM状态数: {stats['hmm_states']}")
    print(f"   泊松事件数: {stats['poisson_events']}")
    print(f"   维纳过程时间: {stats['wiener_time']:.4f}")
    
    model.adapt_by_performance(True, 0.5)
    model.adapt_by_performance(False, 0.7)
    print(f"   性能自适应后泊松到达率: {model.poisson.current_rate:.4f}")
    
    print("\n" + "=" * 60)
    print("所有随机过程验证完成!")
    print("=" * 60)


from scipy.stats import qmc


class EntropyBasedStratifier:
    """
    按题目分类熵值分层器
    计算各类别的熵值并分配分层权重
    """
    
    def __init__(self, n_categories: int):
        self.n_categories = n_categories
        self.category_entropies: Dict[int, float] = {}
        self.stratum_weights: np.ndarray = None
        self.category_samples: Dict[int, List[Any]] = {}
        
    def compute_category_entropy(self, samples: List[Dict], 
                                  labels: List[int],
                                  category: int) -> float:
        category_samples = [s for s, l in zip(samples, labels) if l == category]
        if not category_samples:
            return 0.0
        
        feature_values = defaultdict(int)
        for sample in category_samples:
            for key, value in sample.items():
                if isinstance(value, (int, float)):
                    discretized = int(value * 10) if value < 10 else int(value)
                    feature_values[(key, discretized)] += 1
                else:
                    feature_values[(key, hash(str(value)) % 100)] += 1
        
        total = sum(feature_values.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in feature_values.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p + 1e-10)
        
        return entropy
    
    def fit(self, samples: List[Dict], labels: List[int]):
        total_entropy = 0.0
        
        for cat in range(self.n_categories):
            entropy = self.compute_category_entropy(samples, labels, cat)
            self.category_entropies[cat] = entropy
            total_entropy += entropy
        
        if total_entropy > 0:
            self.stratum_weights = np.array([
                self.category_entropies.get(cat, 0.0) / total_entropy
                for cat in range(self.n_categories)
            ])
        else:
            self.stratum_weights = np.ones(self.n_categories) / self.n_categories
        
        for s, l in zip(samples, labels):
            if l not in self.category_samples:
                self.category_samples[l] = []
            self.category_samples[l].append(s)
    
    def get_stratum_allocation(self, total_samples: int) -> Dict[int, int]:
        allocation = {}
        remaining = total_samples
        
        for cat in range(self.n_categories):
            weight = self.stratum_weights[cat] if self.stratum_weights is not None else 1.0 / self.n_categories
            n_samples = max(1, int(total_samples * weight))
            available = len(self.category_samples.get(cat, []))
            allocation[cat] = min(n_samples, available)
            remaining -= allocation[cat]
        
        sorted_cats = sorted(range(self.n_categories), 
                            key=lambda c: self.stratum_weights[c] if self.stratum_weights is not None else 1.0,
                            reverse=True)
        for cat in sorted_cats:
            if remaining <= 0:
                break
            available = len(self.category_samples.get(cat, []))
            if allocation[cat] < available:
                add = min(remaining, available - allocation[cat])
                allocation[cat] += add
                remaining -= add
        
        return allocation


class MultiStageMonteCarloSampler:
    """
    多阶段蒙特卡洛抽样器
    实现阶段递进的精细化抽样
    """
    
    def __init__(self, n_stages: int = 3, 
                 base_samples: int = 100,
                 refinement_factor: float = 0.5):
        self.n_stages = n_stages
        self.base_samples = base_samples
        self.refinement_factor = refinement_factor
        self.stage_samples: List[List[Any]] = []
        self.stage_weights: List[np.ndarray] = []
        
    def _compute_feature_importance(self, samples: List[Dict]) -> Dict[str, float]:
        if not samples:
            return {}
        
        all_features = set()
        for s in samples:
            all_features.update(s.keys())
        
        importance = {}
        for feature in all_features:
            values = [s.get(feature, 0) for s in samples if feature in s]
            if values and len(values) > 1:
                std = np.std(values)
                importance[feature] = std
            else:
                importance[feature] = 0.0
        
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _select_samples_by_importance(self, samples: List[Dict], 
                                       n_select: int,
                                       importance: Dict[str, float]) -> List[int]:
        if len(samples) <= n_select:
            return list(range(len(samples)))
        
        scores = []
        for i, sample in enumerate(samples):
            score = sum(
                importance.get(f, 0.0) * abs(v) if isinstance(v, (int, float)) else importance.get(f, 0.0)
                for f, v in sample.items()
            )
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in scores[:n_select]]
        
        return selected
    
    def sample(self, population: List[Dict], 
               stratum_allocation: Dict[int, int],
               category_labels: List[int]) -> Tuple[List[Dict], np.ndarray]:
        self.stage_samples = []
        self.stage_weights = []
        
        current_population = population.copy()
        current_labels = category_labels.copy() if category_labels else list(range(len(population)))
        
        for stage in range(self.n_stages):
            stage_samples = []
            stage_sample_weights = []
            
            n_stage = max(1, int(self.base_samples * (self.refinement_factor ** stage)))
            
            importance = self._compute_feature_importance(current_population)
            
            for cat, n_alloc in stratum_allocation.items():
                cat_indices = [i for i, l in enumerate(current_labels) if l == cat]
                
                if not cat_indices:
                    continue
                
                n_cat = min(n_alloc, len(cat_indices))
                n_stage_cat = max(1, n_stage // len(stratum_allocation))
                
                selected = self._select_samples_by_importance(
                    [current_population[i] for i in cat_indices],
                    n_stage_cat,
                    importance
                )
                
                for idx in selected:
                    original_idx = cat_indices[idx]
                    stage_samples.append(current_population[original_idx])
                    weight = 1.0 / (n_stage_cat * len(stratum_allocation))
                    stage_sample_weights.append(weight)
            
            self.stage_samples.append(stage_samples)
            self.stage_weights.append(np.array(stage_sample_weights))
            
            if stage < self.n_stages - 1:
                all_selected = []
                for s in stage_samples:
                    try:
                        idx = population.index(s)
                        all_selected.append(idx)
                    except ValueError:
                        pass
                
                if all_selected:
                    current_population = [population[i] for i in all_selected]
                    current_labels = [category_labels[i] for i in all_selected] if category_labels else list(range(len(current_population)))
        
        final_samples = []
        final_weights = []
        seen_ids = set()
        
        for stage_idx, (samples, weights) in enumerate(zip(self.stage_samples, self.stage_weights)):
            stage_weight = (self.refinement_factor ** stage_idx)
            for s, w in zip(samples, weights):
                qid = s.get('id') if isinstance(s, dict) else id(s)
                if qid not in seen_ids:
                    seen_ids.add(qid)
                    final_samples.append(s)
                    final_weights.append(w * stage_weight)
        
        total_weight = sum(final_weights)
        if total_weight > 0:
            final_weights = [w / total_weight for w in final_weights]
        
        return final_samples, np.array(final_weights)


class ImportanceWeightIterator:
    """
    重要性抽样权重迭代器
    基于分类错分概率动态调整权重
    """
    
    def __init__(self, learning_rate: float = 0.1,
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-4):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.weights_history: List[np.ndarray] = []
        self.misclassification_rates: List[Dict[int, float]] = []
        
    def _compute_misclassification_probability(self, samples: List[Dict],
                                                labels: List[int],
                                                predictions: List[int]) -> Dict[int, float]:
        misclass_rates = {}
        
        for cat in set(labels):
            cat_indices = [i for i, l in enumerate(labels) if l == cat]
            if not cat_indices:
                misclass_rates[cat] = 0.0
                continue
            
            misclass_count = sum(1 for i in cat_indices if predictions[i] != labels[i])
            misclass_rates[cat] = misclass_count / len(cat_indices)
        
        return misclass_rates
    
    def _update_weights_by_misclassification(self, weights: np.ndarray,
                                              labels: List[int],
                                              predictions: List[int],
                                              current_misclass: Dict[int, float]) -> np.ndarray:
        new_weights = weights.copy()
        
        for i, (label, pred) in enumerate(zip(labels, predictions)):
            if label != pred:
                misclass_rate = current_misclass.get(label, 0.0)
                adjustment = 1.0 + self.learning_rate * (1.0 + misclass_rate)
                new_weights[i] *= adjustment
        
        total = np.sum(new_weights)
        if total > 0:
            new_weights /= total
        
        return new_weights
    
    def iterate(self, initial_weights: np.ndarray,
                samples: List[Dict],
                labels: List[int],
                predict_func: Callable[[Dict], int]) -> Tuple[np.ndarray, int]:
        current_weights = initial_weights.copy()
        self.weights_history = [current_weights.copy()]
        self.misclassification_rates = []
        
        n_iterations = 0
        
        for iteration in range(self.max_iterations):
            predictions = [predict_func(s) for s in samples]
            
            misclass_rates = self._compute_misclassification_probability(
                samples, labels, predictions
            )
            self.misclassification_rates.append(misclass_rates)
            
            new_weights = self._update_weights_by_misclassification(
                current_weights, labels, predictions, misclass_rates
            )
            
            weight_change = np.linalg.norm(new_weights - current_weights)
            
            self.weights_history.append(new_weights.copy())
            current_weights = new_weights
            n_iterations = iteration + 1
            
            if weight_change < self.convergence_threshold:
                break
        
        return current_weights, n_iterations
    
    def get_effective_sample_size(self, weights: np.ndarray) -> float:
        if len(weights) == 0:
            return 0.0
        
        normalized = weights / np.sum(weights)
        ess = 1.0 / np.sum(normalized ** 2)
        return ess
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        if not self.weights_history:
            return {}
        
        final_weights = self.weights_history[-1]
        
        return {
            'n_iterations': len(self.weights_history) - 1,
            'final_mean': np.mean(final_weights),
            'final_std': np.std(final_weights),
            'final_min': np.min(final_weights),
            'final_max': np.max(final_weights),
            'effective_sample_size': self.get_effective_sample_size(final_weights),
            'weight_evolution': [np.mean(w) for w in self.weights_history]
        }


class QuasiMonteCarloSequence:
    """
    拟蒙特卡洛低差异序列生成器
    支持Sobol和Halton序列
    """
    
    def __init__(self, dimension: int, sequence_type: str = 'sobol'):
        self.dimension = dimension
        self.sequence_type = sequence_type
        self.sobol_engine = None
        self.halton_engine = None
        self._init_engines()
        
    def _init_engines(self):
        if self.sequence_type in ['sobol', 'both']:
            self.sobol_engine = qmc.Sobol(d=self.dimension, scramble=True)
        if self.sequence_type in ['halton', 'both']:
            self.halton_engine = qmc.Halton(d=self.dimension, scramble=True)
    
    def generate_sobol(self, n_samples: int) -> np.ndarray:
        if self.sobol_engine is None:
            self.sobol_engine = qmc.Sobol(d=self.dimension, scramble=True)
        
        n_power = 2 ** int(np.ceil(np.log2(n_samples)))
        samples = self.sobol_engine.random(n_power)
        
        return samples[:n_samples]
    
    def generate_halton(self, n_samples: int) -> np.ndarray:
        if self.halton_engine is None:
            self.halton_engine = qmc.Halton(d=self.dimension, scramble=True)
        
        return self.halton_engine.random(n_samples)
    
    def generate(self, n_samples: int, method: str = None) -> np.ndarray:
        method = method or self.sequence_type
        
        if method == 'sobol':
            return self.generate_sobol(n_samples)
        elif method == 'halton':
            return self.generate_halton(n_samples)
        elif method == 'both':
            sobol = self.generate_sobol(n_samples // 2)
            halton = self.generate_halton(n_samples - n_samples // 2)
            return np.vstack([sobol, halton])
        else:
            return np.random.random((n_samples, self.dimension))
    
    def compute_discrepancy(self, samples: np.ndarray) -> float:
        if len(samples) == 0:
            return float('inf')
        
        return qmc.discrepancy(samples)
    
    def map_to_feature_space(self, sequence: np.ndarray,
                              feature_bounds: Dict[str, Tuple[float, float]]) -> List[Dict]:
        features = list(feature_bounds.keys())
        samples = []
        
        for point in sequence:
            sample = {}
            for i, feature in enumerate(features):
                if i < len(point):
                    lower, upper = feature_bounds[feature]
                    value = lower + point[i] * (upper - lower)
                    sample[feature] = value
            samples.append(sample)
        
        return samples
    
    def reset(self):
        if self.sobol_engine is not None:
            self.sobol_engine = qmc.Sobol(d=self.dimension, scramble=True)
        if self.halton_engine is not None:
            self.halton_engine = qmc.Halton(d=self.dimension, scramble=True)


class HighDimensionalFeatureSampler:
    """
    高维特征空间抽样器
    结合拟蒙特卡洛方法处理高维特征
    """
    
    def __init__(self, n_features: int, 
                 sequence_type: str = 'sobol',
                 use_stratification: bool = True):
        self.n_features = n_features
        self.sequence_type = sequence_type
        self.use_stratification = use_stratification
        self.qmc_sequence = QuasiMonteCarloSequence(n_features, sequence_type)
        self.feature_importance: np.ndarray = None
        
    def _estimate_feature_importance(self, samples: List[Dict]) -> np.ndarray:
        if not samples:
            return np.ones(self.n_features) / self.n_features
        
        feature_values = {f: [] for f in range(self.n_features)}
        
        for sample in samples:
            for i in range(self.n_features):
                key = f'feature_{i}'
                value = sample.get(key, sample.get(i, 0))
                if isinstance(value, (int, float)):
                    feature_values[i].append(value)
        
        importance = np.zeros(self.n_features)
        for i in range(self.n_features):
            if feature_values[i]:
                importance[i] = np.std(feature_values[i])
            else:
                importance[i] = 1.0
        
        total = np.sum(importance)
        if total > 0:
            importance /= total
        
        self.feature_importance = importance
        return importance
    
    def _stratified_qmc_sample(self, n_samples: int,
                                n_strata: int = 4) -> np.ndarray:
        samples_per_stratum = n_samples // n_strata
        all_samples = []
        
        for stratum in range(n_strata):
            stratum_samples = self.qmc_sequence.generate(samples_per_stratum)
            
            lower = stratum / n_strata
            upper = (stratum + 1) / n_strata
            
            stratified = lower + stratum_samples * (upper - lower)
            all_samples.append(stratified)
        
        remaining = n_samples - len(all_samples) * samples_per_stratum
        if remaining > 0:
            extra = self.qmc_sequence.generate(remaining)
            all_samples.append(extra)
        
        return np.vstack(all_samples)
    
    def sample(self, n_samples: int,
               feature_bounds: Dict[str, Tuple[float, float]],
               reference_samples: List[Dict] = None) -> Tuple[List[Dict], np.ndarray]:
        if reference_samples:
            self._estimate_feature_importance(reference_samples)
        
        if self.use_stratification:
            raw_samples = self._stratified_qmc_sample(n_samples)
        else:
            raw_samples = self.qmc_sequence.generate(n_samples)
        
        features = list(feature_bounds.keys())[:self.n_features]
        
        samples = []
        weights = []
        
        for point in raw_samples:
            sample = {}
            weight = 1.0
            
            for i, feature in enumerate(features):
                if i < len(point):
                    lower, upper = feature_bounds.get(feature, (0.0, 1.0))
                    value = lower + point[i] * (upper - lower)
                    sample[feature] = value
                    
                    if self.feature_importance is not None:
                        weight *= (1.0 + self.feature_importance[i])
            
            samples.append(sample)
            weights.append(weight)
        
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        return samples, weights
    
    def compute_sampling_quality(self, samples: np.ndarray) -> Dict[str, float]:
        if len(samples) == 0:
            return {'discrepancy': float('inf'), 'coverage': 0.0}
        
        discrepancy = self.qmc_sequence.compute_discrepancy(samples)
        
        coverage = np.mean([
            np.min(samples[:, i]) < 0.1 and np.max(samples[:, i]) > 0.9
            for i in range(min(samples.shape[1], self.n_features))
        ])
        
        min_dist = float('inf')
        n = len(samples)
        for i in range(min(n, 100)):
            for j in range(i + 1, min(n, 100)):
                dist = np.linalg.norm(samples[i] - samples[j])
                if dist < min_dist:
                    min_dist = dist
        
        return {
            'discrepancy': discrepancy,
            'coverage': coverage,
            'min_distance': min_dist,
            'uniformity': 1.0 / (1.0 + discrepancy * 100)
        }


class MonteCarloStratifiedSamplingFramework:
    """
    蒙特卡洛分层抽样框架
    整合熵值分层、多阶段抽样、重要性权重迭代和拟蒙特卡洛方法
    """
    
    def __init__(self, n_features: int,
                 n_categories: int,
                 n_stages: int = 3,
                 base_samples: int = 100,
                 sequence_type: str = 'sobol'):
        self.n_features = n_features
        self.n_categories = n_categories
        
        self.stratifier = EntropyBasedStratifier(n_categories)
        self.multi_stage_sampler = MultiStageMonteCarloSampler(n_stages, base_samples)
        self.weight_iterator = ImportanceWeightIterator()
        self.hd_sampler = HighDimensionalFeatureSampler(n_features, sequence_type)
        
        self.is_fitted = False
        self.sampling_history: List[Dict] = []
        
    def fit(self, samples: List[Dict], labels: List[int]):
        self.stratifier.fit(samples, labels)
        self.is_fitted = True
        
    def sample(self, n_samples: int,
               samples: List[Dict],
               labels: List[int],
               predict_func: Callable[[Dict], int] = None,
               feature_bounds: Dict[str, Tuple[float, float]] = None,
               use_qmc: bool = True,
               use_importance_iteration: bool = True) -> Dict[str, Any]:
        if not self.is_fitted:
            self.fit(samples, labels)
        
        stratum_allocation = self.stratifier.get_stratum_allocation(n_samples)
        
        mc_samples, mc_weights = self.multi_stage_sampler.sample(
            samples, stratum_allocation, labels
        )
        
        qmc_samples = None
        qmc_weights = None
        if use_qmc and feature_bounds:
            qmc_samples, qmc_weights = self.hd_sampler.sample(
                n_samples, feature_bounds, samples
            )
        
        final_samples = mc_samples
        final_weights = mc_weights
        
        if qmc_samples is not None:
            seen_ids = {s.get('id') if isinstance(s, dict) else id(s) for s in mc_samples}
            combined_samples = mc_samples.copy()
            combined_weights = list(mc_weights * 0.7)
            
            for i, s in enumerate(qmc_samples):
                qid = s.get('id') if isinstance(s, dict) else id(s)
                if qid not in seen_ids:
                    seen_ids.add(qid)
                    combined_samples.append(s)
                    combined_weights.append(qmc_weights[i] * 0.3)
            
            combined_weights = np.array(combined_weights)
            combined_weights /= np.sum(combined_weights)
            final_samples = combined_samples
            final_weights = combined_weights
        
        if use_importance_iteration and predict_func is not None:
            sample_labels = []
            for s in final_samples:
                try:
                    idx = samples.index(s)
                    sample_labels.append(labels[idx])
                except ValueError:
                    sample_labels.append(0)
            
            final_weights, n_iterations = self.weight_iterator.iterate(
                final_weights, final_samples, sample_labels, predict_func
            )
        
        quality_metrics = self._compute_quality_metrics(
            final_samples, final_weights, stratum_allocation
        )
        
        result = {
            'samples': final_samples,
            'weights': final_weights,
            'stratum_allocation': stratum_allocation,
            'quality_metrics': quality_metrics,
            'n_samples': len(final_samples)
        }
        
        self.sampling_history.append(result)
        
        return result
    
    def _compute_quality_metrics(self, samples: List[Dict],
                                  weights: np.ndarray,
                                  stratum_allocation: Dict[int, int]) -> Dict[str, Any]:
        ess = self.weight_iterator.get_effective_sample_size(weights)
        
        weight_stats = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights),
            'cv': np.std(weights) / np.mean(weights) if np.mean(weights) > 0 else float('inf')
        }
        
        allocation_balance = 1.0 - np.std(list(stratum_allocation.values())) / np.mean(list(stratum_allocation.values())) if stratum_allocation else 0.0
        
        return {
            'effective_sample_size': ess,
            'weight_statistics': weight_stats,
            'allocation_balance': allocation_balance,
            'entropy_based_stratification': self.stratifier.category_entropies.copy()
        }
    
    def get_sampling_summary(self) -> Dict[str, Any]:
        if not self.sampling_history:
            return {'total_sampling_rounds': 0}
        
        avg_ess = np.mean([r['quality_metrics']['effective_sample_size'] 
                          for r in self.sampling_history])
        avg_balance = np.mean([r['quality_metrics']['allocation_balance'] 
                              for r in self.sampling_history])
        
        return {
            'total_sampling_rounds': len(self.sampling_history),
            'average_effective_sample_size': avg_ess,
            'average_allocation_balance': avg_balance,
            'total_samples_generated': sum(r['n_samples'] for r in self.sampling_history)
        }
    
    def reset(self):
        self.stratifier = EntropyBasedStratifier(self.n_categories)
        self.multi_stage_sampler = MultiStageMonteCarloSampler()
        self.weight_iterator = ImportanceWeightIterator()
        self.hd_sampler.qmc_sequence.reset()
        self.is_fitted = False
        self.sampling_history = []


def validate_monte_carlo_sampling():
    print("=" * 60)
    print("蒙特卡洛分层抽样框架验证")
    print("=" * 60)
    
    np.random.seed(42)
    
    print("\n1. 熵值分层器验证")
    stratifier = EntropyBasedStratifier(n_categories=3)
    samples = [
        {'f1': 1.0, 'f2': 2.0}, {'f1': 1.5, 'f2': 2.5}, {'f1': 1.2, 'f2': 2.2},
        {'f1': 5.0, 'f2': 6.0}, {'f1': 5.5, 'f2': 6.5}, {'f1': 5.2, 'f2': 6.2},
        {'f1': 10.0, 'f2': 11.0}, {'f1': 10.5, 'f2': 11.5}, {'f1': 10.2, 'f2': 11.2}
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    stratifier.fit(samples, labels)
    print(f"   类别熵值: {stratifier.category_entropies}")
    print(f"   分层权重: {stratifier.stratum_weights}")
    
    allocation = stratifier.get_stratum_allocation(100)
    print(f"   样本分配: {allocation}")
    
    print("\n2. 多阶段蒙特卡洛抽样验证")
    sampler = MultiStageMonteCarloSampler(n_stages=3, base_samples=10)
    mc_samples, mc_weights = sampler.sample(samples, allocation, labels)
    print(f"   抽样数量: {len(mc_samples)}")
    print(f"   权重均值: {np.mean(mc_weights):.4f}")
    print(f"   权重标准差: {np.std(mc_weights):.4f}")
    
    print("\n3. 重要性权重迭代验证")
    def dummy_predictor(s):
        if s.get('f1', 0) < 4:
            return 0
        elif s.get('f1', 0) < 8:
            return 1
        else:
            return 2
    
    iterator = ImportanceWeightIterator(learning_rate=0.1, max_iterations=5)
    initial_weights = np.ones(len(mc_samples)) / len(mc_samples)
    sample_labels = [labels[samples.index(s)] if s in samples else 0 for s in mc_samples]
    
    final_weights, n_iter = iterator.iterate(
        initial_weights, mc_samples, sample_labels, dummy_predictor
    )
    stats = iterator.get_weight_statistics()
    print(f"   迭代次数: {n_iter}")
    print(f"   有效样本量: {stats['effective_sample_size']:.2f}")
    print(f"   最终权重均值: {stats['final_mean']:.4f}")
    
    print("\n4. 拟蒙特卡洛序列验证")
    qmc_seq = QuasiMonteCarloSequence(dimension=5, sequence_type='sobol')
    sobol_samples = qmc_seq.generate_sobol(64)
    discrepancy = qmc_seq.compute_discrepancy(sobol_samples)
    print(f"   Sobol序列样本数: {len(sobol_samples)}")
    print(f"   差异性指标: {discrepancy:.6f}")
    
    halton_samples = qmc_seq.generate_halton(64)
    halton_discrepancy = qmc_seq.compute_discrepancy(halton_samples)
    print(f"   Halton序列差异性: {halton_discrepancy:.6f}")
    
    print("\n5. 高维特征空间抽样验证")
    hd_sampler = HighDimensionalFeatureSampler(n_features=3, sequence_type='sobol')
    feature_bounds = {
        'f1': (0.0, 10.0),
        'f2': (0.0, 20.0),
        'f3': (0.0, 5.0)
    }
    hd_samples, hd_weights = hd_sampler.sample(50, feature_bounds, samples)
    quality = hd_sampler.compute_sampling_quality(
        np.array([[s.get('f1', 0)/10, s.get('f2', 0)/20, s.get('f3', 0)/5] 
                  for s in hd_samples])
    )
    print(f"   高维抽样数量: {len(hd_samples)}")
    print(f"   覆盖率: {quality['coverage']:.4f}")
    print(f"   均匀性: {quality['uniformity']:.4f}")
    
    print("\n6. 完整框架验证")
    framework = MonteCarloStratifiedSamplingFramework(
        n_features=2,
        n_categories=3,
        n_stages=2,
        base_samples=20,
        sequence_type='sobol'
    )
    
    result = framework.sample(
        n_samples=50,
        samples=samples,
        labels=labels,
        predict_func=dummy_predictor,
        feature_bounds=feature_bounds,
        use_qmc=True,
        use_importance_iteration=True
    )
    
    print(f"   最终样本数: {result['n_samples']}")
    print(f"   有效样本量: {result['quality_metrics']['effective_sample_size']:.2f}")
    print(f"   分配均衡度: {result['quality_metrics']['allocation_balance']:.4f}")
    
    summary = framework.get_sampling_summary()
    print(f"   总抽样轮次: {summary['total_sampling_rounds']}")
    
    print("\n" + "=" * 60)
    print("蒙特卡洛分层抽样框架验证完成!")
    print("=" * 60)


class PermutationGroup:
    """
    n阶置换群类
    实现置换群的基本操作和子群分解
    """
    
    def __init__(self, n: int):
        if n < 2:
            raise ValueError("n must be at least 2 for permutation group")
        self.n = n
        self.identity = tuple(range(n))
        self._elements_cache = None
        self._subgroups_cache = {}
        
    def generate_all_permutations(self) -> List[Tuple[int, ...]]:
        """生成S_n的所有置换"""
        if self._elements_cache is not None:
            return self._elements_cache
        
        self._elements_cache = list(itertools.permutations(range(self.n)))
        return self._elements_cache
    
    def compose(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
        """置换复合 p1 ∘ p2 (先p2后p1)"""
        return tuple(p1[p2[i]] for i in range(self.n))
    
    def inverse(self, p: Tuple[int, ...]) -> Tuple[int, ...]:
        """求置换的逆"""
        inv = [0] * self.n
        for i, v in enumerate(p):
            inv[v] = i
        return tuple(inv)
    
    def get_cycle_decomposition(self, p: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """获取置换的循环分解"""
        visited = [False] * self.n
        cycles = []
        
        for start in range(self.n):
            if visited[start]:
                continue
            
            cycle = []
            current = start
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = p[current]
            
            if len(cycle) > 1:
                cycles.append(tuple(cycle))
        
        return cycles
    
    def get_cycle_type(self, p: Tuple[int, ...]) -> Tuple[int, ...]:
        """获取置换的循环类型（各循环长度的降序排列）"""
        cycles = self.get_cycle_decomposition(p)
        
        fixed_points = self.n - sum(len(c) for c in cycles)
        cycle_lengths = [1] * fixed_points + [len(c) for c in cycles]
        
        return tuple(sorted(cycle_lengths, reverse=True))
    
    def is_even(self, p: Tuple[int, ...]) -> bool:
        """判断置换是否为偶置换"""
        cycles = self.get_cycle_decomposition(p)
        transposition_count = sum(len(c) - 1 for c in cycles)
        return transposition_count % 2 == 0
    
    def get_sign(self, p: Tuple[int, ...]) -> int:
        """获取置换的符号（偶置换为1，奇置换为-1）"""
        return 1 if self.is_even(p) else -1
    
    def generate_alternating_subgroup(self) -> List[Tuple[int, ...]]:
        """生成交错群A_n（所有偶置换）"""
        all_perms = self.generate_all_permutations()
        return [p for p in all_perms if self.is_even(p)]
    
    def generate_cyclic_subgroup(self, generator: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """生成由generator生成的循环子群"""
        subgroup = [self.identity]
        current = generator
        
        while current != self.identity:
            subgroup.append(current)
            current = self.compose(generator, current)
        
        return subgroup
    
    def is_subgroup(self, subset: List[Tuple[int, ...]]) -> bool:
        """判断子集是否为子群"""
        if self.identity not in subset:
            return False
        
        subset_set = set(subset)
        for p1 in subset:
            if self.inverse(p1) not in subset_set:
                return False
            for p2 in subset:
                if self.compose(p1, p2) not in subset_set:
                    return False
        
        return True
    
    def decompose_subgroups(self) -> Dict[str, List[Tuple[int, ...]]]:
        """分解S_n的主要子群"""
        if self.n in self._subgroups_cache:
            return self._subgroups_cache[self.n]
        
        subgroups = {}
        
        subgroups['S_n'] = self.generate_all_permutations()
        
        subgroups['A_n'] = self.generate_alternating_subgroup()
        
        subgroups['trivial'] = [self.identity]
        
        all_perms = self.generate_all_permutations()
        for i, p in enumerate(all_perms):
            if p != self.identity:
                cyclic = self.generate_cyclic_subgroup(p)
                if len(cyclic) > 1:
                    subgroups[f'cyclic_{i}'] = cyclic
        
        self._subgroups_cache[self.n] = subgroups
        return subgroups
    
    def get_subgroup_lattice(self) -> Dict[str, List[str]]:
        """获取子群格（包含关系）"""
        subgroups = self.decompose_subgroups()
        lattice = {}
        
        for name1, sg1 in subgroups.items():
            contained_in = []
            sg1_set = set(sg1)
            for name2, sg2 in subgroups.items():
                if name1 != name2:
                    if sg1_set <= set(sg2):
                        contained_in.append(name2)
            lattice[name1] = contained_in
        
        return lattice
    
    def orbit(self, element: Any, group_elements: List[Tuple[int, ...]], 
              action: Callable) -> set:
        """计算群作用下元素的轨道"""
        orbit_set = set()
        for g in group_elements:
            orbit_set.add(action(g, element))
        return orbit_set
    
    def stabilizer(self, element: Any, group_elements: List[Tuple[int, ...]],
                   action: Callable) -> List[Tuple[int, ...]]:
        """计算元素的稳定子群"""
        stab = []
        for g in group_elements:
            if action(g, element) == element:
                stab.append(g)
        return stab


class PermutationParityWeighting:
    """
    置换奇偶性加权模型
    根据题目分类隶属度调整权重
    """
    
    def __init__(self, n: int, category_memberships: Optional[Dict[str, np.ndarray]] = None):
        self.perm_group = PermutationGroup(n)
        self.n = n
        self.category_memberships = category_memberships or {}
        self.parity_weights = {'even': 1.0, 'odd': 1.0}
        self._weight_cache = {}
    
    def set_category_memberships(self, memberships: Dict[str, np.ndarray]):
        """设置各类别的隶属度向量"""
        self.category_memberships = memberships
        self._weight_cache.clear()
    
    def calculate_parity_weight(self, permutation: Tuple[int, ...], 
                                category: Optional[str] = None) -> float:
        """计算单个置换的加权权重"""
        is_even = self.perm_group.is_even(permutation)
        base_weight = self.parity_weights['even'] if is_even else self.parity_weights['odd']
        
        if category is None or category not in self.category_memberships:
            return base_weight
        
        cache_key = (permutation, category)
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]
        
        membership = self.category_memberships[category]
        
        perm_array = np.array(permutation)
        n = len(perm_array)
        
        if len(membership) != n:
            weight = base_weight
        else:
            permuted_membership = membership[list(perm_array)]
            
            correlation = np.corrcoef(membership, permuted_membership)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            membership_factor = 1.0 + 0.5 * abs(correlation)
            
            entropy = -np.sum(membership * np.log(membership + 1e-10))
            entropy_factor = 1.0 / (1.0 + entropy)
            
            weight = base_weight * membership_factor * entropy_factor
        
        self._weight_cache[cache_key] = weight
        return weight
    
    def set_parity_base_weights(self, even_weight: float, odd_weight: float):
        """设置奇偶性的基础权重"""
        self.parity_weights['even'] = even_weight
        self.parity_weights['odd'] = odd_weight
        self._weight_cache.clear()
    
    def get_weighted_permutations(self, category: Optional[str] = None) -> Dict[Tuple[int, ...], float]:
        """获取所有置换及其权重"""
        all_perms = self.perm_group.generate_all_permutations()
        
        weighted = {}
        for p in all_perms:
            weighted[p] = self.calculate_parity_weight(p, category)
        
        return weighted
    
    def weighted_random_select(self, n_select: int, 
                               category: Optional[str] = None,
                               replace: bool = False) -> List[Tuple[int, ...]]:
        """加权随机选择置换"""
        weighted_perms = self.get_weighted_permutations(category)
        perms = list(weighted_perms.keys())
        weights = np.array(list(weighted_perms.values()))
        
        weights = weights / weights.sum()
        
        indices = np.random.choice(len(perms), size=n_select, replace=replace, p=weights)
        
        return [perms[i] for i in indices]
    
    def calculate_category_adjusted_weights(self, 
                                            category_weights: Dict[str, float]) -> Dict[Tuple[int, ...], float]:
        """根据类别权重计算综合调整后的置换权重"""
        all_perms = self.perm_group.generate_all_permutations()
        
        adjusted_weights = {p: 0.0 for p in all_perms}
        
        for category, cat_weight in category_weights.items():
            for p in all_perms:
                perm_weight = self.calculate_parity_weight(p, category)
                adjusted_weights[p] += cat_weight * perm_weight
        
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {p: w / total for p, w in adjusted_weights.items()}
        
        return adjusted_weights
    
    def get_parity_statistics(self) -> Dict[str, Any]:
        """获取奇偶性统计信息"""
        all_perms = self.perm_group.generate_all_permutations()
        
        even_count = sum(1 for p in all_perms if self.perm_group.is_even(p))
        odd_count = len(all_perms) - even_count
        
        return {
            'total_permutations': len(all_perms),
            'even_count': even_count,
            'odd_count': odd_count,
            'even_ratio': even_count / len(all_perms),
            'odd_ratio': odd_count / len(all_perms),
            'base_weights': self.parity_weights.copy()
        }


class BurnsideLemmaCalculator:
    """
    伯恩赛德引理计算器
    计算群作用下的轨道数（有效置换数）
    """
    
    def __init__(self, n: int):
        self.perm_group = PermutationGroup(n)
        self.n = n
        self._fixed_points_cache = {}
    
    def count_fixed_points(self, permutation: Tuple[int, ...], 
                          elements: Optional[set] = None) -> int:
        """计算置换的不动点数"""
        if elements is None:
            count = sum(1 for i in range(self.n) if permutation[i] == i)
            return count
        
        count = 0
        for elem in elements:
            if self._is_fixed(permutation, elem):
                count += 1
        return count
    
    def _is_fixed(self, permutation: Tuple[int, ...], element: Any) -> bool:
        """判断元素是否被置换固定"""
        if isinstance(element, (tuple, list)):
            permuted = tuple(permutation[i] if i < len(permutation) else i for i in element)
            return permuted == tuple(element)
        elif isinstance(element, int):
            return permutation[element] == element if element < len(permutation) else True
        return False
    
    def calculate_orbits(self, group_elements: Optional[List[Tuple[int, ...]]] = None,
                        elements: Optional[set] = None) -> int:
        """
        使用伯恩赛德引理计算轨道数
        |X/G| = (1/|G|) * Σ|X^g|
        """
        if group_elements is None:
            group_elements = self.perm_group.generate_all_permutations()
        
        total_fixed_points = 0
        for g in group_elements:
            fixed = self.count_fixed_points(g, elements)
            total_fixed_points += fixed
        
        orbit_count = total_fixed_points / len(group_elements)
        
        return int(round(orbit_count))
    
    def calculate_orbits_detailed(self, group_elements: Optional[List[Tuple[int, ...]]] = None,
                                  elements: Optional[set] = None) -> Dict[str, Any]:
        """详细计算轨道信息"""
        if group_elements is None:
            group_elements = self.perm_group.generate_all_permutations()
        
        fixed_points_by_perm = {}
        for g in group_elements:
            fixed = self.count_fixed_points(g, elements)
            fixed_points_by_perm[g] = fixed
        
        total_fixed = sum(fixed_points_by_perm.values())
        group_order = len(group_elements)
        orbit_count = total_fixed / group_order
        
        return {
            'orbit_count': orbit_count,
            'group_order': group_order,
            'total_fixed_points': total_fixed,
            'fixed_points_by_permutation': fixed_points_by_perm,
            'average_fixed_points': total_fixed / group_order
        }
    
    def calculate_effective_permutations(self, 
                                         threshold: float = 0.0,
                                         group_elements: Optional[List[Tuple[int, ...]]] = None) -> List[Tuple[int, ...]]:
        """动态筛选有效置换"""
        if group_elements is None:
            group_elements = self.perm_group.generate_all_permutations()
        
        effective = []
        for g in group_elements:
            fixed = self.count_fixed_points(g)
            effectiveness = 1.0 - (fixed / self.n)
            
            if effectiveness >= threshold:
                effective.append(g)
        
        return effective
    
    def calculate_colorings(self, n_colors: int, 
                           group_elements: Optional[List[Tuple[int, ...]]] = None) -> int:
        """
        计算在群作用下的不等价着色数
        使用伯恩赛德引理
        """
        if group_elements is None:
            group_elements = self.perm_group.generate_all_permutations()
        
        total = 0
        for g in group_elements:
            cycles = self.perm_group.get_cycle_decomposition(g)
            n_cycles = len(cycles) + (self.n - sum(len(c) for c in cycles))
            total += n_colors ** n_cycles
        
        return total // len(group_elements)
    
    def filter_by_effectiveness(self, 
                                min_effectiveness: float = 0.5,
                                group_elements: Optional[List[Tuple[int, ...]]] = None) -> Dict[Tuple[int, ...], float]:
        """按有效性筛选置换"""
        if group_elements is None:
            group_elements = self.perm_group.generate_all_permutations()
        
        effectiveness_scores = {}
        for g in group_elements:
            fixed = self.count_fixed_points(g)
            effectiveness = 1.0 - (fixed / self.n)
            
            if effectiveness >= min_effectiveness:
                effectiveness_scores[g] = effectiveness
        
        return effectiveness_scores
    
    def get_cycle_structure_contribution(self, 
                                         permutation: Tuple[int, ...]) -> Dict[str, Any]:
        """获取置换的循环结构贡献"""
        cycles = self.perm_group.get_cycle_decomposition(permutation)
        cycle_type = self.perm_group.get_cycle_type(permutation)
        
        fixed_points = self.n - sum(len(c) for c in cycles)
        
        return {
            'cycles': cycles,
            'cycle_type': cycle_type,
            'num_cycles': len(cycles) + fixed_points,
            'fixed_points': fixed_points,
            'cycle_structure': [len(c) for c in cycles] if cycles else []
        }


class ConjugacyClassPartitioner:
    """
    置换群共轭类划分器
    实现分层随机排序
    """
    
    def __init__(self, n: int):
        self.perm_group = PermutationGroup(n)
        self.n = n
        self._conjugacy_classes_cache = None
        self._class_representatives_cache = None
    
    def are_conjugate(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> bool:
        """判断两个置换是否共轭（具有相同的循环类型）"""
        return self.perm_group.get_cycle_type(p1) == self.perm_group.get_cycle_type(p2)
    
    def get_conjugacy_classes(self) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
        """获取所有共轭类"""
        if self._conjugacy_classes_cache is not None:
            return self._conjugacy_classes_cache
        
        all_perms = self.perm_group.generate_all_permutations()
        
        classes = {}
        for p in all_perms:
            cycle_type = self.perm_group.get_cycle_type(p)
            if cycle_type not in classes:
                classes[cycle_type] = []
            classes[cycle_type].append(p)
        
        self._conjugacy_classes_cache = classes
        return classes
    
    def get_class_representatives(self) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        """获取每个共轭类的代表元"""
        if self._class_representatives_cache is not None:
            return self._class_representatives_cache
        
        classes = self.get_conjugacy_classes()
        representatives = {}
        
        for cycle_type, perms in classes.items():
            representatives[cycle_type] = perms[0]
        
        self._class_representatives_cache = representatives
        return representatives
    
    def get_class_sizes(self) -> Dict[Tuple[int, ...], int]:
        """获取各共轭类的大小"""
        classes = self.get_conjugacy_classes()
        return {ct: len(perms) for ct, perms in classes.items()}
    
    def calculate_class_size_theoretical(self, cycle_type: Tuple[int, ...]) -> int:
        """理论计算共轭类大小"""
        n = sum(cycle_type)
        
        cycle_counts = {}
        for c in cycle_type:
            cycle_counts[c] = cycle_counts.get(c, 0) + 1
        
        numerator = math.factorial(n)
        denominator = 1
        for length, count in cycle_counts.items():
            denominator *= (length ** count) * math.factorial(count)
        
        return numerator // denominator
    
    def stratified_random_sample(self, 
                                 n_samples: int,
                                 weights: Optional[Dict[Tuple[int, ...], float]] = None) -> List[Tuple[int, ...]]:
        """分层随机抽样"""
        classes = self.get_conjugacy_classes()
        
        if weights is None:
            weights = {ct: len(perms) for ct, perms in classes.items()}
        
        total_weight = sum(weights.values())
        normalized_weights = {ct: w / total_weight for ct, w in weights.items()}
        
        samples = []
        remaining = n_samples
        
        sorted_classes = sorted(classes.items(), 
                               key=lambda x: normalized_weights.get(x[0], 0), 
                               reverse=True)
        
        for i, (cycle_type, perms) in enumerate(sorted_classes):
            if i == len(sorted_classes) - 1:
                n_from_class = remaining
            else:
                n_from_class = int(n_samples * normalized_weights[cycle_type])
                n_from_class = min(n_from_class, remaining)
            
            if n_from_class > 0:
                selected = list(np.random.choice(len(perms), 
                                                size=min(n_from_class, len(perms)), 
                                                replace=False))
                samples.extend([perms[j] for j in selected])
                remaining -= n_from_class
            
            if remaining <= 0:
                break
        
        return samples
    
    def stratified_shuffle(self, items: List[Any]) -> List[Any]:
        """使用共轭类分层进行随机排序"""
        n = len(items)
        if n < 2:
            return items.copy()
        
        perm_group_n = PermutationGroup(n)
        partitioner_n = ConjugacyClassPartitioner.__new__(ConjugacyClassPartitioner)
        partitioner_n.perm_group = perm_group_n
        partitioner_n.n = n
        partitioner_n._conjugacy_classes_cache = None
        partitioner_n._class_representatives_cache = None
        
        classes = partitioner_n.get_conjugacy_classes()
        
        class_weights = {}
        for cycle_type, perms in classes.items():
            class_weights[cycle_type] = len(perms)
        
        selected_perms = partitioner_n.stratified_random_sample(1, class_weights)
        
        if selected_perms:
            perm = selected_perms[0]
            return [items[i] for i in perm]
        
        return items.copy()
    
    def get_layer_info(self) -> Dict[str, Any]:
        """获取分层信息"""
        classes = self.get_conjugacy_classes()
        sizes = self.get_class_sizes()
        
        layer_info = {
            'total_classes': len(classes),
            'class_sizes': sizes,
            'largest_class': max(sizes.values()) if sizes else 0,
            'smallest_class': min(sizes.values()) if sizes else 0,
            'average_class_size': np.mean(list(sizes.values())) if sizes else 0,
            'class_distribution': {}
        }
        
        size_distribution = {}
        for size in sizes.values():
            size_distribution[size] = size_distribution.get(size, 0) + 1
        layer_info['class_distribution'] = size_distribution
        
        return layer_info
    
    def get_cycle_type_distribution(self) -> Dict[Tuple[int, ...], float]:
        """获取循环类型分布"""
        sizes = self.get_class_sizes()
        total = sum(sizes.values())
        
        return {ct: size / total for ct, size in sizes.items()}


class PermutationGroupRandomSorter:
    """
    置换群高阶随机排序模型
    整合子群分解、奇偶性加权、伯恩赛德引理和共轭类划分
    """
    
    def __init__(self, n: int, category_memberships: Optional[Dict[str, np.ndarray]] = None):
        if n < 5:
            raise ValueError("n must be at least 5 for advanced permutation group operations")
        
        self.n = n
        self.perm_group = PermutationGroup(n)
        self.parity_weighting = PermutationParityWeighting(n, category_memberships)
        self.burnside_calculator = BurnsideLemmaCalculator(n)
        self.conjugacy_partitioner = ConjugacyClassPartitioner(n)
        
        self.sorting_history: List[Dict] = []
    
    def decompose_and_analyze(self) -> Dict[str, Any]:
        """分解子群并分析结构"""
        subgroups = self.perm_group.decompose_subgroups()
        
        analysis = {
            'n': self.n,
            'total_permutations': len(subgroups.get('S_n', [])),
            'alternating_group_size': len(subgroups.get('A_n', [])),
            'subgroup_count': len(subgroups),
            'subgroup_sizes': {name: len(sg) for name, sg in subgroups.items()},
            'subgroup_lattice': self.perm_group.get_subgroup_lattice()
        }
        
        return analysis
    
    def weighted_parity_sort(self, items: List[Any], 
                            category: Optional[str] = None,
                            n_candidates: int = 10) -> List[Any]:
        """基于奇偶性加权的随机排序"""
        if len(items) != self.n:
            raise ValueError(f"Items length must be {self.n}")
        
        candidates = self.parity_weighting.weighted_random_select(
            n_candidates, category, replace=False
        )
        
        selected = candidates[np.random.randint(len(candidates))]
        
        return [items[i] for i in selected]
    
    def burnside_filtered_sort(self, items: List[Any],
                               effectiveness_threshold: float = 0.3) -> List[Any]:
        """基于伯恩赛德引理筛选后的随机排序"""
        if len(items) != self.n:
            raise ValueError(f"Items length must be {self.n}")
        
        effective_perms = self.burnside_calculator.calculate_effective_permutations(
            effectiveness_threshold
        )
        
        if not effective_perms:
            return items.copy()
        
        selected = effective_perms[np.random.randint(len(effective_perms))]
        
        return [items[i] for i in selected]
    
    def conjugacy_stratified_sort(self, items: List[Any]) -> List[Any]:
        """基于共轭类分层的随机排序"""
        if len(items) != self.n:
            raise ValueError(f"Items length must be {self.n}")
        
        return self.conjugacy_partitioner.stratified_shuffle(items)
    
    def advanced_sort(self, items: List[Any],
                     category: Optional[str] = None,
                     mode: str = 'hybrid',
                     effectiveness_threshold: float = 0.3) -> List[Any]:
        """
        高级随机排序
        mode: 'parity' | 'burnside' | 'conjugacy' | 'hybrid'
        """
        if len(items) != self.n:
            raise ValueError(f"Items length must be {self.n}")
        
        result_info = {
            'mode': mode,
            'category': category,
            'input_items': items.copy()
        }
        
        if mode == 'parity':
            sorted_items = self.weighted_parity_sort(items, category)
        elif mode == 'burnside':
            sorted_items = self.burnside_filtered_sort(items, effectiveness_threshold)
        elif mode == 'conjugacy':
            sorted_items = self.conjugacy_stratified_sort(items)
        else:
            parity_sorted = self.weighted_parity_sort(items, category, n_candidates=5)
            conjugacy_sorted = self.conjugacy_stratified_sort(items)
            
            parity_perm = tuple(items.index(x) for x in parity_sorted)
            conjugacy_perm = tuple(items.index(x) for x in conjugacy_sorted)
            
            parity_weight = self.parity_weighting.calculate_parity_weight(parity_perm, category)
            
            conjugacy_classes = self.conjugacy_partitioner.get_class_sizes()
            conjugacy_type = self.perm_group.get_cycle_type(conjugacy_perm)
            conjugacy_weight = 1.0 / conjugacy_classes.get(conjugacy_type, 1)
            
            if np.random.random() < parity_weight / (parity_weight + conjugacy_weight):
                sorted_items = parity_sorted
            else:
                sorted_items = conjugacy_sorted
        
        result_info['output_items'] = sorted_items
        self.sorting_history.append(result_info)
        
        return sorted_items
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        parity_stats = self.parity_weighting.get_parity_statistics()
        layer_info = self.conjugacy_partitioner.get_layer_info()
        
        orbits_info = self.burnside_calculator.calculate_orbits_detailed()
        
        return {
            'n': self.n,
            'parity_statistics': parity_stats,
            'conjugacy_layer_info': layer_info,
            'burnside_orbits': orbits_info,
            'sorting_history_count': len(self.sorting_history)
        }
    
    def set_category_memberships(self, memberships: Dict[str, np.ndarray]):
        """设置类别隶属度"""
        self.parity_weighting.set_category_memberships(memberships)


from scipy.optimize import linprog, minimize, LinearConstraint
from scipy.linalg import lstsq


class MembershipProbabilityMapping:
    """
    隶属度-概率高阶映射模型
    构建题目分类隶属度与随机抽取概率的三次多项式拟合映射
    引入KL散度监测分类分布与随机分布偏差，触发概率密度函数迭代修正
    """
    
    def __init__(self, n_categories: int, polynomial_degree: int = 3):
        self.n_categories = n_categories
        self.polynomial_degree = polynomial_degree
        
        self.membership_coefficients: Dict[str, np.ndarray] = {}
        self.probability_coefficients: Dict[str, np.ndarray] = {}
        
        self.membership_history: List[Dict[str, np.ndarray]] = []
        self.probability_history: List[Dict[str, np.ndarray]] = []
        self.kl_divergence_history: List[float] = []
        
        self._epsilon = 1e-10
        
    def _build_polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """构建多项式特征矩阵"""
        n_samples = len(x)
        features = np.zeros((n_samples, self.polynomial_degree + 1))
        for i in range(self.polynomial_degree + 1):
            features[:, i] = x ** i
        return features
    
    def _cubic_polynomial_fit(self, membership: np.ndarray, probability: np.ndarray) -> np.ndarray:
        """三次多项式拟合映射"""
        membership = np.clip(membership, self._epsilon, 1.0 - self._epsilon)
        probability = np.clip(probability, self._epsilon, 1.0 - self._epsilon)
        
        X = self._build_polynomial_features(membership)
        
        try:
            coeffs, _, _, _ = lstsq(X, probability)
        except Exception:
            coeffs = np.zeros(self.polynomial_degree + 1)
            coeffs[0] = np.mean(probability)
            
        return coeffs
    
    def set_membership_probability_mapping(self, category: str, 
                                           memberships: np.ndarray, 
                                           probabilities: np.ndarray):
        """设置隶属度到概率的映射"""
        if len(memberships) != len(probabilities):
            raise ValueError("隶属度和概率数组长度必须相同")
            
        coeffs = self._cubic_polynomial_fit(memberships, probabilities)
        self.membership_coefficients[category] = coeffs
        
        self.membership_history.append({category: memberships.copy()})
        self.probability_history.append({category: probabilities.copy()})
        
    def predict_probability(self, category: str, membership: np.ndarray) -> np.ndarray:
        """根据隶属度预测概率"""
        if category not in self.membership_coefficients:
            raise ValueError(f"类别 {category} 未设置映射系数")
            
        coeffs = self.membership_coefficients[category]
        membership = np.clip(membership, self._epsilon, 1.0 - self._epsilon)
        
        X = self._build_polynomial_features(membership)
        probability = X @ coeffs
        
        probability = np.clip(probability, self._epsilon, 1.0 - self._epsilon)
        
        return probability
    
    def compute_kl_divergence(self, p_distribution: np.ndarray, 
                              q_distribution: np.ndarray) -> float:
        """计算KL散度 D_KL(P||Q)"""
        p = np.clip(p_distribution, self._epsilon, 1.0)
        q = np.clip(q_distribution, self._epsilon, 1.0)
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        kl_div = np.sum(p * np.log(p / q))
        
        return float(np.clip(kl_div, 0, 1e10))
    
    def monitor_distribution_deviation(self, classification_dist: np.ndarray, 
                                       random_dist: np.ndarray) -> Dict[str, float]:
        """监测分类分布与随机分布偏差"""
        kl_div = self.compute_kl_divergence(classification_dist, random_dist)
        self.kl_divergence_history.append(kl_div)
        
        js_div = 0.5 * (self.compute_kl_divergence(classification_dist, random_dist) + 
                        self.compute_kl_divergence(random_dist, classification_dist))
        
        total_variation = 0.5 * np.sum(np.abs(classification_dist - random_dist))
        
        return {
            'kl_divergence': kl_div,
            'js_divergence': js_div,
            'total_variation': total_variation,
            'threshold_exceeded': kl_div > 0.1
        }
    
    def iterative_correction(self, category: str, 
                            target_distribution: np.ndarray,
                            max_iterations: int = 100,
                            tolerance: float = 1e-6) -> Dict[str, Any]:
        """迭代修正概率密度函数"""
        if category not in self.membership_coefficients:
            raise ValueError(f"类别 {category} 未设置映射系数")
            
        current_coeffs = self.membership_coefficients[category].copy()
        n_points = len(target_distribution)
        
        membership_range = np.linspace(0.01, 0.99, n_points)
        
        correction_history = []
        
        for iteration in range(max_iterations):
            current_prob = self.predict_probability(category, membership_range)
            current_prob = current_prob / np.sum(current_prob)
            
            kl_div = self.compute_kl_divergence(target_distribution, current_prob)
            correction_history.append({
                'iteration': iteration,
                'kl_divergence': kl_div,
                'coefficients': current_coeffs.copy()
            })
            
            if kl_div < tolerance:
                break
                
            gradient = (target_distribution - current_prob) * membership_range
            learning_rate = 0.01 / (1 + iteration * 0.1)
            
            for i in range(1, len(current_coeffs)):
                if i <= len(gradient):
                    current_coeffs[i] += learning_rate * np.sum(gradient)
                    
            current_coeffs = np.clip(current_coeffs, -10, 10)
            
        self.membership_coefficients[category] = current_coeffs
        
        return {
            'final_kl_divergence': kl_div,
            'iterations': iteration + 1,
            'converged': kl_div < tolerance,
            'correction_history': correction_history
        }
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """获取映射统计信息"""
        return {
            'n_categories': self.n_categories,
            'polynomial_degree': self.polynomial_degree,
            'categories_configured': list(self.membership_coefficients.keys()),
            'kl_divergence_history': self.kl_divergence_history[-10:] if self.kl_divergence_history else [],
            'mean_kl_divergence': np.mean(self.kl_divergence_history) if self.kl_divergence_history else 0.0
        }


class BoundaryConstrainedSubspace:
    """
    分类边界约束随机子空间模型
    基于分类特征空间的高阶统计边界
    通过线性规划可行域随机采样生成约束下的随机特征子空间
    引入拉格朗日乘子法优化子空间与分类的适配性
    """
    
    def __init__(self, n_features: int, n_categories: int):
        self.n_features = n_features
        self.n_categories = n_categories
        
        self.category_centers: Dict[int, np.ndarray] = {}
        self.category_radii: Dict[int, float] = {}
        self.category_covariances: Dict[int, np.ndarray] = {}
        
        self.boundary_constraints: List[LinearConstraint] = []
        self.lagrange_multipliers: Dict[int, np.ndarray] = {}
        
        self._epsilon = 1e-10
        
    def set_category_boundary(self, category: int, 
                              center: np.ndarray, 
                              radius: float,
                              covariance: Optional[np.ndarray] = None):
        """设置分类边界"""
        if len(center) != self.n_features:
            raise ValueError(f"中心点维度 {len(center)} 与特征维度 {self.n_features} 不匹配")
            
        self.category_centers[category] = center.copy()
        self.category_radii[category] = max(radius, self._epsilon)
        
        if covariance is not None:
            self.category_covariances[category] = covariance.copy()
        else:
            self.category_covariances[category] = np.eye(self.n_features) * radius**2
            
    def compute_high_order_statistics(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """计算高阶统计边界"""
        n_samples = len(features)
        
        mean = np.mean(features, axis=0)
        centered = features - mean
        
        covariance = (centered.T @ centered) / max(n_samples - 1, 1)
        
        skewness = np.zeros(self.n_features)
        kurtosis = np.zeros(self.n_features)
        
        std = np.sqrt(np.diag(covariance) + self._epsilon)
        
        for i in range(self.n_features):
            normalized = centered[:, i] / (std[i] + self._epsilon)
            skewness[i] = np.mean(normalized ** 3)
            kurtosis[i] = np.mean(normalized ** 4) - 3
            
        return {
            'mean': mean,
            'covariance': covariance,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'std': std
        }
    
    def build_linear_constraints(self, category: int) -> List[Dict[str, np.ndarray]]:
        """构建线性规划约束"""
        if category not in self.category_centers:
            raise ValueError(f"类别 {category} 未设置边界")
            
        center = self.category_centers[category]
        radius = self.category_radii[category]
        
        constraints = []
        
        for i in range(self.n_features):
            constraint_lower = {
                'type': 'ineq',
                'fun': lambda x, idx=i, c=center, r=radius: x[idx] - (c[idx] - r),
                'jac': lambda x, idx=i: self._sparse_gradient(x, idx, 1.0)
            }
            constraint_upper = {
                'type': 'ineq',
                'fun': lambda x, idx=i, c=center, r=radius: (c[idx] + r) - x[idx],
                'jac': lambda x, idx=i: self._sparse_gradient(x, idx, -1.0)
            }
            constraints.extend([constraint_lower, constraint_upper])
            
        return constraints
    
    def _sparse_gradient(self, x: np.ndarray, idx: int, sign: float) -> np.ndarray:
        """计算稀疏梯度"""
        grad = np.zeros(self.n_features)
        grad[idx] = sign
        return grad
    
    def sample_from_feasible_region(self, category: int, 
                                    n_samples: int = 1,
                                    method: str = 'rejection') -> np.ndarray:
        """从可行域随机采样"""
        if category not in self.category_centers:
            raise ValueError(f"类别 {category} 未设置边界")
            
        center = self.category_centers[category]
        radius = self.category_radii[category]
        cov = self.category_covariances[category]
        
        samples = np.zeros((n_samples, self.n_features))
        
        if method == 'rejection':
            count = 0
            while count < n_samples:
                candidate = np.random.multivariate_normal(center, cov)
                
                if np.linalg.norm(candidate - center) <= radius * 2:
                    samples[count] = candidate
                    count += 1
                    
        elif method == 'direct':
            angles = np.random.randn(n_samples, self.n_features)
            angles = angles / (np.linalg.norm(angles, axis=1, keepdims=True) + self._epsilon)
            
            radii = np.random.uniform(0, radius, n_samples)
            samples = center + angles * radii[:, np.newaxis]
            
        return samples
    
    def lagrangian_optimization(self, category: int, 
                                target_features: np.ndarray,
                                max_iterations: int = 100) -> Dict[str, Any]:
        """拉格朗日乘子法优化子空间适配性"""
        if category not in self.category_centers:
            raise ValueError(f"类别 {category} 未设置边界")
            
        center = self.category_centers[category]
        radius = self.category_radii[category]
        
        n_constraints = self.n_features * 2
        lagrange = np.zeros(n_constraints)
        
        def objective(x):
            dist = np.linalg.norm(x - target_features)
            return dist
        
        def objective_grad(x):
            diff = x - target_features
            norm = np.linalg.norm(diff) + self._epsilon
            return diff / norm
        
        def constraint_func(x):
            constraints = []
            for i in range(self.n_features):
                constraints.append(x[i] - (center[i] - radius))
                constraints.append((center[i] + radius) - x[i])
            return np.array(constraints)
        
        x0 = center.copy()
        
        history = []
        
        for iteration in range(max_iterations):
            constraints_val = constraint_func(x0)
            
            violation = np.minimum(constraints_val, 0)
            
            lagrange = np.maximum(lagrange - 0.1 * violation, 0)
            
            grad = objective_grad(x0)
            
            for i in range(self.n_features):
                grad[i] += lagrange[2*i] - lagrange[2*i + 1]
                
            step_size = 0.1 / (1 + iteration * 0.05)
            x0 = x0 - step_size * grad
            
            x0 = np.clip(x0, center - radius, center + radius)
            
            history.append({
                'iteration': iteration,
                'objective': objective(x0),
                'constraint_violation': np.sum(np.abs(violation))
            })
            
        self.lagrange_multipliers[category] = lagrange
        
        return {
            'optimal_features': x0,
            'optimal_objective': objective(x0),
            'lagrange_multipliers': lagrange,
            'optimization_history': history
        }
    
    def generate_constrained_subspace(self, category: int, 
                                      subspace_dim: int,
                                      n_samples: int = 100) -> np.ndarray:
        """生成约束下的随机特征子空间"""
        if subspace_dim > self.n_features:
            raise ValueError(f"子空间维度 {subspace_dim} 不能大于特征维度 {self.n_features}")
            
        samples = self.sample_from_feasible_region(category, n_samples)
        
        centered = samples - np.mean(samples, axis=0)
        
        cov = (centered.T @ centered) / (n_samples - 1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        subspace_basis = eigenvectors[:, :subspace_dim]
        
        return subspace_basis
    
    def get_subspace_statistics(self) -> Dict[str, Any]:
        """获取子空间统计信息"""
        return {
            'n_features': self.n_features,
            'n_categories': self.n_categories,
            'categories_configured': list(self.category_centers.keys()),
            'lagrange_multipliers_set': list(self.lagrange_multipliers.keys())
        }


class MisclassificationRiskEntropyModel:
    """
    错分风险熵驱动随机迭代模型
    计算分类错分风险熵（基于分类置信度）
    将熵值作为随机抽样惩罚项引入目标函数
    通过牛顿-拉夫逊法迭代调整随机抽样权重
    """
    
    def __init__(self, n_categories: int, n_samples: int):
        self.n_categories = n_categories
        self.n_samples = n_samples
        
        self.classification_confidence: np.ndarray = np.ones(n_samples) / n_categories
        self.sampling_weights: np.ndarray = np.ones(n_samples) / n_samples
        
        self.risk_entropy_history: List[float] = []
        self.weight_history: List[np.ndarray] = []
        
        self._epsilon = 1e-10
        self._hessian_reg = 1e-6
        
    def set_classification_confidence(self, confidence_matrix: np.ndarray):
        """设置分类置信度矩阵 (n_samples, n_categories)"""
        if confidence_matrix.shape != (self.n_samples, self.n_categories):
            raise ValueError(f"置信度矩阵形状 {confidence_matrix.shape} 与预期 ({self.n_samples}, {self.n_categories}) 不匹配")
            
        self.classification_confidence = np.clip(confidence_matrix, self._epsilon, 1.0 - self._epsilon)
        
        row_sums = np.sum(self.classification_confidence, axis=1, keepdims=True)
        self.classification_confidence = self.classification_confidence / row_sums
        
    def compute_misclassification_risk_entropy(self, sample_idx: int) -> float:
        """计算单个样本的错分风险熵"""
        conf = self.classification_confidence[sample_idx]
        
        conf = np.clip(conf, self._epsilon, 1.0 - self._epsilon)
        
        entropy = -np.sum(conf * np.log(conf))
        
        max_entropy = np.log(self.n_categories)
        normalized_entropy = entropy / (max_entropy + self._epsilon)
        
        return float(normalized_entropy)
    
    def compute_all_risk_entropies(self) -> np.ndarray:
        """计算所有样本的错分风险熵"""
        entropies = np.zeros(self.n_samples)
        
        for i in range(self.n_samples):
            entropies[i] = self.compute_misclassification_risk_entropy(i)
            
        return entropies
    
    def entropy_weighted_objective(self, weights: np.ndarray, 
                                   entropies: np.ndarray,
                                   regularization: float = 0.1) -> float:
        """熵加权目标函数"""
        weights = np.clip(weights, self._epsilon, 1.0)
        weights = weights / np.sum(weights)
        
        entropy_penalty = np.sum(entropies * weights)
        
        uniform_weights = np.ones(self.n_samples) / self.n_samples
        kl_penalty = np.sum(weights * np.log(weights / uniform_weights))
        
        variance_penalty = np.var(weights)
        
        objective = entropy_penalty + regularization * kl_penalty + 0.01 * variance_penalty
        
        return objective
    
    def objective_gradient(self, weights: np.ndarray, 
                          entropies: np.ndarray,
                          regularization: float = 0.1) -> np.ndarray:
        """计算目标函数梯度"""
        weights = np.clip(weights, self._epsilon, 1.0)
        weights_normalized = weights / np.sum(weights)
        
        uniform_weights = np.ones(self.n_samples) / self.n_samples
        
        grad = entropies.copy()
        
        grad += regularization * (1 + np.log(weights_normalized / uniform_weights))
        
        grad += 0.01 * 2 * (weights_normalized - np.mean(weights_normalized))
        
        return grad
    
    def objective_hessian(self, weights: np.ndarray, 
                         regularization: float = 0.1) -> np.ndarray:
        """计算目标函数Hessian矩阵"""
        weights = np.clip(weights, self._epsilon, 1.0)
        weights_normalized = weights / np.sum(weights)
        
        hessian = np.eye(self.n_samples) * self._hessian_reg
        
        for i in range(self.n_samples):
            hessian[i, i] += regularization / weights_normalized[i]
            
        return hessian
    
    def newton_raphson_iteration(self, max_iterations: int = 100, 
                                 tolerance: float = 1e-8) -> Dict[str, Any]:
        """牛顿-拉夫逊法迭代调整抽样权重"""
        entropies = self.compute_all_risk_entropies()
        
        weights = self.sampling_weights.copy()
        
        iteration_history = []
        
        for iteration in range(max_iterations):
            obj_val = self.entropy_weighted_objective(weights, entropies)
            grad = self.objective_gradient(weights, entropies)
            
            self.risk_entropy_history.append(obj_val)
            self.weight_history.append(weights.copy())
            
            iteration_history.append({
                'iteration': iteration,
                'objective': obj_val,
                'gradient_norm': np.linalg.norm(grad),
                'max_weight': np.max(weights),
                'min_weight': np.min(weights)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
                
            hessian = self.objective_hessian(weights)
            
            try:
                delta = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hessian, grad, rcond=None)[0]
                
            step_size = 1.0
            for _ in range(10):
                new_weights = weights - step_size * delta
                if np.all(new_weights > 0):
                    break
                step_size *= 0.5
                
            weights = weights - step_size * delta
            
            weights = np.clip(weights, self._epsilon, 1.0)
            weights = weights / np.sum(weights)
            
        self.sampling_weights = weights
        
        return {
            'final_weights': weights,
            'final_objective': obj_val,
            'iterations': iteration + 1,
            'converged': np.linalg.norm(grad) < tolerance,
            'iteration_history': iteration_history
        }
    
    def sample_with_entropy_penalty(self, n_draws: int = 1) -> np.ndarray:
        """带熵惩罚的随机抽样"""
        weights = self.sampling_weights / np.sum(self.sampling_weights)
        
        indices = np.random.choice(self.n_samples, size=n_draws, replace=True, p=weights)
        
        return indices
    
    def update_weights_with_feedback(self, sampled_indices: np.ndarray, 
                                     rewards: np.ndarray):
        """根据反馈更新权重"""
        for idx, reward in zip(sampled_indices, rewards):
            self.sampling_weights[idx] *= (1 + 0.1 * reward)
            
        self.sampling_weights = np.clip(self.sampling_weights, self._epsilon, 1.0)
        self.sampling_weights = self.sampling_weights / np.sum(self.sampling_weights)
        
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        entropies = self.compute_all_risk_entropies()
        
        return {
            'n_categories': self.n_categories,
            'n_samples': self.n_samples,
            'mean_risk_entropy': float(np.mean(entropies)),
            'std_risk_entropy': float(np.std(entropies)),
            'weight_entropy': float(-np.sum(self.sampling_weights * np.log(self.sampling_weights + self._epsilon))),
            'entropy_history_length': len(self.risk_entropy_history),
            'current_weights_stats': {
                'mean': float(np.mean(self.sampling_weights)),
                'std': float(np.std(self.sampling_weights)),
                'min': float(np.min(self.sampling_weights)),
                'max': float(np.max(self.sampling_weights))
            }
        }


from scipy import stats
from scipy.stats import ks_2samp, skew, kurtosis, entropy


class HigherOrderMomentsAnalyzer:
    """
    高阶矩分析体系
    计算分类结果分布的三阶矩（偏度）、四阶矩（峰度）及交叉矩
    引入K-S检验验证分类分布与理论分布拟合度
    结合bootstrap重采样迭代计算矩估计置信区间
    """
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.analysis_results: Dict[str, Any] = {}
        
    def compute_moments(self, data: np.ndarray) -> Dict[str, float]:
        """计算数据的高阶矩统计量"""
        data = np.asarray(data).flatten()
        
        if len(data) < 3:
            raise ValueError("数据点数不足，至少需要3个数据点")
        
        n = len(data)
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        if std_val == 0:
            std_val = 1e-10
        
        central_moments = {}
        for k in range(1, 5):
            central_moments[k] = np.mean((data - mean_val) ** k)
        
        skewness = skew(data, bias=False)
        kurt = kurtosis(data, bias=False, fisher=True)
        
        standardized_moments = {}
        for k in range(3, 5):
            if std_val > 0:
                standardized_moments[k] = central_moments[k] / (std_val ** k)
            else:
                standardized_moments[k] = 0.0
        
        return {
            'mean': mean_val,
            'variance': central_moments[2],
            'std': std_val,
            'skewness': skewness,
            'kurtosis': kurt,
            'third_moment': central_moments[3],
            'fourth_moment': central_moments[4],
            'standardized_third_moment': standardized_moments[3],
            'standardized_fourth_moment': standardized_moments[4]
        }
    
    def compute_cross_moments(self, data1: np.ndarray, data2: np.ndarray, 
                              order: Tuple[int, int] = (1, 1)) -> float:
        """计算两个数据序列的交叉矩 E[X^m * Y^n]"""
        data1 = np.asarray(data1).flatten()
        data2 = np.asarray(data2).flatten()
        
        if len(data1) != len(data2):
            raise ValueError("两个数据序列长度必须相同")
        
        m, n = order
        cross_moment = np.mean((data1 ** m) * (data2 ** n))
        
        return cross_moment
    
    def compute_covariance_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        """计算多变量数据的协方差矩阵"""
        if data_matrix.ndim == 1:
            data_matrix = data_matrix.reshape(-1, 1)
        
        return np.cov(data_matrix, rowvar=False)
    
    def ks_goodness_of_fit(self, observed: np.ndarray, 
                           theoretical_dist: str = 'norm',
                           dist_params: Optional[Tuple] = None) -> Dict[str, float]:
        """K-S检验验证分类分布与理论分布拟合度"""
        observed = np.asarray(observed).flatten()
        
        if theoretical_dist == 'norm':
            if dist_params is None:
                mu, sigma = np.mean(observed), np.std(observed, ddof=1)
            else:
                mu, sigma = dist_params
            theoretical_samples = np.random.normal(mu, sigma, 10000)
            
        elif theoretical_dist == 'uniform':
            if dist_params is None:
                a, b = np.min(observed), np.max(observed)
            else:
                a, b = dist_params
            theoretical_samples = np.random.uniform(a, b, 10000)
            
        elif theoretical_dist == 'expon':
            if dist_params is None:
                scale = np.mean(observed)
            else:
                scale = dist_params[0]
            theoretical_samples = np.random.exponential(scale, 10000)
            
        else:
            raise ValueError(f"不支持的理论分布类型: {theoretical_dist}")
        
        ks_stat, p_value = ks_2samp(observed, theoretical_samples)
        
        return {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'theoretical_dist': theoretical_dist,
            'is_fitted': p_value > (1 - self.confidence_level)
        }
    
    def bootstrap_moment_confidence_interval(self, data: np.ndarray,
                                             moment_type: str = 'skewness') -> Dict[str, float]:
        """Bootstrap重采样计算矩估计置信区间"""
        data = np.asarray(data).flatten()
        n = len(data)
        
        bootstrap_estimates = []
        
        for _ in range(self.n_bootstrap):
            resample_idx = np.random.choice(n, size=n, replace=True)
            resample_data = data[resample_idx]
            
            if moment_type == 'skewness':
                estimate = skew(resample_data, bias=False)
            elif moment_type == 'kurtosis':
                estimate = kurtosis(resample_data, bias=False, fisher=True)
            elif moment_type == 'mean':
                estimate = np.mean(resample_data)
            elif moment_type == 'variance':
                estimate = np.var(resample_data, ddof=1)
            else:
                raise ValueError(f"不支持的矩类型: {moment_type}")
            
            bootstrap_estimates.append(estimate)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
        ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
        
        return {
            'moment_type': moment_type,
            'point_estimate': np.mean(bootstrap_estimates),
            'std_error': np.std(bootstrap_estimates),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': self.confidence_level,
            'n_bootstrap': self.n_bootstrap
        }
    
    def analyze_classification_distribution(self, 
                                            class_probabilities: np.ndarray,
                                            class_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """分析分类结果分布的高阶矩特征"""
        class_probabilities = np.asarray(class_probabilities)
        
        if class_probabilities.ndim == 1:
            class_probabilities = class_probabilities.reshape(1, -1)
        
        n_samples, n_classes = class_probabilities.shape
        
        moments_results = []
        for i in range(n_samples):
            sample_moments = self.compute_moments(class_probabilities[i])
            moments_results.append(sample_moments)
        
        aggregated_moments = {}
        for key in moments_results[0].keys():
            values = [m[key] for m in moments_results]
            aggregated_moments[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        cross_moments_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    cross_moments_matrix[i, j] = self.compute_cross_moments(
                        class_probabilities[:, i], 
                        class_probabilities[:, j]
                    )
        
        ks_results = []
        for i in range(n_classes):
            ks_result = self.ks_goodness_of_fit(class_probabilities[:, i], 'norm')
            ks_results.append(ks_result)
        
        skewness_ci = self.bootstrap_moment_confidence_interval(
            class_probabilities.flatten(), 'skewness'
        )
        kurtosis_ci = self.bootstrap_moment_confidence_interval(
            class_probabilities.flatten(), 'kurtosis'
        )
        
        self.analysis_results = {
            'n_samples': n_samples,
            'n_classes': n_classes,
            'aggregated_moments': aggregated_moments,
            'cross_moments_matrix': cross_moments_matrix,
            'ks_goodness_of_fit': ks_results,
            'bootstrap_confidence_intervals': {
                'skewness': skewness_ci,
                'kurtosis': kurtosis_ci
            }
        }
        
        return self.analysis_results
    
    def get_moment_summary(self) -> str:
        """获取矩分析的文本摘要"""
        if not self.analysis_results:
            return "尚未进行矩分析"
        
        results = self.analysis_results
        summary = []
        summary.append("=== 高阶矩分析摘要 ===")
        summary.append(f"样本数: {results['n_samples']}, 类别数: {results['n_classes']}")
        
        agg = results['aggregated_moments']
        summary.append(f"偏度: {agg['skewness']['mean']:.4f} ± {agg['skewness']['std']:.4f}")
        summary.append(f"峰度: {agg['kurtosis']['mean']:.4f} ± {agg['kurtosis']['std']:.4f}")
        
        skew_ci = results['bootstrap_confidence_intervals']['skewness']
        summary.append(f"偏度95%置信区间: [{skew_ci['ci_lower']:.4f}, {skew_ci['ci_upper']:.4f}]")
        
        kurt_ci = results['bootstrap_confidence_intervals']['kurtosis']
        summary.append(f"峰度95%置信区间: [{kurt_ci['ci_lower']:.4f}, {kurt_ci['ci_upper']:.4f}]")
        
        return "\n".join(summary)


class ErgodicityAnalyzer:
    """
    随机结果遍历性分析
    验证随机抽取/排序结果的时间/空间遍历性
    引入遍历熵计算随机均匀性
    结合大数定律迭代验证结果收敛速度
    """
    
    def __init__(self, n_bins: int = 10, convergence_threshold: float = 0.01):
        self.n_bins = n_bins
        self.convergence_threshold = convergence_threshold
        self.analysis_results: Dict[str, Any] = {}
        
    def compute_ergodic_entropy(self, sequence: np.ndarray) -> float:
        """计算遍历熵"""
        sequence = np.asarray(sequence).flatten()
        
        hist, _ = np.histogram(sequence, bins=self.n_bins, density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        
        ergodic_entropy = entropy(hist)
        
        max_entropy = np.log(self.n_bins)
        normalized_entropy = ergodic_entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def test_temporal_ergodicity(self, time_series: np.ndarray, 
                                  window_size: Optional[int] = None) -> Dict[str, Any]:
        """验证时间遍历性"""
        time_series = np.asarray(time_series).flatten()
        n = len(time_series)
        
        if window_size is None:
            window_size = max(n // 10, 10)
        
        n_windows = n // window_size
        window_entropies = []
        window_means = []
        window_vars = []
        
        for i in range(n_windows):
            window_data = time_series[i * window_size:(i + 1) * window_size]
            window_entropies.append(self.compute_ergodic_entropy(window_data))
            window_means.append(np.mean(window_data))
            window_vars.append(np.var(window_data))
        
        global_entropy = self.compute_ergodic_entropy(time_series)
        global_mean = np.mean(time_series)
        global_var = np.var(time_series)
        
        mean_stability = 1 - np.std(window_means) / (np.abs(global_mean) + 1e-10)
        var_stability = 1 - np.std(window_vars) / (global_var + 1e-10)
        entropy_stability = 1 - np.std(window_entropies) / (global_entropy + 1e-10)
        
        is_ergodic = (mean_stability > 0.8 and var_stability > 0.8 and 
                      entropy_stability > 0.8 and global_entropy > 0.5)
        
        return {
            'global_entropy': global_entropy,
            'global_mean': global_mean,
            'global_variance': global_var,
            'window_entropies': np.array(window_entropies),
            'window_means': np.array(window_means),
            'window_variances': np.array(window_vars),
            'mean_stability': mean_stability,
            'variance_stability': var_stability,
            'entropy_stability': entropy_stability,
            'is_temporally_ergodic': is_ergodic,
            'window_size': window_size
        }
    
    def test_spatial_ergodicity(self, spatial_data: np.ndarray) -> Dict[str, Any]:
        """验证空间遍历性"""
        spatial_data = np.asarray(spatial_data)
        
        if spatial_data.ndim == 1:
            spatial_data = spatial_data.reshape(-1, 1)
        
        n_samples, n_dims = spatial_data.shape
        
        dim_entropies = []
        for d in range(n_dims):
            dim_entropies.append(self.compute_ergodic_entropy(spatial_data[:, d]))
        
        joint_entropy = self.compute_ergodic_entropy(spatial_data.flatten())
        
        if n_dims > 1:
            cov_matrix = np.cov(spatial_data.T)
            correlation_matrix = np.corrcoef(spatial_data.T)
            
            off_diagonal_corrs = correlation_matrix[np.triu_indices(n_dims, k=1)]
            spatial_independence = 1 - np.mean(np.abs(off_diagonal_corrs))
        else:
            cov_matrix = np.array([[np.var(spatial_data)]])
            correlation_matrix = np.array([[1.0]])
            spatial_independence = 1.0
        
        uniformity_score = np.mean(dim_entropies)
        is_spatially_ergodic = uniformity_score > 0.5 and spatial_independence > 0.5
        
        return {
            'n_dimensions': n_dims,
            'dimension_entropies': np.array(dim_entropies),
            'joint_entropy': joint_entropy,
            'uniformity_score': uniformity_score,
            'spatial_independence': spatial_independence,
            'covariance_matrix': cov_matrix,
            'correlation_matrix': correlation_matrix,
            'is_spatially_ergodic': is_spatially_ergodic
        }
    
    def verify_law_of_large_numbers(self, sequence: np.ndarray,
                                    true_mean: Optional[float] = None) -> Dict[str, Any]:
        """结合大数定律验证结果收敛速度"""
        sequence = np.asarray(sequence).flatten()
        n = len(sequence)
        
        if true_mean is None:
            true_mean = np.mean(sequence)
        
        cumulative_means = np.cumsum(sequence) / np.arange(1, n + 1)
        
        convergence_errors = np.abs(cumulative_means - true_mean)
        
        window = min(20, n // 5)
        convergence_speeds = []
        for i in range(window, n):
            recent_error = np.mean(convergence_errors[i-window:i])
            earlier_error = np.mean(convergence_errors[max(0, i-2*window):i-window]) if i >= 2*window else recent_error
            speed = (earlier_error - recent_error) / (earlier_error + 1e-10)
            convergence_speeds.append(speed)
        
        convergence_speeds = np.array(convergence_speeds)
        
        final_error = convergence_errors[-1]
        is_converged = final_error < self.convergence_threshold
        
        n_for_convergence = None
        for i, error in enumerate(convergence_errors):
            if error < self.convergence_threshold:
                n_for_convergence = i + 1
                break
        
        return {
            'true_mean': true_mean,
            'cumulative_means': cumulative_means,
            'convergence_errors': convergence_errors,
            'final_error': final_error,
            'mean_convergence_speed': np.mean(convergence_speeds) if len(convergence_speeds) > 0 else 0,
            'is_converged': is_converged,
            'n_for_convergence': n_for_convergence,
            'convergence_threshold': self.convergence_threshold
        }
    
    def analyze_random_results(self, results_sequence: np.ndarray,
                               spatial_dimensions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """综合分析随机结果的遍历性"""
        results_sequence = np.asarray(results_sequence)
        
        temporal_result = self.test_temporal_ergodicity(results_sequence)
        lln_result = self.verify_law_of_large_numbers(results_sequence)
        
        spatial_result = None
        if spatial_dimensions is not None:
            spatial_result = self.test_spatial_ergodicity(spatial_dimensions)
        
        overall_ergodicity = temporal_result['is_temporally_ergodic'] and lln_result['is_converged']
        if spatial_result is not None:
            overall_ergodicity = overall_ergodicity and spatial_result['is_spatially_ergodic']
        
        self.analysis_results = {
            'temporal_ergodicity': temporal_result,
            'law_of_large_numbers': lln_result,
            'spatial_ergodicity': spatial_result,
            'overall_ergodicity': overall_ergodicity
        }
        
        return self.analysis_results
    
    def get_ergodicity_summary(self) -> str:
        """获取遍历性分析的文本摘要"""
        if not self.analysis_results:
            return "尚未进行遍历性分析"
        
        results = self.analysis_results
        summary = []
        summary.append("=== 遍历性分析摘要 ===")
        
        temporal = results['temporal_ergodicity']
        summary.append(f"时间遍历性: {'通过' if temporal['is_temporally_ergodic'] else '未通过'}")
        summary.append(f"  - 全局熵: {temporal['global_entropy']:.4f}")
        summary.append(f"  - 均值稳定性: {temporal['mean_stability']:.4f}")
        
        lln = results['law_of_large_numbers']
        summary.append(f"大数定律收敛: {'是' if lln['is_converged'] else '否'}")
        summary.append(f"  - 最终误差: {lln['final_error']:.6f}")
        if lln['n_for_convergence']:
            summary.append(f"  - 收敛所需样本数: {lln['n_for_convergence']}")
        
        if results['spatial_ergodicity'] is not None:
            spatial = results['spatial_ergodicity']
            summary.append(f"空间遍历性: {'通过' if spatial['is_spatially_ergodic'] else '未通过'}")
            summary.append(f"  - 均匀性得分: {spatial['uniformity_score']:.4f}")
        
        summary.append(f"整体遍历性: {'通过' if results['overall_ergodicity'] else '未通过'}")
        
        return "\n".join(summary)


class GlobalSensitivityAnalyzer:
    """
    全局灵敏度分析
    计算高阶模型参数的偏导数矩阵
    引入Sobol指数量化各参数对结果的贡献度
    提升结果归因的数学深度
    """
    
    def __init__(self, n_samples: int = 1000, n_bootstrap: int = 100):
        self.n_samples = n_samples
        self.n_bootstrap = n_bootstrap
        self.analysis_results: Dict[str, Any] = {}
        
    def compute_partial_derivatives(self, model_func: Callable,
                                    params: np.ndarray,
                                    param_names: Optional[List[str]] = None,
                                    delta: float = 1e-5) -> Dict[str, np.ndarray]:
        """计算模型参数的偏导数矩阵"""
        params = np.asarray(params).flatten()
        n_params = len(params)
        
        if param_names is None:
            param_names = [f'param_{i}' for i in range(n_params)]
        
        base_output = model_func(*params)
        
        if np.isscalar(base_output):
            output_shape = (1,)
            base_output = np.array([base_output])
        else:
            base_output = np.asarray(base_output)
            output_shape = base_output.shape
        
        jacobian = np.zeros((*output_shape, n_params))
        
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += delta
            
            params_minus = params.copy()
            params_minus[i] -= delta
            
            output_plus = np.asarray(model_func(*params_plus))
            output_minus = np.asarray(model_func(*params_minus))
            
            if output_plus.shape != output_shape:
                output_plus = output_plus.flatten()[:np.prod(output_shape)].reshape(output_shape)
            if output_minus.shape != output_shape:
                output_minus = output_minus.flatten()[:np.prod(output_shape)].reshape(output_shape)
            
            jacobian[..., i] = (output_plus - output_minus) / (2 * delta)
        
        partial_derivatives = {}
        for i, name in enumerate(param_names):
            partial_derivatives[name] = {
                'values': jacobian[..., i],
                'mean': np.mean(np.abs(jacobian[..., i])),
                'std': np.std(jacobian[..., i]),
                'max': np.max(np.abs(jacobian[..., i]))
            }
        
        return {
            'jacobian_matrix': jacobian,
            'partial_derivatives': partial_derivatives,
            'param_names': param_names,
            'output_shape': output_shape
        }
    
    def compute_sobol_indices(self, model_func: Callable,
                              param_bounds: List[Tuple[float, float]],
                              param_names: Optional[List[str]] = None,
                              order: int = 2) -> Dict[str, Any]:
        """计算Sobol灵敏度指数"""
        n_params = len(param_bounds)
        
        if param_names is None:
            param_names = [f'param_{i}' for i in range(n_params)]
        
        A = np.zeros((self.n_samples, n_params))
        B = np.zeros((self.n_samples, n_params))
        
        for i, (low, high) in enumerate(param_bounds):
            A[:, i] = np.random.uniform(low, high, self.n_samples)
            B[:, i] = np.random.uniform(low, high, self.n_samples)
        
        f_A = np.array([model_func(*A[j]) for j in range(self.n_samples)])
        f_B = np.array([model_func(*B[j]) for j in range(self.n_samples)])
        
        if f_A.ndim == 1:
            f_A = f_A.reshape(-1, 1)
            f_B = f_B.reshape(-1, 1)
        
        n_outputs = f_A.shape[1]
        
        first_order_indices = np.zeros((n_params, n_outputs))
        total_order_indices = np.zeros((n_params, n_outputs))
        
        for i in range(n_params):
            C_i = B.copy()
            C_i[:, i] = A[:, i]
            
            f_C_i = np.array([model_func(*C_i[j]) for j in range(self.n_samples)])
            if f_C_i.ndim == 1:
                f_C_i = f_C_i.reshape(-1, 1)
            
            for out_idx in range(n_outputs):
                var_total = np.var(f_A[:, out_idx], ddof=1)
                
                if var_total > 1e-10:
                    first_order = (np.mean(f_B[:, out_idx] * (f_C_i[:, out_idx] - f_A[:, out_idx])) / 
                                   var_total)
                    first_order_indices[i, out_idx] = first_order
                    
                    total = (np.mean((f_A[:, out_idx] - f_C_i[:, out_idx]) ** 2) / 
                             (2 * var_total))
                    total_order_indices[i, out_idx] = total
        
        sobol_dict = {}
        for i, name in enumerate(param_names):
            sobol_dict[name] = {
                'first_order': first_order_indices[i].tolist(),
                'total_order': total_order_indices[i].tolist(),
                'first_order_mean': np.mean(first_order_indices[i]),
                'total_order_mean': np.mean(total_order_indices[i])
            }
        
        return {
            'first_order_indices': first_order_indices,
            'total_order_indices': total_order_indices,
            'sobol_dict': sobol_dict,
            'param_names': param_names,
            'n_outputs': n_outputs
        }
    
    def bootstrap_sobol_confidence(self, model_func: Callable,
                                   param_bounds: List[Tuple[float, float]],
                                   param_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Bootstrap计算Sobol指数的置信区间"""
        n_params = len(param_bounds)
        
        if param_names is None:
            param_names = [f'param_{i}' for i in range(n_params)]
        
        first_order_bootstrap = np.zeros((self.n_bootstrap, n_params))
        total_order_bootstrap = np.zeros((self.n_bootstrap, n_params))
        
        original_samples = self.n_samples
        
        for b in range(self.n_bootstrap):
            self.n_samples = max(100, original_samples // 2)
            
            try:
                sobol_result = self.compute_sobol_indices(
                    model_func, param_bounds, param_names
                )
                first_order_bootstrap[b] = sobol_result['first_order_indices'].mean(axis=1)
                total_order_bootstrap[b] = sobol_result['total_order_indices'].mean(axis=1)
            except Exception:
                first_order_bootstrap[b] = np.nan
                total_order_bootstrap[b] = np.nan
        
        self.n_samples = original_samples
        
        first_order_bootstrap = first_order_bootstrap[~np.isnan(first_order_bootstrap).any(axis=1)]
        total_order_bootstrap = total_order_bootstrap[~np.isnan(total_order_bootstrap).any(axis=1)]
        
        confidence_intervals = {}
        for i, name in enumerate(param_names):
            if len(first_order_bootstrap) > 0:
                confidence_intervals[name] = {
                    'first_order_ci': (
                        np.percentile(first_order_bootstrap[:, i], 2.5),
                        np.percentile(first_order_bootstrap[:, i], 97.5)
                    ),
                    'total_order_ci': (
                        np.percentile(total_order_bootstrap[:, i], 2.5),
                        np.percentile(total_order_bootstrap[:, i], 97.5)
                    )
                }
            else:
                confidence_intervals[name] = {
                    'first_order_ci': (0.0, 1.0),
                    'total_order_ci': (0.0, 1.0)
                }
        
        return confidence_intervals
    
    def analyze_parameter_sensitivity(self, model_func: Callable,
                                      params: np.ndarray,
                                      param_bounds: List[Tuple[float, float]],
                                      param_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """综合分析参数灵敏度"""
        params = np.asarray(params).flatten()
        
        if param_names is None:
            param_names = [f'param_{i}' for i in range(len(params))]
        
        derivative_result = self.compute_partial_derivatives(
            model_func, params, param_names
        )
        
        sobol_result = self.compute_sobol_indices(
            model_func, param_bounds, param_names
        )
        
        sensitivity_ranking = sorted(
            sobol_result['sobol_dict'].items(),
            key=lambda x: x[1]['total_order_mean'],
            reverse=True
        )
        
        contribution_analysis = {}
        total_contribution = sum(s[1]['total_order_mean'] for s in sensitivity_ranking)
        
        for name, indices in sobol_result['sobol_dict'].items():
            contribution_analysis[name] = {
                'first_order': indices['first_order_mean'],
                'total_order': indices['total_order_mean'],
                'relative_contribution': indices['total_order_mean'] / (total_contribution + 1e-10)
            }
        
        self.analysis_results = {
            'partial_derivatives': derivative_result,
            'sobol_indices': sobol_result,
            'sensitivity_ranking': [s[0] for s in sensitivity_ranking],
            'contribution_analysis': contribution_analysis
        }
        
        return self.analysis_results
    
    def get_sensitivity_summary(self) -> str:
        """获取灵敏度分析的文本摘要"""
        if not self.analysis_results:
            return "尚未进行灵敏度分析"
        
        results = self.analysis_results
        summary = []
        summary.append("=== 全局灵敏度分析摘要 ===")
        
        summary.append("\n参数灵敏度排名 (按总阶Sobol指数):")
        for i, name in enumerate(results['sensitivity_ranking']):
            contrib = results['contribution_analysis'][name]
            summary.append(f"  {i+1}. {name}:")
            summary.append(f"     一阶指数: {contrib['first_order']:.4f}")
            summary.append(f"     总阶指数: {contrib['total_order']:.4f}")
            summary.append(f"     相对贡献: {contrib['relative_contribution']*100:.2f}%")
        
        summary.append("\n偏导数统计:")
        for name, pd in results['partial_derivatives']['partial_derivatives'].items():
            summary.append(f"  {name}: 均值={pd['mean']:.4e}, 标准差={pd['std']:.4e}")
        
        return "\n".join(summary)


def validate_result_analysis_models():
    print("=" * 60)
    print("结果数学分析模型验证")
    print("=" * 60)
    
    print("\n1. 高阶矩分析体系验证")
    np.random.seed(42)
    class_probs = np.random.dirichlet(np.ones(5), size=100)
    
    moments_analyzer = HigherOrderMomentsAnalyzer(confidence_level=0.95, n_bootstrap=500)
    moments_result = moments_analyzer.analyze_classification_distribution(class_probs)
    
    print(f"   样本数: {moments_result['n_samples']}, 类别数: {moments_result['n_classes']}")
    print(f"   偏度均值: {moments_result['aggregated_moments']['skewness']['mean']:.4f}")
    print(f"   峰度均值: {moments_result['aggregated_moments']['kurtosis']['mean']:.4f}")
    
    skew_ci = moments_result['bootstrap_confidence_intervals']['skewness']
    print(f"   偏度95%置信区间: [{skew_ci['ci_lower']:.4f}, {skew_ci['ci_upper']:.4f}]")
    
    print(f"\n{moments_analyzer.get_moment_summary()}")
    
    print("\n2. 随机结果遍历性分析验证")
    random_sequence = np.random.randn(1000)
    random_sequence = np.cumsum(random_sequence) / np.sqrt(np.arange(1, 1001))
    
    ergodicity_analyzer = ErgodicityAnalyzer(n_bins=10, convergence_threshold=0.1)
    ergodicity_result = ergodicity_analyzer.analyze_random_results(random_sequence)
    
    temporal = ergodicity_result['temporal_ergodicity']
    print(f"   时间遍历性: {'通过' if temporal['is_temporally_ergodic'] else '未通过'}")
    print(f"   全局熵: {temporal['global_entropy']:.4f}")
    print(f"   均值稳定性: {temporal['mean_stability']:.4f}")
    
    lln = ergodicity_result['law_of_large_numbers']
    print(f"   大数定律收敛: {'是' if lln['is_converged'] else '否'}")
    print(f"   最终误差: {lln['final_error']:.6f}")
    
    print(f"\n{ergodicity_analyzer.get_ergodicity_summary()}")
    
    print("\n3. 全局灵敏度分析验证")
    
    def test_model(x1, x2, x3):
        return x1**2 + 2*x2 + np.sin(x3)
    
    params = np.array([1.0, 2.0, 0.5])
    param_bounds = [(0, 2), (0, 4), (-np.pi, np.pi)]
    param_names = ['x1', 'x2', 'x3']
    
    sensitivity_analyzer = GlobalSensitivityAnalyzer(n_samples=500, n_bootstrap=50)
    sensitivity_result = sensitivity_analyzer.analyze_parameter_sensitivity(
        test_model, params, param_bounds, param_names
    )
    
    print(f"   灵敏度排名: {sensitivity_result['sensitivity_ranking']}")
    
    for name, contrib in sensitivity_result['contribution_analysis'].items():
        print(f"   {name}: 一阶={contrib['first_order']:.4f}, 总阶={contrib['total_order']:.4f}")
    
    print(f"\n{sensitivity_analyzer.get_sensitivity_summary()}")
    
    print("\n" + "=" * 60)
    print("结果数学分析模型验证完成!")
    print("=" * 60)


def validate_permutation_group_models():
    print("=" * 60)
    print("置换群高阶随机排序模型验证")
    print("=" * 60)
    
    print("\n1. n阶置换群子群分解验证 (n=5)")
    pg = PermutationGroup(5)
    subgroups = pg.decompose_subgroups()
    print(f"   S_5 元素数: {len(subgroups['S_n'])} (期望: 120)")
    print(f"   A_5 元素数: {len(subgroups['A_n'])} (期望: 60)")
    print(f"   平凡子群元素数: {len(subgroups['trivial'])} (期望: 1)")
    
    test_perm = (1, 2, 0, 4, 3)
    cycles = pg.get_cycle_decomposition(test_perm)
    print(f"   置换 {test_perm} 的循环分解: {cycles}")
    print(f"   循环类型: {pg.get_cycle_type(test_perm)}")
    print(f"   是否偶置换: {pg.is_even(test_perm)}")
    
    print("\n2. 置换奇偶性加权验证")
    memberships = {
        'cat1': np.array([0.9, 0.1, 0.3, 0.7, 0.5]),
        'cat2': np.array([0.2, 0.8, 0.6, 0.4, 0.3])
    }
    pw = PermutationParityWeighting(5, memberships)
    stats = pw.get_parity_statistics()
    print(f"   总置换数: {stats['total_permutations']}")
    print(f"   偶置换数: {stats['even_count']} (期望: 60)")
    print(f"   奇置换数: {stats['odd_count']} (期望: 60)")
    
    perm = (0, 2, 1, 3, 4)
    weight_cat1 = pw.calculate_parity_weight(perm, 'cat1')
    weight_cat2 = pw.calculate_parity_weight(perm, 'cat2')
    print(f"   置换 {perm} 在cat1的权重: {weight_cat1:.4f}")
    print(f"   置换 {perm} 在cat2的权重: {weight_cat2:.4f}")
    
    print("\n3. 伯恩赛德引理验证")
    bc = BurnsideLemmaCalculator(5)
    
    orbits = bc.calculate_orbits()
    print(f"   轨道数(默认元素): {orbits}")
    
    elements = {0, 1, 2, 3, 4}
    orbits_detailed = bc.calculate_orbits_detailed(elements=elements)
    print(f"   详细轨道数: {orbits_detailed['orbit_count']}")
    print(f"   群阶: {orbits_detailed['group_order']}")
    
    effective = bc.calculate_effective_permutations(threshold=0.5)
    print(f"   有效性>=0.5的置换数: {len(effective)}")
    
    colorings = bc.calculate_colorings(n_colors=2)
    print(f"   2色着色不等价方案数: {colorings}")
    
    print("\n4. 共轭类划分验证")
    cp = ConjugacyClassPartitioner(5)
    classes = cp.get_conjugacy_classes()
    sizes = cp.get_class_sizes()
    print(f"   共轭类数量: {len(classes)}")
    print(f"   各类大小: {sizes}")
    
    layer_info = cp.get_layer_info()
    print(f"   最大类大小: {layer_info['largest_class']}")
    print(f"   最小类大小: {layer_info['smallest_class']}")
    
    items = ['A', 'B', 'C', 'D', 'E']
    shuffled = cp.stratified_shuffle(items)
    print(f"   分层随机排序结果: {shuffled}")
    
    print("\n5. 整合模型验证")
    sorter = PermutationGroupRandomSorter(5, memberships)
    
    analysis = sorter.decompose_and_analyze()
    print(f"   子群数: {analysis['subgroup_count']}")
    
    items = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    
    sorted_parity = sorter.advanced_sort(items, category='cat1', mode='parity')
    print(f"   奇偶性加权排序: {sorted_parity}")
    
    sorted_burnside = sorter.advanced_sort(items, mode='burnside')
    print(f"   伯恩赛德筛选排序: {sorted_burnside}")
    
    sorted_conjugacy = sorter.advanced_sort(items, mode='conjugacy')
    print(f"   共轭类分层排序: {sorted_conjugacy}")
    
    sorted_hybrid = sorter.advanced_sort(items, category='cat1', mode='hybrid')
    print(f"   混合模式排序: {sorted_hybrid}")
    
    stats = sorter.get_statistics()
    print(f"   排序历史记录数: {stats['sorting_history_count']}")
    
    print("\n" + "=" * 60)
    print("所有验证完成!")
    print("=" * 60)


def validate_classification_random_coupling_models():
    print("=" * 60)
    print("分类与随机耦合模型验证")
    print("=" * 60)
    
    print("\n1. 隶属度-概率高阶映射模型验证")
    mpm = MembershipProbabilityMapping(n_categories=3, polynomial_degree=3)
    
    memberships = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    probabilities = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
    mpm.set_membership_probability_mapping('math', memberships, probabilities)
    
    test_membership = np.array([0.2, 0.4, 0.6, 0.8])
    predicted_prob = mpm.predict_probability('math', test_membership)
    print(f"   测试隶属度: {test_membership}")
    print(f"   预测概率: {predicted_prob}")
    
    classification_dist = np.array([0.3, 0.5, 0.2])
    random_dist = np.array([0.33, 0.33, 0.34])
    deviation = mpm.monitor_distribution_deviation(classification_dist, random_dist)
    print(f"   KL散度: {deviation['kl_divergence']:.6f}")
    print(f"   JS散度: {deviation['js_divergence']:.6f}")
    print(f"   总变差: {deviation['total_variation']:.6f}")
    
    target_dist = np.array([0.2, 0.3, 0.5])
    correction = mpm.iterative_correction('math', target_dist, max_iterations=50)
    print(f"   迭代修正收敛: {correction['converged']}")
    print(f"   最终KL散度: {correction['final_kl_divergence']:.6f}")
    
    stats = mpm.get_mapping_statistics()
    print(f"   已配置类别数: {len(stats['categories_configured'])}")
    
    print("\n2. 分类边界约束随机子空间模型验证")
    bcs = BoundaryConstrainedSubspace(n_features=4, n_categories=2)
    
    center1 = np.array([1.0, 1.0, 1.0, 1.0])
    center2 = np.array([-1.0, -1.0, -1.0, -1.0])
    bcs.set_category_boundary(0, center1, radius=2.0)
    bcs.set_category_boundary(1, center2, radius=1.5)
    
    features = np.random.randn(50, 4) + center1
    high_order_stats = bcs.compute_high_order_statistics(features)
    print(f"   均值: {high_order_stats['mean']}")
    print(f"   偏度范围: [{high_order_stats['skewness'].min():.3f}, {high_order_stats['skewness'].max():.3f}]")
    print(f"   峰度范围: [{high_order_stats['kurtosis'].min():.3f}, {high_order_stats['kurtosis'].max():.3f}]")
    
    samples = bcs.sample_from_feasible_region(0, n_samples=10, method='direct')
    print(f"   采样点数: {len(samples)}")
    print(f"   采样点距中心平均距离: {np.mean(np.linalg.norm(samples - center1, axis=1)):.4f}")
    
    target = np.array([0.5, 0.5, 0.5, 0.5])
    lagrange_result = bcs.lagrangian_optimization(0, target, max_iterations=50)
    print(f"   拉格朗日优化目标值: {lagrange_result['optimal_objective']:.6f}")
    print(f"   最优特征: {lagrange_result['optimal_features']}")
    
    subspace = bcs.generate_constrained_subspace(0, subspace_dim=2, n_samples=100)
    print(f"   子空间基维度: {subspace.shape}")
    
    print("\n3. 错分风险熵驱动随机迭代模型验证")
    n_samples = 20
    n_categories = 3
    mre = MisclassificationRiskEntropyModel(n_categories=n_categories, n_samples=n_samples)
    
    np.random.seed(42)
    confidence_matrix = np.random.dirichlet(np.ones(n_categories), size=n_samples)
    mre.set_classification_confidence(confidence_matrix)
    
    entropies = mre.compute_all_risk_entropies()
    print(f"   风险熵均值: {np.mean(entropies):.4f}")
    print(f"   风险熵标准差: {np.std(entropies):.4f}")
    
    nr_result = mre.newton_raphson_iteration(max_iterations=50, tolerance=1e-6)
    print(f"   牛顿-拉夫逊迭代次数: {nr_result['iterations']}")
    print(f"   最终目标值: {nr_result['final_objective']:.6f}")
    print(f"   收敛状态: {nr_result['converged']}")
    
    sampled_indices = mre.sample_with_entropy_penalty(n_draws=5)
    print(f"   熵惩罚抽样索引: {sampled_indices}")
    
    rewards = np.random.randn(5)
    mre.update_weights_with_feedback(sampled_indices, rewards)
    
    model_stats = mre.get_model_statistics()
    print(f"   权重熵: {model_stats['weight_entropy']:.4f}")
    print(f"   权重范围: [{model_stats['current_weights_stats']['min']:.6f}, {model_stats['current_weights_stats']['max']:.6f}]")
    
    print("\n" + "=" * 60)
    print("分类与随机耦合模型验证完成!")
    print("=" * 60)


import hashlib
import psutil
import os
from collections import OrderedDict
import threading
import time


@dataclass
class BlockComputationConfig:
    """分块计算配置"""
    max_block_size: int = 1024
    min_block_size: int = 32
    memory_threshold: float = 0.8
    cpu_utilization_target: float = 0.75
    cache_max_entries: int = 10000
    entropy_decay_factor: float = 0.95
    jacobian_h: float = 1e-7
    newton_max_iter: int = 100
    newton_tolerance: float = 1e-8


class BlockComputationUnit:
    """
    分块式数值计算单元
    将高阶张量分解、多阶概率分布计算拆解为本地可分块执行的子计算单元
    """
    
    def __init__(self, config: Optional[BlockComputationConfig] = None):
        self.config = config or BlockComputationConfig()
        self.block_registry: Dict[str, Dict] = {}
        self.computation_history: List[Dict] = []
        self._lock = threading.Lock()
        
    def _compute_entropy(self, data: np.ndarray) -> float:
        """计算数据块的熵值（复杂度度量）"""
        if data.size == 0:
            return 0.0
        
        flat_data = data.flatten().astype(np.float64)
        
        if np.all(flat_data == flat_data[0]):
            return 0.0
        
        abs_data = np.abs(flat_data)
        total = np.sum(abs_data)
        
        if total < 1e-15:
            return 0.0
        
        probs = abs_data / total
        probs = probs[probs > 1e-15]
        
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        
        return float(entropy)
    
    def _estimate_block_complexity(self, shape: Tuple[int, ...], operation_type: str) -> float:
        """估算计算块的计算复杂度"""
        n_elements = np.prod(shape) if shape else 1
        
        complexity_factors = {
            'tensor_decompose': 2.5,
            'probability_dist': 1.8,
            'matrix_multiply': 2.0,
            'eigen_decompose': 3.0,
            'svd': 3.5,
            'fft': 1.5,
            'convolution': 2.2,
            'integration': 2.0,
            'optimization': 3.0
        }
        
        factor = complexity_factors.get(operation_type, 2.0)
        
        complexity = n_elements ** (factor / 2.0)
        
        return float(complexity)
    
    def create_block(self, block_id: str, data: np.ndarray, 
                    operation_type: str = 'generic') -> Dict:
        """创建计算块"""
        with self._lock:
            entropy = self._compute_entropy(data)
            complexity = self._estimate_block_complexity(data.shape, operation_type)
            
            optimal_block_size = self._compute_optimal_block_size(data.shape)
            
            sub_blocks = self._partition_data(data, optimal_block_size)
            
            block_info = {
                'block_id': block_id,
                'data_shape': data.shape,
                'data_dtype': str(data.dtype),
                'entropy': entropy,
                'complexity': complexity,
                'operation_type': operation_type,
                'optimal_block_size': optimal_block_size,
                'sub_blocks_count': len(sub_blocks),
                'created_at': time.time(),
                'status': 'pending',
                'priority': self._compute_priority(entropy, complexity)
            }
            
            self.block_registry[block_id] = {
                'info': block_info,
                'data': data,
                'sub_blocks': sub_blocks
            }
            
            return block_info
    
    def _compute_optimal_block_size(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """基于本地硬件算力计算最优分块大小"""
        available_mem = psutil.virtual_memory().available
        cpu_count = psutil.cpu_count()
        
        target_mem_per_block = available_mem * 0.1 / cpu_count
        
        element_size = 8
        max_elements = int(target_mem_per_block / element_size)
        
        block_dims = []
        for dim in shape:
            optimal_dim = min(dim, max(1, int(max_elements ** (1.0 / len(shape)))))
            optimal_dim = max(self.config.min_block_size, 
                            min(optimal_dim, self.config.max_block_size))
            block_dims.append(optimal_dim)
        
        return tuple(block_dims)
    
    def _partition_data(self, data: np.ndarray, block_size: Tuple[int, ...]) -> List[Dict]:
        """将数据分区为子块"""
        sub_blocks = []
        
        slices = []
        for i, (dim, block_dim) in enumerate(zip(data.shape, block_size)):
            dim_slices = []
            for start in range(0, dim, block_dim):
                end = min(start + block_dim, dim)
                dim_slices.append((start, end))
            slices.append(dim_slices)
        
        from itertools import product
        for indices in product(*[range(len(s)) for s in slices]):
            slice_tuple = tuple(slice(slices[i][j][0], slices[i][j][1]) 
                              for i, j in enumerate(indices))
            
            sub_data = data[slice_tuple]
            
            sub_blocks.append({
                'slice': slice_tuple,
                'shape': sub_data.shape,
                'entropy': self._compute_entropy(sub_data),
                'data_ref': sub_data
            })
        
        return sub_blocks
    
    def _compute_priority(self, entropy: float, complexity: float) -> float:
        """基于熵值和复杂度计算优先级"""
        normalized_entropy = min(entropy / 20.0, 1.0)
        log_complexity = np.log1p(complexity)
        normalized_complexity = min(log_complexity / 20.0, 1.0)
        
        priority = 0.6 * normalized_entropy + 0.4 * normalized_complexity
        
        return float(priority)
    
    def execute_block(self, block_id: str, 
                     compute_func: Callable[[np.ndarray], np.ndarray]) -> Optional[np.ndarray]:
        """执行分块计算"""
        if block_id not in self.block_registry:
            return None
        
        block = self.block_registry[block_id]
        data = block['data']
        sub_blocks = block['sub_blocks']
        
        result_shape = self._estimate_result_shape(data.shape, compute_func)
        result = np.zeros(result_shape, dtype=np.float64)
        
        for i, sub_block in enumerate(sub_blocks):
            sub_data = sub_block['data_ref']
            
            try:
                sub_result = compute_func(sub_data)
                
                if sub_result.shape == result[sub_block['slice']].shape:
                    result[sub_block['slice']] = sub_result
                else:
                    result = self._merge_sub_result(result, sub_result, sub_block['slice'])
                    
            except Exception as e:
                warnings.warn(f"Sub-block {i} computation failed: {str(e)}")
                continue
        
        with self._lock:
            block['info']['status'] = 'completed'
            block['info']['completed_at'] = time.time()
            
            self.computation_history.append({
                'block_id': block_id,
                'execution_time': time.time() - block['info']['created_at'],
                'sub_blocks_executed': len(sub_blocks)
            })
        
        return result
    
    def _estimate_result_shape(self, input_shape: Tuple[int, ...], 
                               compute_func: Callable) -> Tuple[int, ...]:
        """估算结果形状"""
        try:
            test_input = np.zeros((min(8, s) for s in input_shape), dtype=np.float64)
            test_result = compute_func(test_input)
            return input_shape
        except:
            return input_shape
    
    def _merge_sub_result(self, result: np.ndarray, sub_result: np.ndarray, 
                         slice_tuple: Tuple[slice, ...]) -> np.ndarray:
        """合并子块结果"""
        try:
            target_shape = result[slice_tuple].shape
            if sub_result.shape == target_shape:
                result[slice_tuple] = sub_result
            elif sub_result.size == np.prod(target_shape):
                result[slice_tuple] = sub_result.reshape(target_shape)
        except:
            pass
        return result
    
    def get_block_priority_order(self) -> List[str]:
        """获取按优先级排序的块ID列表"""
        blocks_with_priority = [
            (block_id, block['info']['priority'])
            for block_id, block in self.block_registry.items()
            if block['info']['status'] == 'pending'
        ]
        
        blocks_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        return [block_id for block_id, _ in blocks_with_priority]
    
    def clear_completed_blocks(self):
        """清理已完成的块以释放内存"""
        with self._lock:
            completed_ids = [
                block_id for block_id, block in self.block_registry.items()
                if block['info']['status'] == 'completed'
            ]
            
            for block_id in completed_ids:
                del self.block_registry[block_id]


class HardwareAwareScheduler:
    """
    硬件感知调度器
    基于本地硬件算力（CPU/内存）按计算复杂度熵值动态分配子单元优先级
    """
    
    def __init__(self, config: Optional[BlockComputationConfig] = None):
        self.config = config or BlockComputationConfig()
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cpu_count = psutil.cpu_count(logical=False)
        self.total_memory = psutil.virtual_memory().total
        self.task_queue: List[Dict] = []
        self.execution_log: List[Dict] = []
        self._lock = threading.Lock()
        
    def get_hardware_status(self) -> Dict[str, float]:
        """获取当前硬件状态"""
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'cpu_count': self.cpu_count,
            'physical_cpu_count': self.physical_cpu_count,
            'cpu_utilization': cpu_percent / 100.0,
            'total_memory_gb': self.total_memory / (1024**3),
            'available_memory_gb': mem.available / (1024**3),
            'memory_utilization': mem.percent / 100.0,
            'available_for_computation': mem.available / self.total_memory
        }
    
    def estimate_resource_requirement(self, block_info: Dict) -> Dict[str, float]:
        """估算资源需求"""
        shape = block_info.get('data_shape', (1,))
        n_elements = np.prod(shape) if shape else 1
        
        memory_mb = n_elements * 8 * 3 / (1024**2)
        
        complexity = block_info.get('complexity', 1.0)
        estimated_time = complexity / (self.cpu_count * 1000)
        
        return {
            'memory_mb': memory_mb,
            'estimated_time_s': estimated_time,
            'cpu_threads_recommended': min(self.cpu_count, max(1, int(np.log2(n_elements + 1))))
        }
    
    def schedule_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """调度计算块"""
        hw_status = self.get_hardware_status()
        
        if hw_status['memory_utilization'] > self.config.memory_threshold:
            blocks = self._apply_memory_pressure_priority(blocks, hw_status)
        else:
            blocks = self._apply_standard_priority(blocks, hw_status)
        
        with self._lock:
            self.task_queue = blocks.copy()
        
        return blocks
    
    def _apply_memory_pressure_priority(self, blocks: List[Dict], 
                                        hw_status: Dict) -> List[Dict]:
        """内存压力下的优先级调整"""
        for block in blocks:
            entropy = block.get('entropy', 0)
            complexity = block.get('complexity', 1)
            
            memory_efficiency = 1.0 / (1.0 + np.log1p(complexity))
            
            adjusted_priority = (
                0.3 * block.get('priority', 0.5) +
                0.4 * memory_efficiency +
                0.3 * (1.0 - hw_status['memory_utilization'])
            )
            
            block['adjusted_priority'] = adjusted_priority
            block['scheduling_reason'] = 'memory_pressure'
        
        blocks.sort(key=lambda x: x['adjusted_priority'], reverse=True)
        
        return blocks
    
    def _apply_standard_priority(self, blocks: List[Dict], 
                                 hw_status: Dict) -> List[Dict]:
        """标准优先级调度"""
        for block in blocks:
            entropy = block.get('entropy', 0)
            complexity = block.get('complexity', 1)
            
            cpu_factor = hw_status['available_for_computation']
            
            adjusted_priority = (
                0.5 * block.get('priority', 0.5) +
                0.3 * min(entropy / 15.0, 1.0) +
                0.2 * cpu_factor
            )
            
            block['adjusted_priority'] = adjusted_priority
            block['scheduling_reason'] = 'standard'
        
        blocks.sort(key=lambda x: x['adjusted_priority'], reverse=True)
        
        return blocks
    
    def get_next_batch(self, batch_size: Optional[int] = None) -> List[Dict]:
        """获取下一批待执行任务"""
        if batch_size is None:
            hw_status = self.get_hardware_status()
            available_mem_ratio = hw_status['available_for_computation']
            batch_size = max(1, int(self.cpu_count * available_mem_ratio))
        
        with self._lock:
            batch = self.task_queue[:batch_size]
            self.task_queue = self.task_queue[batch_size:]
        
        return batch
    
    def log_execution(self, block_id: str, execution_time: float, 
                     memory_used: float, success: bool):
        """记录执行日志"""
        with self._lock:
            self.execution_log.append({
                'block_id': block_id,
                'execution_time': execution_time,
                'memory_used_mb': memory_used,
                'success': success,
                'timestamp': time.time()
            })


class NewtonRaphsonCache:
    """
    牛顿-拉夫逊迭代缓存
    结合本地哈希索引缓存中间计算结果，按数据块熵值动态淘汰
    """
    
    def __init__(self, config: Optional[BlockComputationConfig] = None):
        self.config = config or BlockComputationConfig()
        self.max_entries = self.config.cache_max_entries
        
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._entropy_index: Dict[str, float] = {}
        self._access_count: Dict[str, int] = {}
        self._lock = threading.Lock()
        
        self.hit_count = 0
        self.miss_count = 0
        
    def _compute_hash(self, data: np.ndarray, operation_key: str) -> str:
        """计算数据块的哈希索引"""
        data_bytes = data.tobytes()
        
        shape_str = str(data.shape)
        dtype_str = str(data.dtype)
        
        hash_input = f"{operation_key}:{shape_str}:{dtype_str}:{len(data_bytes)}"
        
        hash_obj = hashlib.sha256(hash_input.encode())
        hash_obj.update(data_bytes[:min(len(data_bytes), 1024)])
        
        return hash_obj.hexdigest()
    
    def _compute_data_entropy(self, data: np.ndarray) -> float:
        """计算数据熵值"""
        if data.size == 0:
            return 0.0
        
        flat = data.flatten().astype(np.float64)
        
        if np.all(flat == flat[0]):
            return 0.0
        
        abs_data = np.abs(flat)
        total = np.sum(abs_data)
        
        if total < 1e-15:
            return 0.0
        
        probs = abs_data / total
        probs = probs[probs > 1e-15]
        
        return float(-np.sum(probs * np.log2(probs + 1e-15)))
    
    def get(self, data: np.ndarray, operation_key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """从缓存获取结果"""
        cache_key = self._compute_hash(data, operation_key)
        
        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                self.hit_count += 1
                
                entry = self._cache[cache_key]
                return entry['result'].copy(), entry['metadata'].copy()
            
            self.miss_count += 1
            return None
    
    def put(self, data: np.ndarray, operation_key: str, 
            result: np.ndarray, metadata: Optional[Dict] = None):
        """将结果存入缓存"""
        cache_key = self._compute_hash(data, operation_key)
        entropy = self._compute_data_entropy(data)
        
        with self._lock:
            if len(self._cache) >= self.max_entries:
                self._evict_by_entropy()
            
            self._cache[cache_key] = {
                'result': result.copy(),
                'metadata': metadata or {},
                'entropy': entropy,
                'created_at': time.time(),
                'data_shape': data.shape,
                'result_shape': result.shape
            }
            
            self._entropy_index[cache_key] = entropy
            self._access_count[cache_key] = 1
    
    def _evict_by_entropy(self):
        """按熵值动态淘汰缓存项"""
        if not self._cache:
            return
        
        items_with_scores = []
        for key, entry in self._cache.items():
            entropy = entry['entropy']
            access_count = self._access_count.get(key, 1)
            age = time.time() - entry['created_at']
            
            score = (
                0.4 * (1.0 - min(entropy / 20.0, 1.0)) +
                0.3 * (1.0 / (1.0 + np.log1p(access_count))) +
                0.3 * min(age / 3600.0, 1.0)
            )
            
            items_with_scores.append((key, score))
        
        items_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        evict_count = max(1, len(self._cache) // 10)
        
        for key, _ in items_with_scores[:evict_count]:
            del self._cache[key]
            if key in self._entropy_index:
                del self._entropy_index[key]
            if key in self._access_count:
                del self._access_count[key]
    
    def get_statistics(self) -> Dict:
        """获取缓存统计信息"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        with self._lock:
            total_memory = sum(
                entry['result'].nbytes 
                for entry in self._cache.values() 
                if isinstance(entry.get('result'), np.ndarray)
            )
            
            return {
                'entries_count': len(self._cache),
                'max_entries': self.max_entries,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_memory_mb': total_memory / (1024**2),
                'avg_entropy': np.mean(list(self._entropy_index.values())) if self._entropy_index else 0.0
            }
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._entropy_index.clear()
            self._access_count.clear()
            self.hit_count = 0
            self.miss_count = 0


class LocalDeploymentOptimizer:
    """
    本地部署优化器
    整合分块计算、硬件调度和牛顿-拉夫逊迭代求解
    """
    
    def __init__(self, config: Optional[BlockComputationConfig] = None):
        self.config = config or BlockComputationConfig()
        
        self.block_unit = BlockComputationUnit(self.config)
        self.scheduler = HardwareAwareScheduler(self.config)
        self.cache = NewtonRaphsonCache(self.config)
        
        self.optimization_history: List[Dict] = []
        self._lock = threading.Lock()
        
    def newton_raphson_solve(self, 
                            objective_func: Callable[[np.ndarray], float],
                            gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                            initial_params: Optional[np.ndarray] = None,
                            param_shape: Tuple[int, ...] = (10,)) -> Tuple[np.ndarray, Dict]:
        """
        牛顿-拉夫逊法迭代求解高阶模型参数
        """
        if initial_params is None:
            params = np.random.randn(*param_shape) * 0.1
        else:
            params = initial_params.copy().astype(np.float64)
        
        cache_key = f"nr_solve_{objective_func.__name__}_{param_shape}"
        
        cached = self.cache.get(params, cache_key)
        if cached is not None:
            return cached[0], {**cached[1], 'cache_hit': True}
        
        convergence_history = []
        
        for iteration in range(self.config.newton_max_iter):
            f_val = objective_func(params)
            
            if gradient_func is not None:
                grad = gradient_func(params)
            else:
                grad = self._numerical_gradient(objective_func, params)
            
            hessian = self._numerical_hessian(objective_func, params)
            
            try:
                hessian_reg = hessian + np.eye(len(params)) * 1e-6
                delta = np.linalg.solve(hessian_reg, grad)
            except np.linalg.LinAlgError:
                delta = grad * self.config.jacobian_h
            
            step_size = self._line_search(objective_func, params, -delta)
            params_new = params - step_size * delta
            
            param_change = np.linalg.norm(params_new - params)
            grad_norm = np.linalg.norm(grad)
            
            convergence_history.append({
                'iteration': iteration,
                'objective': f_val,
                'gradient_norm': grad_norm,
                'param_change': param_change,
                'step_size': step_size
            })
            
            if param_change < self.config.newton_tolerance:
                break
            
            params = params_new
            
            if iteration % 10 == 0:
                self.cache.put(params, cache_key, params, {
                    'iteration': iteration,
                    'objective': f_val
                })
        
        result_metadata = {
            'iterations': len(convergence_history),
            'final_objective': convergence_history[-1]['objective'] if convergence_history else None,
            'convergence_history': convergence_history,
            'cache_hit': False
        }
        
        self.cache.put(params, cache_key, params, result_metadata)
        
        return params, result_metadata
    
    def _numerical_gradient(self, func: Callable, params: np.ndarray) -> np.ndarray:
        """数值计算梯度"""
        h = self.config.jacobian_h
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += h
            params_minus[i] -= h
            
            grad[i] = (func(params_plus) - func(params_minus)) / (2 * h)
        
        return grad
    
    def _numerical_hessian(self, func: Callable, params: np.ndarray) -> np.ndarray:
        """数值计算Hessian矩阵"""
        h = self.config.jacobian_h
        n = len(params)
        hessian = np.zeros((n, n))
        
        f0 = func(params)
        
        for i in range(n):
            for j in range(i, n):
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()
                
                params_pp[i] += h
                params_pp[j] += h
                
                params_pm[i] += h
                params_pm[j] -= h
                
                params_mp[i] -= h
                params_mp[j] += h
                
                params_mm[i] -= h
                params_mm[j] -= h
                
                hessian[i, j] = (func(params_pp) - func(params_pm) - 
                                func(params_mp) + func(params_mm)) / (4 * h * h)
                hessian[j, i] = hessian[i, j]
        
        return hessian
    
    def _line_search(self, func: Callable, params: np.ndarray, 
                    direction: np.ndarray) -> float:
        """线搜索确定步长"""
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        f0 = func(params)
        
        for _ in range(20):
            new_params = params + alpha * direction
            if func(new_params) <= f0 + c * alpha * np.dot(direction, direction):
                break
            alpha *= rho
        
        return alpha
    
    def tensor_decomposition_blocked(self, tensor: np.ndarray, 
                                    rank: int = 2) -> Tuple[List[np.ndarray], Dict]:
        """分块式张量分解"""
        cache_key = f"tensor_decomp_{tensor.shape}_{rank}"
        
        cached = self.cache.get(tensor, cache_key)
        if cached is not None:
            return [cached[0]], {**cached[1], 'cache_hit': True}
        
        block_info = self.block_unit.create_block(
            f"tensor_{id(tensor)}", tensor, 'tensor_decompose'
        )
        
        def decomp_block(block_data: np.ndarray) -> np.ndarray:
            if block_data.ndim == 2:
                u, s, vt = np.linalg.svd(block_data, full_matrices=False)
                k = min(rank, len(s))
                return u[:, :k] @ np.diag(s[:k])
            elif block_data.ndim == 3:
                unfolded = block_data.reshape(block_data.shape[0], -1)
                u, s, vt = np.linalg.svd(unfolded, full_matrices=False)
                k = min(rank, len(s))
                return u[:, :k]
            else:
                return block_data
        
        result = self.block_unit.execute_block(f"tensor_{id(tensor)}", decomp_block)
        
        metadata = {
            'block_info': block_info,
            'cache_hit': False
        }
        
        if result is not None:
            self.cache.put(tensor, cache_key, result, metadata)
        
        return [result] if result is not None else [], metadata
    
    def probability_distribution_blocked(self, samples: np.ndarray, 
                                        n_bins: int = 50,
                                        order: int = 2) -> Tuple[np.ndarray, Dict]:
        """分块式多阶概率分布计算"""
        cache_key = f"prob_dist_{samples.shape}_{n_bins}_{order}"
        
        cached = self.cache.get(samples, cache_key)
        if cached is not None:
            return cached[0], {**cached[1], 'cache_hit': True}
        
        block_info = self.block_unit.create_block(
            f"prob_{id(samples)}", samples, 'probability_dist'
        )
        
        def compute_dist_block(block_data: np.ndarray) -> np.ndarray:
            if block_data.ndim == 1:
                hist, edges = np.histogram(block_data, bins=n_bins, density=True)
                return hist
            else:
                results = []
                for i in range(block_data.shape[1]):
                    hist, _ = np.histogram(block_data[:, i], bins=n_bins, density=True)
                    results.append(hist)
                return np.column_stack(results)
        
        partial_result = self.block_unit.execute_block(
            f"prob_{id(samples)}", compute_dist_block
        )
        
        if partial_result is not None:
            if partial_result.ndim == 1:
                partial_result = partial_result / (np.sum(partial_result) + 1e-10)
            else:
                partial_result = partial_result / (np.sum(partial_result, axis=0, keepdims=True) + 1e-10)
            
            for _ in range(order - 1):
                if partial_result.ndim == 1:
                    partial_result = np.convolve(partial_result, partial_result, mode='same')
                    partial_result = partial_result / (np.sum(partial_result) + 1e-10)
        
        metadata = {
            'block_info': block_info,
            'n_bins': n_bins,
            'order': order,
            'cache_hit': False
        }
        
        if partial_result is not None:
            self.cache.put(samples, cache_key, partial_result, metadata)
        
        return partial_result, metadata
    
    def optimize_with_scheduling(self, 
                                objective_func: Callable[[np.ndarray], float],
                                initial_params: np.ndarray,
                                n_iterations: int = 10) -> Tuple[np.ndarray, Dict]:
        """带硬件感知调度的优化"""
        hw_status = self.scheduler.get_hardware_status()
        
        block_info = self.block_unit.create_block(
            f"opt_{id(initial_params)}", initial_params, 'optimization'
        )
        
        resource_req = self.scheduler.estimate_resource_requirement(block_info['info'])
        
        scheduled = self.scheduler.schedule_blocks([block_info['info']])
        
        best_params = initial_params.copy()
        best_value = objective_func(best_params)
        
        optimization_log = []
        
        for i in range(n_iterations):
            current_hw = self.scheduler.get_hardware_status()
            
            if current_hw['memory_utilization'] > self.config.memory_threshold:
                self.cache._evict_by_entropy()
            
            params, meta = self.newton_raphson_solve(
                objective_func,
                initial_params=best_params,
                param_shape=best_params.shape
            )
            
            current_value = objective_func(params)
            
            if current_value < best_value:
                best_params = params.copy()
                best_value = current_value
            
            optimization_log.append({
                'iteration': i,
                'value': current_value,
                'best_value': best_value,
                'hw_status': current_hw,
                'cache_stats': self.cache.get_statistics()
            })
            
            self.scheduler.log_execution(
                f"opt_iter_{i}",
                meta.get('iterations', 0) * 0.001,
                resource_req['memory_mb'],
                True
            )
        
        result_metadata = {
            'best_value': best_value,
            'n_iterations': n_iterations,
            'optimization_log': optimization_log,
            'final_hw_status': self.scheduler.get_hardware_status(),
            'final_cache_stats': self.cache.get_statistics()
        }
        
        with self._lock:
            self.optimization_history.append(result_metadata)
        
        return best_params, result_metadata
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'hardware': self.scheduler.get_hardware_status(),
            'cache': self.cache.get_statistics(),
            'block_unit': {
                'registered_blocks': len(self.block_unit.block_registry),
                'computation_history_count': len(self.block_unit.computation_history)
            },
            'scheduler': {
                'pending_tasks': len(self.scheduler.task_queue),
                'execution_log_count': len(self.scheduler.execution_log)
            }
        }
    
    def cleanup(self):
        """清理资源"""
        self.block_unit.clear_completed_blocks()
        self.cache._evict_by_entropy()
        
        if self.cache.get_statistics()['hit_rate'] < 0.1:
            self.cache.clear()


def validate_local_deployment_optimizer():
    print("=" * 60)
    print("本地部署适配优化验证")
    print("=" * 60)
    
    config = BlockComputationConfig(
        max_block_size=512,
        cache_max_entries=1000,
        newton_max_iter=50
    )
    
    optimizer = LocalDeploymentOptimizer(config)
    
    print("\n1. 系统状态检查")
    status = optimizer.get_system_status()
    print(f"   CPU核心数: {status['hardware']['cpu_count']}")
    print(f"   物理核心数: {status['hardware']['physical_cpu_count']}")
    print(f"   总内存: {status['hardware']['total_memory_gb']:.2f} GB")
    print(f"   可用内存: {status['hardware']['available_memory_gb']:.2f} GB")
    print(f"   内存使用率: {status['hardware']['memory_utilization']:.2%}")
    
    print("\n2. 分块式张量分解测试")
    test_tensor = np.random.randn(100, 80)
    factors, meta = optimizer.tensor_decomposition_blocked(test_tensor, rank=5)
    print(f"   输入张量形状: {test_tensor.shape}")
    print(f"   分解因子数: {len(factors)}")
    if factors:
        print(f"   分解结果形状: {factors[0].shape}")
    print(f"   缓存命中: {meta.get('cache_hit', False)}")
    
    print("\n3. 分块式概率分布计算测试")
    test_samples = np.random.randn(500, 3)
    dist, prob_meta = optimizer.probability_distribution_blocked(test_samples, n_bins=30, order=2)
    print(f"   样本形状: {test_samples.shape}")
    print(f"   分布形状: {dist.shape if dist is not None else 'None'}")
    print(f"   缓存命中: {prob_meta.get('cache_hit', False)}")
    
    print("\n4. 牛顿-拉夫逊迭代求解测试")
    
    def quadratic_objective(x):
        return np.sum((x - 2.0) ** 2)
    
    initial = np.zeros(10)
    result, nr_meta = optimizer.newton_raphson_solve(
        quadratic_objective,
        initial_params=initial,
        param_shape=(10,)
    )
    print(f"   初始目标值: {quadratic_objective(initial):.6f}")
    print(f"   最终目标值: {nr_meta['final_objective']:.6f}")
    print(f"   迭代次数: {nr_meta['iterations']}")
    print(f"   最优参数均值: {np.mean(result):.6f} (期望接近2.0)")
    
    print("\n5. 缓存统计")
    cache_stats = optimizer.cache.get_statistics()
    print(f"   缓存条目数: {cache_stats['entries_count']}")
    print(f"   命中次数: {cache_stats['hit_count']}")
    print(f"   未命中次数: {cache_stats['miss_count']}")
    print(f"   命中率: {cache_stats['hit_rate']:.2%}")
    print(f"   缓存内存: {cache_stats['total_memory_mb']:.4f} MB")
    
    print("\n6. 硬件感知调度测试")
    blocks = [
        {'entropy': 5.0, 'complexity': 1000, 'priority': 0.5},
        {'entropy': 10.0, 'complexity': 500, 'priority': 0.7},
        {'entropy': 3.0, 'complexity': 2000, 'priority': 0.3}
    ]
    scheduled = optimizer.scheduler.schedule_blocks(blocks)
    print(f"   调度顺序 (按调整后优先级):")
    for i, block in enumerate(scheduled):
        print(f"     {i+1}. 熵值={block['entropy']:.1f}, "
              f"复杂度={block['complexity']:.0f}, "
              f"调整后优先级={block['adjusted_priority']:.4f}")
    
    print("\n7. 带调度的优化测试")
    
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    init_params = np.zeros(5)
    opt_result, opt_meta = optimizer.optimize_with_scheduling(
        rosenbrock, init_params, n_iterations=3
    )
    print(f"   初始目标值: {rosenbrock(init_params):.6f}")
    print(f"   最终目标值: {opt_meta['best_value']:.6f}")
    print(f"   优化迭代次数: {opt_meta['n_iterations']}")
    
    print("\n8. 最终系统状态")
    final_status = optimizer.get_system_status()
    print(f"   待处理任务: {final_status['scheduler']['pending_tasks']}")
    print(f"   执行日志数: {final_status['scheduler']['execution_log_count']}")
    print(f"   最终缓存命中率: {final_status['cache']['hit_rate']:.2%}")
    
    optimizer.cleanup()
    print("\n   资源清理完成")
    
    print("\n" + "=" * 60)
    print("本地部署适配优化验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    validate_probability_models()
    print("\n")
    validate_number_theory_random_encoder()
    print("\n")
    validate_permutation_group_models()
    print("\n")
    validate_monte_carlo_sampling()
    print("\n")
    validate_stochastic_processes()
    print("\n")
    validate_local_deployment_optimizer()
    print("\n")
    validate_classification_random_coupling_models()
    print("\n")
    validate_result_analysis_models()
