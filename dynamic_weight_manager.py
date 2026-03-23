import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

CATEGORY_GROUPS = {
    '文学类': ['人文常识', '历史常识', '地理常识', '安徽文化常识', '传统文化常识', '文学常识'],
    '专业类': ['计算机基础', '网络基础', '数据库', '编程语言', '软件工程', '信息安全', '操作系统', '数据结构', '计算机组成原理', '专业课题目', '计算机网络', '云计算', '人工智能', '大数据', '办公软件', '程序设计基础', '计算机基础-硬件', '计算机基础-软件', '计算机基础-安全', '科学基础-数制', '科学基础-编码', '科学基础-计算机发展'],
    '政治类': ['政治常识', '法律常识', '经济常识', '时事政治', '政治素质'],
    '科技类': ['科技常识', '生活常识', '自然科学', '科学基础-物理', '科学基础-化学', '科学基础-生物', '科学基础-地理'],
    '其他': ['未知', '职业道德', '心理健康', '思想道德素质']
}

GROUP_WEIGHTS = {
    '文学类': 1.0,
    '专业类': 1.2,
    '政治类': 1.0,
    '科技类': 0.8,
    '其他': 0.5
}

FEATURE_DIMENSIONS = 12
FEATURE_NAMES = ['accuracy', 'weight', 'wrong_rate', 'difficulty_factor', 'time_factor', 'base_score', 'momentum', 'entropy_score', 'fuzzy_score', 'probability_score', 'dynamic_score', 'sensitivity_score']

def get_category_group(category: str) -> str:
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return group
    return '其他'

class DynamicWeightManager:
    def __init__(self, stats_file):
        self.stats_file = stats_file
        self.category_stats = self.load_stats()
        
        self.professional_categories = [
            "计算机基础-硬件", "计算机基础-软件", "计算机基础-安全",
            "计算机网络", "信息安全", "云计算", "人工智能", "大数据",
            "操作系统", "办公软件", "程序设计基础",
            "科学基础-数制", "科学基础-编码", "科学基础-计算机发展",
            "编程语言", "数据库", "网络基础"
        ]
        
        self.general_categories = [
            "职业道德", "心理健康", "思想道德素质", "政治素质",
            "法律常识", "人文常识", "历史常识", "地理常识", "安徽文化常识",
            "科学基础-物理", "科学基础-化学", "科学基础-生物", "科学基础-地理",
            "政治常识", "科技常识"
        ]
    
    def load_stats(self):
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_stats(self):
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.category_stats, f, ensure_ascii=False, indent=2)
    
    def _ensure_category_structure(self, category: str) -> Dict:
        if category not in self.category_stats:
            self.category_stats[category] = {
                "total": 0, "correct": 0, "wrong": 0, 
                "accuracy": 1.0, "weight": 1.0,
                "group": get_category_group(category),
                "fuzzy_membership": {
                    "triangular": {"l": 0.3, "m": 0.5, "u": 0.7},
                    "trapezoidal": {"a": 0.25, "b": 0.45, "c": 0.55, "d": 0.75},
                    "fusion_weight": 0.5
                },
                "probability_params": {
                    "multinomial_weight": 0.4,
                    "negative_binomial_weight": 0.3,
                    "hypergeometric_weight": 0.3,
                    "kl_divergence": 0.0,
                    "entropy": 0.0
                },
                "feature_vector": [0.5] * FEATURE_DIMENSIONS,
                "higher_order_moments": {
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                    "cross_moment": 0.0
                },
                "entropy_metrics": {
                    "shannon_entropy": 0.0,
                    "fuzzy_entropy": 0.0,
                    "classification_entropy": 0.0
                },
                "dynamic_state": {
                    "feature_velocity": [0.0] * FEATURE_DIMENSIONS,
                    "feature_acceleration": [0.0] * FEATURE_DIMENSIONS,
                    "boundary_threshold": 0.5
                },
                "sensitivity_indices": {
                    "sobol_first_order": {},
                    "sobol_total_order": {}
                },
                "last_updated": datetime.now().isoformat()
            }
        return self.category_stats[category]
    
    def record_answer(self, category, is_correct):
        stats = self._ensure_category_structure(category)
        
        stats["total"] += 1
        
        if is_correct:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
        
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        
        self.update_weight(category)
        self.update_fuzzy_membership(category)
        self.update_probability_params(category)
        self.update_feature_vector(category)
        self.update_entropy_metrics(category)
        self.update_higher_order_moments(category)
        stats["last_updated"] = datetime.now().isoformat()
        
        self.save_stats()
    
    def update_weight(self, category):
        stats = self.category_stats.get(category, {})
        accuracy = stats.get("accuracy", 1.0)
        
        if accuracy < 0.5:
            weight = 1.5
        elif accuracy < 0.7:
            weight = 1.2
        elif accuracy > 0.9:
            weight = 0.8
        elif accuracy > 0.8:
            weight = 0.9
        else:
            weight = 1.0
        
        stats["weight"] = weight
    
    def update_fuzzy_membership(self, category):
        stats = self.category_stats.get(category, {})
        accuracy = stats.get("accuracy", 0.5)
        
        l = max(0.0, accuracy - 0.15)
        m = accuracy
        u = min(1.0, accuracy + 0.15)
        
        a = max(0.0, accuracy - 0.20)
        b = max(0.0, accuracy - 0.08)
        c = min(1.0, accuracy + 0.08)
        d = min(1.0, accuracy + 0.20)
        
        if "fuzzy_membership" not in stats:
            stats["fuzzy_membership"] = {}
        
        stats["fuzzy_membership"]["triangular"] = {"l": l, "m": m, "u": u}
        stats["fuzzy_membership"]["trapezoidal"] = {"a": a, "b": b, "c": c, "d": d}
    
    def update_probability_params(self, category):
        stats = self.category_stats.get(category, {})
        accuracy = stats.get("accuracy", 0.5)
        wrong_rate = 1.0 - accuracy
        
        if accuracy > 0.8:
            stats["probability_params"]["multinomial_weight"] = 0.5
            stats["probability_params"]["negative_binomial_weight"] = 0.25
            stats["probability_params"]["hypergeometric_weight"] = 0.25
        elif accuracy < 0.5:
            stats["probability_params"]["multinomial_weight"] = 0.3
            stats["probability_params"]["negative_binomial_weight"] = 0.4
            stats["probability_params"]["hypergeometric_weight"] = 0.3
        else:
            stats["probability_params"]["multinomial_weight"] = 0.4
            stats["probability_params"]["negative_binomial_weight"] = 0.3
            stats["probability_params"]["hypergeometric_weight"] = 0.3
        
        if wrong_rate > 0 and wrong_rate < 1:
            entropy = -wrong_rate * np.log(wrong_rate + 1e-10) - accuracy * np.log(accuracy + 1e-10)
        else:
            entropy = 0.0
        stats["probability_params"]["entropy"] = entropy
    
    def update_feature_vector(self, category):
        stats = self.category_stats.get(category, {})
        accuracy = stats.get("accuracy", 0.5)
        weight = stats.get("weight", 1.0)
        wrong_rate = 1.0 - accuracy
        
        group = get_category_group(category)
        group_weight = GROUP_WEIGHTS.get(group, 1.0)
        
        feature_vector = [
            accuracy,
            weight,
            wrong_rate,
            group_weight,
            0.0,
            0.5,
            0.0,
            stats.get("entropy_metrics", {}).get("shannon_entropy", 0.0),
            stats.get("fuzzy_membership", {}).get("triangular", {}).get("m", 0.5),
            stats.get("probability_params", {}).get("entropy", 0.0),
            0.0,
            0.0
        ]
        
        stats["feature_vector"] = feature_vector
    
    def update_entropy_metrics(self, category):
        stats = self.category_stats.get(category, {})
        accuracy = stats.get("accuracy", 0.5)
        wrong_rate = 1.0 - accuracy
        
        if wrong_rate > 0 and wrong_rate < 1:
            shannon_entropy = -wrong_rate * np.log(wrong_rate + 1e-10) - accuracy * np.log(accuracy + 1e-10)
        else:
            shannon_entropy = 0.0
        
        fuzzy_m = stats.get("fuzzy_membership", {}).get("triangular", {}).get("m", 0.5)
        fuzzy_entropy = min(fuzzy_m, 1 - fuzzy_m)
        
        classification_entropy = shannon_entropy * 0.5 + fuzzy_entropy * 0.5
        
        if "entropy_metrics" not in stats:
            stats["entropy_metrics"] = {}
        
        stats["entropy_metrics"]["shannon_entropy"] = shannon_entropy
        stats["entropy_metrics"]["fuzzy_entropy"] = fuzzy_entropy
        stats["entropy_metrics"]["classification_entropy"] = classification_entropy
    
    def update_higher_order_moments(self, category):
        stats = self.category_stats.get(category, {})
        accuracy = stats.get("accuracy", 0.5)
        
        skewness = (accuracy - 0.5) * 2
        kurtosis = -1.5 + abs(accuracy - 0.5) * 3
        
        if "higher_order_moments" not in stats:
            stats["higher_order_moments"] = {}
        
        stats["higher_order_moments"]["skewness"] = skewness
        stats["higher_order_moments"]["kurtosis"] = kurtosis
        stats["higher_order_moments"]["cross_moment"] = 0.0
    
    def get_category_weights(self):
        weights = {}
        for cat in self.professional_categories + self.general_categories:
            stats = self.category_stats.get(cat, {"weight": 1.0})
            weights[cat] = stats.get("weight", 1.0)
        return weights
    
    def get_feature_vectors(self) -> Dict[str, List[float]]:
        vectors = {}
        for cat, stats in self.category_stats.items():
            if cat.startswith("_"):
                continue
            vectors[cat] = stats.get("feature_vector", [0.5] * FEATURE_DIMENSIONS)
        return vectors
    
    def get_fuzzy_memberships(self) -> Dict[str, Dict]:
        memberships = {}
        for cat, stats in self.category_stats.items():
            if cat.startswith("_"):
                continue
            memberships[cat] = stats.get("fuzzy_membership", {})
        return memberships
    
    def get_probability_params(self) -> Dict[str, Dict]:
        params = {}
        for cat, stats in self.category_stats.items():
            if cat.startswith("_"):
                continue
            params[cat] = stats.get("probability_params", {})
        return params
    
    def get_entropy_metrics(self) -> Dict[str, Dict]:
        metrics = {}
        for cat, stats in self.category_stats.items():
            if cat.startswith("_"):
                continue
            metrics[cat] = stats.get("entropy_metrics", {})
        return metrics
    
    def get_higher_order_moments(self) -> Dict[str, Dict]:
        moments = {}
        for cat, stats in self.category_stats.items():
            if cat.startswith("_"):
                continue
            moments[cat] = stats.get("higher_order_moments", {})
        return moments
    
    def get_dynamic_states(self) -> Dict[str, Dict]:
        states = {}
        for cat, stats in self.category_stats.items():
            if cat.startswith("_"):
                continue
            states[cat] = stats.get("dynamic_state", {})
        return states
    
    def get_difficulty_for_category(self, category):
        stats = self.category_stats.get(category, {"accuracy": 1.0})
        accuracy = stats.get("accuracy", 1.0)
        
        if accuracy < 0.5:
            return "简单"
        elif accuracy > 0.8:
            return "较难"
        else:
            return "中等"
    
    def calculate_distribution(self, total_count=100):
        professional_ratio = 0.20
        general_ratio = 0.80
        
        professional_count = int(total_count * professional_ratio)
        general_count = total_count - professional_count
        
        distribution = {}
        
        prof_weights = {}
        for cat in self.professional_categories:
            stats = self.category_stats.get(cat, {"weight": 1.0})
            prof_weights[cat] = stats.get("weight", 1.0)
        
        total_prof_weight = sum(prof_weights.values())
        for cat in self.professional_categories:
            distribution[cat] = (prof_weights[cat] / total_prof_weight) * professional_count
        
        gen_weights = {}
        for cat in self.general_categories:
            stats = self.category_stats.get(cat, {"weight": 1.0})
            gen_weights[cat] = stats.get("weight", 1.0)
        
        total_gen_weight = sum(gen_weights.values())
        for cat in self.general_categories:
            distribution[cat] = (gen_weights[cat] / total_gen_weight) * general_count
        
        return distribution
    
    def get_stats_summary(self):
        summary = {
            "professional": {"total": 0, "correct": 0, "wrong": 0},
            "general": {"total": 0, "correct": 0, "wrong": 0},
            "by_category": {},
            "feature_vectors": {},
            "entropy_metrics": {},
            "group_weights": GROUP_WEIGHTS.copy()
        }
        
        for cat in self.professional_categories:
            stats = self.category_stats.get(cat, {})
            summary["professional"]["total"] += stats.get("total", 0)
            summary["professional"]["correct"] += stats.get("correct", 0)
            summary["professional"]["wrong"] += stats.get("wrong", 0)
            if stats:
                summary["by_category"][cat] = stats
                if "feature_vector" in stats:
                    summary["feature_vectors"][cat] = stats["feature_vector"]
                if "entropy_metrics" in stats:
                    summary["entropy_metrics"][cat] = stats["entropy_metrics"]
        
        for cat in self.general_categories:
            stats = self.category_stats.get(cat, {})
            summary["general"]["total"] += stats.get("total", 0)
            summary["general"]["correct"] += stats.get("correct", 0)
            summary["general"]["wrong"] += stats.get("wrong", 0)
            if stats:
                summary["by_category"][cat] = stats
                if "feature_vector" in stats:
                    summary["feature_vectors"][cat] = stats["feature_vector"]
                if "entropy_metrics" in stats:
                    summary["entropy_metrics"][cat] = stats["entropy_metrics"]
        
        return summary
    
    def update_sensitivity_indices(self, category: str, sobol_first: Dict[str, float], sobol_total: Dict[str, float]):
        if category in self.category_stats:
            self.category_stats[category]["sensitivity_indices"] = {
                "sobol_first_order": sobol_first,
                "sobol_total_order": sobol_total
            }
            self.save_stats()
    
    def update_dynamic_state(self, category: str, velocity: List[float], acceleration: List[float], threshold: float):
        if category in self.category_stats:
            self.category_stats[category]["dynamic_state"] = {
                "feature_velocity": velocity,
                "feature_acceleration": acceleration,
                "boundary_threshold": threshold
            }
            self.save_stats()
