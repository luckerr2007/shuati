"""
组合数学决策模型
实现排列组合决策树、容斥原理边界修正、鸽巢原理分桶、组合计数验证
"""

import itertools
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional
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
                           used_features: Set[str] = None) -> Dict:
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
    
    def add_set(self, name: str, elements: Set[Any]):
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
    
    def correct_classification_boundary(self, category_sets: Dict[str, Set[Any]], 
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
    
    def get_exclusive_elements(self, category: str, all_categories: List[str]) -> Set[Any]:
        if category not in self.sets:
            return set()
        
        exclusive = self.sets[category].copy()
        for other in all_categories:
            if other != category and other in self.sets:
                exclusive -= self.sets[other]
        
        return exclusive
    
    def calculate_membership_probability(self, element: Any, 
                                         category_sets: Dict[str, Set[Any]]) -> Dict[str, float]:
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
                                    population_categories: Dict[str, Set[Any]],
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
    
    def correct_boundaries(self, category_sets: Dict[str, Set[Any]]) -> Dict[str, Tuple[float, float]]:
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


if __name__ == "__main__":
    validate_combinatorial_operations()
