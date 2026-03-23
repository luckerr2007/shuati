import json
import random
import math
import hashlib
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple, Optional, Callable
import os

from advanced_math_models import (
    NumberTheoryRandomEncoder,
    MonteCarloStratifiedSamplingFramework,
    PermutationGroupRandomSorter,
    MembershipProbabilityMapping,
    LocalDeploymentOptimizer,
)

CATEGORY_GROUPS = {
    '文学类': ['人文常识', '历史常识', '地理常识', '安徽文化常识', '传统文化常识', '文学常识'],
    '专业类': ['计算机基础', '网络基础', '数据库', '编程语言', '软件工程', '信息安全', '操作系统', '数据结构', '计算机组成原理', '专业课题目'],
    '政治类': ['政治常识', '法律常识', '经济常识', '时事政治'],
    '科技类': ['科技常识', '生活常识', '自然科学'],
    '其他': ['未知']
}

GROUP_WEIGHTS_DEFAULT = {
    '文学类': 1.0,
    '专业类': 1.2,
    '政治类': 1.0,
    '科技类': 0.8,
    '其他': 0.5
}

DIFFICULTY_DISTRIBUTION = {
    '简单': 0.25,
    '中等': 0.50,
    '较难': 0.25
}

DEFAULT_QUESTION_COUNT = 100

def get_category_group(category: str) -> str:
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return group
    return '其他'

class RandomGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(time.time() * 1000) % (2**32)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
    
    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
    
    def fisher_yates_shuffle(self, items: List) -> List:
        result = items.copy()
        n = len(result)
        for i in range(n - 1, 0, -1):
            j = self.rng.randint(0, i)
            result[i], result[j] = result[j], result[i]
        return result
    
    def weighted_random_choice(self, items: List[Tuple[Any, float]]) -> Any:
        if not items:
            return None
        
        total_weight = sum(w for _, w in items)
        if total_weight <= 0:
            return self.rng.choice([item for item, _ in items])
        
        r = self.rng.uniform(0, total_weight)
        cumulative = 0
        for item, weight in items:
            cumulative += weight
            if r <= cumulative:
                return item
        
        return items[-1][0]
    
    def weighted_random_sample(self, items: List[Tuple[Any, float]], k: int) -> List[Any]:
        if len(items) <= k:
            return [item for item, _ in items]
        
        total_weight = sum(max(0.001, w) for _, w in items)
        probabilities = [max(0.001, w) / total_weight for _, w in items]
        
        indices = self.np_rng.choice(len(items), size=k, replace=False, p=probabilities)
        return [items[i][0] for i in indices]
    
    def gaussian_noise(self, mean: float = 0, std: float = 1) -> float:
        return self.np_rng.normal(mean, std)
    
    def exponential_random(self, scale: float = 1.0) -> float:
        return self.np_rng.exponential(scale)
    
    def stratified_sample(self, items: List[Any], strata_key: callable, k: int) -> List[Any]:
        if len(items) <= k:
            return items
        
        strata = defaultdict(list)
        for item in items:
            strata[strata_key(item)].append(item)
        
        result = []
        strata_count = len(strata)
        base_per_stratum = k // strata_count
        remainder = k % strata_count
        
        stratum_list = list(strata.items())
        self.rng.shuffle(stratum_list)
        
        for i, (key, stratum_items) in enumerate(stratum_list):
            take = base_per_stratum + (1 if i < remainder else 0)
            take = min(take, len(stratum_items))
            if take > 0:
                selected = self.rng.sample(stratum_items, take)
                result.extend(selected)
        
        return result

class BloomFilter:
    def __init__(self, size: int = 10000, hash_count: int = 7):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size
    
    def _hashes(self, item: str) -> List[int]:
        hashes = []
        item_bytes = item.encode('utf-8')
        for i in range(self.hash_count):
            h = hashlib.md5(item_bytes + str(i).encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes
    
    def add(self, item: str):
        for h in self._hashes(item):
            self.bit_array[h] = True
    
    def might_contain(self, item: str) -> bool:
        return all(self.bit_array[h] for h in self._hashes(item))
    
    def clear(self):
        self.bit_array = [False] * self.size

class QuestionDeduplicator:
    def __init__(self):
        self.selected_ids: Set[int] = set()
        self.selected_hashes: Set[str] = set()
        self.bloom_filter = BloomFilter()
        self.similarity_threshold = 0.7
    
    def _normalize_text(self, text: str) -> str:
        return ''.join(text.lower().split())
    
    def _get_text_hash(self, question: Dict) -> str:
        normalized = self._normalize_text(question.get('question', ''))
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = set(self._normalize_text(text1))
        words2 = set(self._normalize_text(text2))
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, question: Dict, selected_questions: List[Dict]) -> bool:
        qid = question.get('id')
        if qid in self.selected_ids:
            return True
        
        question_hash = self._get_text_hash(question)
        if question_hash in self.selected_hashes:
            return True
        
        if self.bloom_filter.might_contain(question_hash):
            for sel_q in selected_questions:
                similarity = self._calculate_similarity(
                    question.get('question', ''),
                    sel_q.get('question', '')
                )
                if similarity > self.similarity_threshold:
                    return True
        
        return False
    
    def add_question(self, question: Dict):
        qid = question.get('id')
        if qid is not None:
            self.selected_ids.add(qid)
        
        question_hash = self._get_text_hash(question)
        self.selected_hashes.add(question_hash)
        self.bloom_filter.add(question_hash)
    
    def clear(self):
        self.selected_ids.clear()
        self.selected_hashes.clear()
        self.bloom_filter.clear()
    
    def get_selected_count(self) -> int:
        return len(self.selected_ids)

class AntColonyOptimizer:
    def __init__(self, questions: List[Dict], n_ants: int = 10, n_iterations: int = 5,
                 alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.5):
        self.questions = questions
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromones = defaultdict(lambda: 1.0)
        self.rng = RandomGenerator()
    
    def _calculate_heuristic(self, question: Dict, category_weights: Dict[str, float],
                            group_weights: Dict[str, float]) -> float:
        category = question.get('category', '未知')
        cat_weight = category_weights.get(category, 1.0)
        group = get_category_group(category)
        group_weight = group_weights.get(group, 1.0)
        
        seen = question.get('seen', 0)
        wrong_count = question.get('wrong_count', 0)
        correct_count = question.get('correct_count', 0)
        
        heuristic = cat_weight * group_weight
        heuristic *= (1.0 / (1.0 + seen * 0.1))
        
        if wrong_count > correct_count:
            heuristic *= 1.5
        
        status = question.get('status', 'normal')
        if status == 'weak':
            heuristic *= 1.3
        elif status == 'mastered':
            heuristic *= 0.7
        
        return heuristic
    
    def _select_next_question(self, available: List[Dict], category_weights: Dict[str, float],
                             group_weights: Dict[str, float]) -> Optional[Dict]:
        if not available:
            return None
        
        probabilities = []
        total = 0.0
        
        for q in available:
            qid = q.get('id')
            pheromone = self.pheromones[qid] ** self.alpha
            heuristic = self._calculate_heuristic(q, category_weights, group_weights) ** self.beta
            prob = pheromone * heuristic
            probabilities.append((q, prob))
            total += prob
        
        if total <= 0:
            return self.rng.rng.choice(available)
        
        r = self.rng.rng.uniform(0, total)
        cumulative = 0
        for q, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return q
        
        return probabilities[-1][0]
    
    def optimize_selection(self, count: int, category_weights: Dict[str, float],
                          group_weights: Dict[str, float],
                          deduplicator: QuestionDeduplicator) -> List[Dict]:
        best_selection = []
        best_score = -float('inf')
        
        for iteration in range(self.n_iterations):
            all_selections = []
            
            for ant in range(self.n_ants):
                selection = []
                available = self.questions.copy()
                temp_deduplicator = QuestionDeduplicator()
                
                while len(selection) < count and available:
                    q = self._select_next_question(available, category_weights, group_weights)
                    if q and not temp_deduplicator.is_duplicate(q, selection):
                        selection.append(q)
                        temp_deduplicator.add_question(q)
                    available = [x for x in available if x != q]
                
                all_selections.append(selection)
                
                score = self._evaluate_selection(selection, category_weights, group_weights)
                if score > best_score:
                    best_score = score
                    best_selection = selection.copy()
            
            self._update_pheromones(all_selections, category_weights, group_weights)
        
        return best_selection
    
    def _evaluate_selection(self, selection: List[Dict], category_weights: Dict[str, float],
                           group_weights: Dict[str, float]) -> float:
        if not selection:
            return 0.0
        
        score = 0.0
        category_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        
        for q in selection:
            category = q.get('category', '未知')
            category_counts[category] += 1
            difficulty = q.get('difficulty', '中等')
            difficulty_counts[difficulty] += 1
            
            cat_weight = category_weights.get(category, 1.0)
            group = get_category_group(category)
            group_weight = group_weights.get(group, 1.0)
            score += cat_weight * group_weight
        
        total_cats = len(category_counts)
        if total_cats > 0:
            diversity_bonus = total_cats * 2
            score += diversity_bonus
        
        for diff, target_ratio in DIFFICULTY_DISTRIBUTION.items():
            actual_ratio = difficulty_counts.get(diff, 0) / len(selection)
            deviation = abs(actual_ratio - target_ratio)
            score -= deviation * 10
        
        return score
    
    def _update_pheromones(self, all_selections: List[List[Dict]], 
                          category_weights: Dict[str, float],
                          group_weights: Dict[str, float]):
        for qid in self.pheromones:
            self.pheromones[qid] *= (1 - self.evaporation)
        
        for selection in all_selections:
            score = self._evaluate_selection(selection, category_weights, group_weights)
            for q in selection:
                qid = q.get('id')
                self.pheromones[qid] += score / len(selection)

class RandomForestQuestionSelector:
    def __init__(self, questions_file: str):
        self.questions_file = questions_file
        self.questions = self._load_questions()
        self.category_weights = self._init_category_weights()
        self.group_weights = GROUP_WEIGHTS_DEFAULT.copy()
        self.difficulty_weights = {'简单': 0.8, '中等': 1.0, '较难': 1.2}
        self.used_questions: Set[int] = set()
        self.feature_importance = {}
        self.category_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'by_difficulty': {}})
        self.rng = RandomGenerator()
        self.question_history = defaultdict(list)
        self.diversity_factor = 0.3
        self.exploration_rate = 0.15
        self.deduplicator = QuestionDeduplicator()
        self.aco_optimizer = AntColonyOptimizer(self.questions)
        self.use_aco = True
        self.session_selected: Dict[str, Set[int]] = defaultdict(set)
        
        self._init_advanced_math_models()
        
    def _init_advanced_math_models(self):
        categories = list(set(q.get('category', '未知') for q in self.questions))
        n_categories = len(categories)
        n_features = 6
        
        self.number_theory_encoder = NumberTheoryRandomEncoder(seed=42, prime_count=50)
        
        self.monte_carlo_sampler = MonteCarloStratifiedSamplingFramework(
            n_features=n_features,
            n_categories=n_categories,
            n_stages=3,
            base_samples=100,
            sequence_type='sobol'
        )
        
        self.membership_mapping = MembershipProbabilityMapping(
            n_categories=n_categories,
            polynomial_degree=3
        )
        
        self.local_optimizer = LocalDeploymentOptimizer()
        
        self.permutation_sorter = None
        self._permutation_cache: Dict[str, List[int]] = {}
        
        self._advanced_analysis_cache: Dict[str, Any] = {}
        self._feature_modulus_configured = False
        
    def _load_questions(self) -> List[Dict]:
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _init_category_weights(self) -> Dict[str, float]:
        categories = set(q.get('category', '未知') for q in self.questions)
        return {cat: 1.0 for cat in categories}
    
    def _extract_features(self, question: Dict) -> np.ndarray:
        features = []
        
        category = question.get('category', '未知')
        categories = list(set(q.get('category', '未知') for q in self.questions))
        cat_vector = [1.0 if c == category else 0.0 for c in sorted(categories)]
        features.extend(cat_vector)
        
        difficulty = question.get('difficulty', '中等')
        diff_map = {'简单': 0.3, '中等': 0.5, '较难': 0.7}
        features.append(diff_map.get(difficulty, 0.5))
        
        seen = question.get('seen', 0)
        features.append(min(seen / 10.0, 1.0))
        
        wrong_count = question.get('wrong_count', 0)
        features.append(min(wrong_count / 5.0, 1.0))
        
        correct_count = question.get('correct_count', 0)
        features.append(min(correct_count / 5.0, 1.0))
        
        status = question.get('status', 'normal')
        status_map = {'normal': 0.0, 'weak': 1.0, 'mastered': 0.0}
        features.append(status_map.get(status, 0.0))
        
        return np.array(features)
    
    def _extract_advanced_features(self, question: Dict) -> Dict[str, float]:
        base_features = self._extract_features(question)
        
        advanced_features = {}
        
        feature_dict = {
            'difficulty': base_features[len(base_features) - 5] if len(base_features) > 5 else 0.5,
            'seen': base_features[len(base_features) - 4] if len(base_features) > 4 else 0.0,
            'wrong_count': base_features[len(base_features) - 3] if len(base_features) > 3 else 0.0,
            'correct_count': base_features[len(base_features) - 2] if len(base_features) > 2 else 0.0,
            'status': base_features[len(base_features) - 1] if len(base_features) > 1 else 0.0,
        }
        
        if not self._feature_modulus_configured:
            for i, fname in enumerate(feature_dict.keys()):
                self.number_theory_encoder.set_feature_modulus_mapping(fname, i % 5)
            self._feature_modulus_configured = True
        
        modulus_index = self.number_theory_encoder.dynamic_modulus_switch(base_features)
        
        for fname, fval in feature_dict.items():
            advanced_features[f'nt_{fname}'] = self.number_theory_encoder.map_feature_to_random(fval, fname)
        
        advanced_features['modulus_index'] = float(modulus_index)
        
        multi_mapped = self.number_theory_encoder.multi_modulus_mapping(feature_dict)
        for k, v in multi_mapped.items():
            advanced_features[f'multi_{k}'] = v
        
        return advanced_features
    
    def _calculate_question_score(self, question: Dict, context: Dict = None) -> float:
        score = 0.0
        
        category = question.get('category', '未知')
        cat_weight = self.category_weights.get(category, 1.0)
        group = get_category_group(category)
        group_weight = self.group_weights.get(group, 1.0)
        
        combined_weight = cat_weight * group_weight
        score += combined_weight * 10
        
        if question['id'] in self.used_questions:
            return -1000
        
        difficulty = question.get('difficulty', '中等')
        diff_weight = self.difficulty_weights.get(difficulty, 1.0)
        
        stats = self.category_stats.get(category, {'correct': 0, 'wrong': 0, 'by_difficulty': {}})
        diff_stats = stats.get('by_difficulty', {}).get(difficulty, {'correct': 0, 'total': 0})
        
        if diff_stats.get('total', 0) >= 2:
            diff_accuracy = diff_stats.get('correct', 0) / diff_stats['total']
            if diff_accuracy < 0.6:
                score += 5 * diff_weight
            elif diff_accuracy > 0.8:
                score -= 3
        
        seen = question.get('seen', 0)
        recency_penalty = 0
        if question['id'] in self.question_history:
            history = self.question_history[question['id']]
            if history:
                time_decay = math.exp(-0.1 * len(history))
                recency_penalty = 3 * time_decay
        score -= seen * 0.5 + recency_penalty
        
        wrong_count = question.get('wrong_count', 0)
        correct_count = question.get('correct_count', 0)
        if wrong_count > correct_count:
            score += 5
        
        status = question.get('status', 'normal')
        if status == 'weak':
            score += 8
        elif status == 'mastered':
            score -= 20
        
        advanced_features = self._extract_advanced_features(question)
        
        nt_factor = advanced_features.get('nt_wrong_count', 0.5)
        score += nt_factor * 3
        
        nt_status = advanced_features.get('nt_status', 0.0)
        if nt_status > 0.5:
            score += 2
        
        modulus_idx = advanced_features.get('modulus_index', 0)
        score += (modulus_idx % 3) * 0.5
        
        noise = self.rng.gaussian_noise(0, 2)
        nt_noise = self.number_theory_encoder.generate_random(-1, 1)
        score += noise * 0.7 + nt_noise * 0.3
        
        if context and 'selected_categories' in context:
            cat_count = context['selected_categories'].get(category, 0)
            if cat_count > 3:
                score -= cat_count * 2
        
        return score
    
    def _calculate_diversity_score(self, question: Dict, selected: List[Dict]) -> float:
        if not selected:
            return 0.0
        
        diversity_score = 0.0
        selected_categories = [q.get('category', '未知') for q in selected]
        selected_difficulties = [q.get('difficulty', '中等') for q in selected]
        
        category = question.get('category', '未知')
        cat_count = selected_categories.count(category)
        diversity_score -= cat_count * 1.5
        
        difficulty = question.get('difficulty', '中等')
        diff_count = selected_difficulties.count(difficulty)
        diversity_score -= diff_count * 0.5
        
        question_words = set(question.get('question', '').split())
        for sel_q in selected[-5:]:
            sel_words = set(sel_q.get('question', '').split())
            overlap = len(question_words & sel_words)
            diversity_score -= overlap * 0.1
        
        return diversity_score
    
    def select_questions(self, count: int = DEFAULT_QUESTION_COUNT, session_id: str = None) -> List[Dict]:
        self.rng = RandomGenerator()
        
        if session_id:
            self.deduplicator.selected_ids = self.session_selected[session_id].copy()
        else:
            self.deduplicator.clear()
        
        available = [q for q in self.questions if q['id'] not in self.used_questions]
        
        if len(available) < count:
            self.used_questions.clear()
            self.deduplicator.clear()
            if session_id:
                self.session_selected[session_id].clear()
            available = self.questions.copy()
        
        selected = self._select_with_advanced_models(count, available, session_id)
        
        if selected:
            return selected
        
        if self.use_aco and count <= 100:
            selected = self.aco_optimizer.optimize_selection(
                count, self.category_weights, self.group_weights, self.deduplicator
            )
            
            final_selected = []
            for q in selected:
                if not self.deduplicator.is_duplicate(q, final_selected):
                    final_selected.append(q)
                    self.deduplicator.add_question(q)
                    self.used_questions.add(q['id'])
                    if session_id:
                        self.session_selected[session_id].add(q['id'])
            
            if len(final_selected) < count:
                remaining = self._select_remaining_questions(count - len(final_selected), final_selected, available, session_id)
                final_selected.extend(remaining)
            
            final_selected = self._balance_difficulty(final_selected, self._calculate_difficulty_distribution(count))
            final_selected = self.rng.fisher_yates_shuffle(final_selected)
            return final_selected
        
        return self._select_questions_standard(count, available, session_id)
    
    def _select_with_advanced_models(self, count: int, available: List[Dict], session_id: str = None) -> Optional[List[Dict]]:
        try:
            labels = [hash(q.get('category', '未知')) % 10 for q in available]
            self.monte_carlo_sampler.fit(available, labels)
            
            feature_bounds = {
                'difficulty': (0.0, 1.0),
                'seen': (0.0, 1.0),
                'wrong_count': (0.0, 1.0),
                'correct_count': (0.0, 1.0),
                'status': (0.0, 1.0),
            }
            
            def predict_func(sample: Dict) -> int:
                return hash(sample.get('category', '未知')) % 10
            
            sampling_result = self.monte_carlo_sampler.sample(
                n_samples=min(count, len(available)),
                samples=available,
                labels=labels,
                predict_func=predict_func,
                feature_bounds=feature_bounds,
                use_qmc=True,
                use_importance_iteration=True
            )
            
            sampled_questions = sampling_result['samples']
            weights = sampling_result['weights']
            
            selected = []
            for i, q in enumerate(sampled_questions):
                if len(selected) >= count:
                    break
                if not self.deduplicator.is_duplicate(q, selected):
                    selected.append(q)
                    self.deduplicator.add_question(q)
                    self.used_questions.add(q['id'])
                    if session_id:
                        self.session_selected[session_id].add(q['id'])
            
            if len(selected) < count:
                remaining = self._select_remaining_questions(
                    count - len(selected), selected, available, session_id
                )
                selected.extend(remaining)
            
            selected = self._balance_difficulty(selected, self._calculate_difficulty_distribution(count))
            
            if len(selected) >= 5:
                try:
                    if self.permutation_sorter is None or self.permutation_sorter.n != len(selected):
                        category_memberships = {}
                        for q in selected:
                            cat = q.get('category', '未知')
                            if cat not in category_memberships:
                                category_memberships[cat] = np.zeros(len(selected))
                            idx = selected.index(q)
                            category_memberships[cat][idx] = 1.0
                        
                        self.permutation_sorter = PermutationGroupRandomSorter(
                            n=len(selected),
                            category_memberships=category_memberships if category_memberships else None
                        )
                    
                    selected = self.permutation_sorter.advanced_sort(
                        selected,
                        category=None,
                        mode='hybrid',
                        effectiveness_threshold=0.3
                    )
                except Exception:
                    selected = self.rng.fisher_yates_shuffle(selected)
            else:
                selected = self.rng.fisher_yates_shuffle(selected)
            
            self._advanced_analysis_cache[session_id or 'default'] = {
                'monte_carlo_quality': sampling_result.get('quality_metrics', {}),
                'sampling_summary': self.monte_carlo_sampler.get_sampling_summary(),
                'selected_count': len(selected),
            }
            
            return selected
            
        except Exception as e:
            return None
    
    def _select_questions_standard(self, count: int, available: List[Dict], session_id: str = None) -> List[Dict]:
        category_distribution = self._calculate_category_distribution(count)
        difficulty_distribution = self._calculate_difficulty_distribution(count)
        
        selected = []
        selected_categories = defaultdict(int)
        selected_difficulties = defaultdict(int)
        
        context = {
            'selected_categories': selected_categories,
            'selected_difficulties': selected_difficulties
        }
        
        for category, target in category_distribution.items():
            cat_questions = [q for q in available 
                           if q.get('category') == category 
                           and q['id'] not in self.used_questions
                           and not self.deduplicator.is_duplicate(q, selected)]
            
            if not cat_questions:
                continue
            
            scored_questions = []
            for q in cat_questions:
                base_score = self._calculate_question_score(q, context)
                diversity_score = self._calculate_diversity_score(q, selected)
                total_score = base_score + diversity_score * self.diversity_factor
                scored_questions.append((q, total_score))
            
            take = min(target, len(scored_questions))
            
            if self.rng.rng.random() < self.exploration_rate:
                weighted_items = [(q, max(0.1, score + 10)) for q, score in scored_questions]
                chosen = self.rng.weighted_random_sample(weighted_items, take)
            else:
                scored_questions.sort(key=lambda x: x[1], reverse=True)
                chosen = [q for q, _ in scored_questions[:take]]
            
            for q in chosen:
                if not self.deduplicator.is_duplicate(q, selected):
                    selected.append(q)
                    self.deduplicator.add_question(q)
                    self.used_questions.add(q['id'])
                    if session_id:
                        self.session_selected[session_id].add(q['id'])
                    selected_categories[q.get('category', '未知')] += 1
                    selected_difficulties[q.get('difficulty', '中等')] += 1
                    self.question_history[q['id']].append(len(selected))
        
        remaining = count - len(selected)
        if remaining > 0:
            remaining_selected = self._select_remaining_questions(remaining, selected, available, session_id)
            selected.extend(remaining_selected)
        
        selected = self._balance_difficulty(selected, difficulty_distribution)
        selected = self.rng.fisher_yates_shuffle(selected)
        
        return selected
    
    def _select_remaining_questions(self, count: int, selected: List[Dict], 
                                   available: List[Dict], session_id: str = None) -> List[Dict]:
        remaining_questions = [q for q in available 
                              if q['id'] not in self.used_questions 
                              and not self.deduplicator.is_duplicate(q, selected)]
        
        if not remaining_questions:
            return []
        
        scored_remaining = []
        for q in remaining_questions:
            base_score = self._calculate_question_score(q, None)
            diversity_score = self._calculate_diversity_score(q, selected)
            total_score = base_score + diversity_score * self.diversity_factor
            scored_remaining.append((q, total_score))
        
        weighted_items = [(q, max(0.1, score + 10)) for q, score in scored_remaining]
        chosen = self.rng.weighted_random_sample(weighted_items, count)
        
        result = []
        for q in chosen:
            if not self.deduplicator.is_duplicate(q, selected):
                result.append(q)
                self.deduplicator.add_question(q)
                self.used_questions.add(q['id'])
                if session_id:
                    self.session_selected[session_id].add(q['id'])
                self.question_history[q['id']].append(len(selected) + len(result))
        
        return result
    
    def _balance_difficulty(self, selected: List[Dict], target_distribution: Dict[str, int]) -> List[Dict]:
        current_difficulty = defaultdict(list)
        for q in selected:
            diff = q.get('difficulty', '中等')
            current_difficulty[diff].append(q)
        
        balanced = []
        used_ids = set()
        
        for difficulty, target_count in target_distribution.items():
            available = current_difficulty.get(difficulty, [])
            take = min(target_count, len(available))
            
            chosen = self.rng.rng.sample(available, take) if take > 0 else []
            for q in chosen:
                if q['id'] not in used_ids:
                    balanced.append(q)
                    used_ids.add(q['id'])
        
        for diff, questions in current_difficulty.items():
            for q in questions:
                if q['id'] not in used_ids:
                    balanced.append(q)
                    used_ids.add(q['id'])
        
        return balanced
    
    def _calculate_category_distribution(self, total: int) -> Dict[str, int]:
        categories = defaultdict(int)
        for q in self.questions:
            categories[q.get('category', '未知')] += 1
        
        total_questions = len(self.questions)
        distribution = {}
        
        group_category_counts = defaultdict(list)
        for cat, count in categories.items():
            group = get_category_group(cat)
            group_category_counts[group].append((cat, count))
        
        group_totals = defaultdict(int)
        for group, cat_list in group_category_counts.items():
            group_totals[group] = sum(c for _, c in cat_list)
        
        total_weighted = sum(
            group_totals[g] * self.group_weights.get(g, 1.0) 
            for g in group_totals
        )
        
        for group, cat_list in group_category_counts.items():
            group_weight = self.group_weights.get(group, 1.0)
            group_share = (group_totals[group] * group_weight) / total_weighted
            group_target = max(1, int(total * group_share))
            
            group_cat_total = sum(c for _, c in cat_list)
            for cat, count in cat_list:
                cat_ratio = count / group_cat_total if group_cat_total > 0 else 0
                cat_weight = self.category_weights.get(cat, 1.0)
                weighted_ratio = cat_ratio * cat_weight
                
                cat_target = max(1, int(group_target * weighted_ratio))
                distribution[cat] = cat_target
        
        current_total = sum(distribution.values())
        diff = total - current_total
        
        if diff > 0:
            sorted_cats = sorted(
                distribution.keys(),
                key=lambda c: (self.category_weights.get(c, 1.0), -distribution[c]),
                reverse=True
            )
            idx = 0
            while diff > 0:
                distribution[sorted_cats[idx % len(sorted_cats)]] += 1
                diff -= 1
                idx += 1
        elif diff < 0:
            sorted_cats = sorted(
                distribution.keys(),
                key=lambda c: (distribution[c], -self.category_weights.get(c, 1.0)),
                reverse=True
            )
            idx = 0
            while diff < 0 and distribution[sorted_cats[idx % len(sorted_cats)]] > 1:
                if distribution[sorted_cats[idx % len(sorted_cats)]] > 1:
                    distribution[sorted_cats[idx % len(sorted_cats)]] -= 1
                    diff += 1
                idx += 1
        
        return distribution
    
    def _calculate_difficulty_distribution(self, total: int) -> Dict[str, int]:
        distribution = {}
        remaining = total
        
        for diff, ratio in DIFFICULTY_DISTRIBUTION.items():
            count = int(total * ratio)
            distribution[diff] = count
            remaining -= count
        
        if remaining > 0:
            distribution['中等'] += remaining
        
        return distribution
    
    def update_weights_from_result(self, question_id: int, is_correct: bool):
        question = next((q for q in self.questions if q['id'] == question_id), None)
        if not question:
            return
        
        category = question.get('category', '未知')
        difficulty = question.get('difficulty', '中等')
        
        if category not in self.category_stats:
            self.category_stats[category] = {'correct': 0, 'wrong': 0, 'by_difficulty': {}}
        
        stats = self.category_stats[category]
        if is_correct:
            stats['correct'] += 1
        else:
            stats['wrong'] += 1
        
        if difficulty not in stats['by_difficulty']:
            stats['by_difficulty'][difficulty] = {'correct': 0, 'total': 0}
        
        stats['by_difficulty'][difficulty]['total'] += 1
        if is_correct:
            stats['by_difficulty'][difficulty]['correct'] += 1
        
        current_weight = self.category_weights.get(category, 1.0)
        
        diff_multiplier = {'简单': 1.3, '中等': 1.0, '较难': 0.7}.get(difficulty, 1.0)
        
        if is_correct:
            adjustment = -0.05 * diff_multiplier
        else:
            adjustment = 0.15 * diff_multiplier
        
        noise = self.rng.gaussian_noise(0, 0.02)
        self.category_weights[category] = max(0.5, min(2.0, current_weight + adjustment + noise))
        
        group = get_category_group(category)
        group_weight = self.group_weights.get(group, 1.0)
        if is_correct:
            self.group_weights[group] = max(0.5, group_weight - 0.03)
        else:
            self.group_weights[group] = min(2.0, group_weight + 0.05)
    
    def get_category_weights(self) -> Dict[str, float]:
        return self.category_weights.copy()
    
    def set_category_weights(self, weights: Dict[str, float]):
        for cat, weight in weights.items():
            if cat in self.category_weights:
                self.category_weights[cat] = max(0.5, min(2.0, weight))
    
    def get_group_weights(self) -> Dict[str, float]:
        return self.group_weights.copy()
    
    def set_group_weights(self, weights: Dict[str, float]):
        for group, weight in weights.items():
            if group in self.group_weights:
                self.group_weights[group] = max(0.5, min(2.0, weight))
    
    def get_category_stats(self) -> Dict[str, Dict]:
        return dict(self.category_stats)
    
    def reset_session(self):
        self.used_questions.clear()
        self.deduplicator.clear()
    
    def reset_session_by_id(self, session_id: str):
        if session_id in self.session_selected:
            self.session_selected[session_id].clear()
        self.used_questions.clear()
        self.deduplicator.clear()
    
    def set_diversity_factor(self, factor: float):
        self.diversity_factor = max(0.0, min(1.0, factor))
    
    def set_exploration_rate(self, rate: float):
        self.exploration_rate = max(0.0, min(1.0, rate))
    
    def set_use_aco(self, use: bool):
        self.use_aco = use
    
    def get_advanced_analysis(self, session_id: str = None) -> Dict[str, Any]:
        return self._advanced_analysis_cache.get(session_id or 'default', {})
    
    def optimize_selection_parameters(self, 
                                       objective_func: Optional[Callable[[np.ndarray], float]] = None,
                                       initial_params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        if objective_func is None:
            def default_objective(params: np.ndarray) -> float:
                diversity_penalty = abs(params[0] - 0.3) + abs(params[1] - 0.15)
                return diversity_penalty
            
            objective_func = default_objective
        
        if initial_params is None:
            initial_params = np.array([self.diversity_factor, self.exploration_rate])
        
        optimized_params, metadata = self.local_optimizer.newton_raphson_solve(
            objective_func=objective_func,
            initial_params=initial_params,
            param_shape=(2,)
        )
        
        if len(optimized_params) >= 2:
            self.diversity_factor = max(0.0, min(1.0, optimized_params[0]))
            self.exploration_rate = max(0.0, min(1.0, optimized_params[1]))
        
        return optimized_params, metadata
    
    def update_membership_mapping(self, category: str, 
                                   memberships: np.ndarray, 
                                   probabilities: np.ndarray):
        self.membership_mapping.set_membership_probability_mapping(
            category, memberships, probabilities
        )
    
    def get_distribution_deviation(self, classification_dist: np.ndarray, 
                                    random_dist: np.ndarray) -> Dict[str, float]:
        return self.membership_mapping.monitor_distribution_deviation(
            classification_dist, random_dist
        )
    
    def get_advanced_models_summary(self) -> Dict[str, Any]:
        return {
            'number_theory_encoder': {
                'seed': self.number_theory_encoder.seed,
                'counter': self.number_theory_encoder._counter,
            },
            'monte_carlo_sampler': self.monte_carlo_sampler.get_sampling_summary(),
            'membership_mapping': self.membership_mapping.get_mapping_statistics() if hasattr(self.membership_mapping, 'get_mapping_statistics') else {},
            'permutation_sorter': {
                'initialized': self.permutation_sorter is not None,
                'n': self.permutation_sorter.n if self.permutation_sorter else 0,
            } if self.permutation_sorter else {'initialized': False},
            'local_optimizer': {
                'optimization_history_count': len(self.local_optimizer.optimization_history),
            },
            'advanced_analysis_cache_size': len(self._advanced_analysis_cache),
        }
