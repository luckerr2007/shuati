import json
import math
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Callable

from advanced_math_models import (
    HigherOrderMomentsAnalyzer,
    ErgodicityAnalyzer,
    GlobalSensitivityAnalyzer
)

CATEGORY_GROUPS = {
    '文学类': ['人文常识', '历史常识', '地理常识', '安徽文化常识', '传统文化常识', '文学常识'],
    '专业类': ['计算机基础', '网络基础', '数据库', '编程语言', '软件工程', '信息安全', '操作系统', '数据结构', '计算机组成原理', '专业课题目'],
    '政治类': ['政治常识', '法律常识', '经济常识', '时事政治'],
    '科技类': ['科技常识', '生活常识', '自然科学'],
    '其他': ['未知']
}

def get_category_group(category: str) -> str:
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return group
    return '其他'

class LocalOptimizer:
    def __init__(self):
        self.difficulty_weights = {
            '简单': {'wrong_penalty': 0.25, 'correct_bonus': -0.05},
            '中等': {'wrong_penalty': 0.15, 'correct_bonus': -0.05},
            '较难': {'wrong_penalty': 0.10, 'correct_bonus': -0.10}
        }
        
        self.category_stats = {}
        self.group_stats = {}
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_history = defaultdict(list)
        
        self.higher_order_analyzer = HigherOrderMomentsAnalyzer(
            confidence_level=0.95, n_bootstrap=500
        )
        self.ergodicity_analyzer = ErgodicityAnalyzer(
            n_bins=10, convergence_threshold=0.01
        )
        self.sensitivity_analyzer = GlobalSensitivityAnalyzer(
            n_samples=500, n_bootstrap=50
        )
        
        self.moment_analysis_cache: Dict[str, Any] = {}
        self.ergodicity_cache: Dict[str, Any] = {}
        self.sensitivity_cache: Dict[str, Any] = {}
    
    def calculate_weight_adjustment(self, accuracy: float, difficulty: str, 
                                     total_questions: int) -> float:
        base_adjustment = 0
        
        if accuracy < 50:
            base_adjustment = 0.30
        elif accuracy < 60:
            base_adjustment = 0.20
        elif accuracy < 70:
            base_adjustment = 0.10
        elif accuracy < 80:
            base_adjustment = 0.05
        else:
            base_adjustment = -0.05
        
        diff_factor = {'简单': 1.3, '中等': 1.0, '较难': 0.7}.get(difficulty, 1.0)
        
        confidence_factor = min(1.0, total_questions / 10.0)
        
        return base_adjustment * diff_factor * confidence_factor
    
    def batch_analyze_and_update(self, answer_history: List[Dict], 
                                  current_weights: Dict[str, float],
                                  use_ai: bool = False) -> Dict[str, Any]:
        if not answer_history:
            return {
                'weights': current_weights,
                'group_weights': {},
                'category_analysis': {},
                'group_analysis': {},
                'weak_categories': [],
                'weak_groups': [],
                'suggestions': [],
                'higher_order_analysis': {},
                'ergodicity_analysis': {}
            }
        
        category_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
        difficulty_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
        group_stats = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'total': 0})
        
        for answer in answer_history:
            category = answer.get('category', '未知')
            difficulty = answer.get('difficulty', '中等')
            is_correct = answer.get('is_correct', False)
            group = get_category_group(category)
            
            category_stats[category]['total'] += 1
            difficulty_stats[difficulty]['total'] += 1
            group_stats[group]['total'] += 1
            
            if is_correct:
                category_stats[category]['correct'] += 1
                difficulty_stats[difficulty]['correct'] += 1
                group_stats[group]['correct'] += 1
            else:
                category_stats[category]['wrong'] += 1
                difficulty_stats[difficulty]['wrong'] += 1
                group_stats[group]['wrong'] += 1
        
        category_analysis = {}
        for cat, stats in category_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                category_analysis[cat] = {
                    'correct': stats['correct'],
                    'wrong': stats['wrong'],
                    'total': stats['total'],
                    'accuracy': round(accuracy, 1)
                }
        
        group_analysis = {}
        for group, stats in group_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                group_analysis[group] = {
                    'correct': stats['correct'],
                    'wrong': stats['wrong'],
                    'total': stats['total'],
                    'accuracy': round(accuracy, 1)
                }
        
        higher_order_analysis = self._perform_higher_order_analysis(
            category_stats, answer_history
        )
        
        ergodicity_analysis = self._perform_ergodicity_analysis(answer_history)
        
        algorithm_weights = current_weights.copy()
        
        for category, stats in category_stats.items():
            if stats['total'] >= 1:
                accuracy = stats['correct'] / stats['total']
                
                avg_difficulty = self._calculate_avg_difficulty_for_category(
                    answer_history, category
                )
                
                base_adjustment = self.calculate_weight_adjustment(
                    accuracy * 100, avg_difficulty, stats['total']
                )
                
                higher_order_factor = self._get_higher_order_adjustment_factor(
                    category, higher_order_analysis
                )
                
                adjustment = base_adjustment * higher_order_factor
                
                if category in algorithm_weights:
                    old_weight = algorithm_weights[category]
                    self.weight_history[category].append(adjustment)
                    
                    if len(self.weight_history[category]) > 5:
                        momentum_adjustment = sum(
                            self.weight_history[category][-5:]
                        ) / 5 * self.momentum
                        adjustment = adjustment * (1 - self.momentum) + momentum_adjustment
                    
                    new_weight = old_weight + adjustment * self.learning_rate
                    algorithm_weights[category] = max(0.5, min(2.0, new_weight))
        
        group_weights = {}
        for group, stats in group_stats.items():
            if stats['total'] >= 1:
                accuracy = stats['correct'] / stats['total']
                if accuracy < 0.6:
                    group_weights[group] = 1.2
                elif accuracy < 0.8:
                    group_weights[group] = 1.0
                else:
                    group_weights[group] = 0.8
        
        weak_categories = self._identify_weak_categories(category_analysis)
        weak_groups = self._identify_weak_groups(group_analysis)
        suggestions = self._generate_suggestions(category_analysis, group_analysis)
        
        return {
            'weights': algorithm_weights,
            'group_weights': group_weights,
            'category_analysis': category_analysis,
            'group_analysis': group_analysis,
            'weak_categories': weak_categories,
            'weak_groups': weak_groups,
            'suggestions': suggestions,
            'higher_order_analysis': higher_order_analysis,
            'ergodicity_analysis': ergodicity_analysis
        }
    
    def _calculate_avg_difficulty_for_category(self, answer_history: List[Dict], 
                                                category: str) -> str:
        difficulties = []
        for answer in answer_history:
            if answer.get('category') == category:
                diff = answer.get('difficulty', '中等')
                diff_map = {'简单': 1, '中等': 2, '较难': 3}
                difficulties.append(diff_map.get(diff, 2))
        
        if not difficulties:
            return '中等'
        
        avg = sum(difficulties) / len(difficulties)
        if avg < 1.5:
            return '简单'
        elif avg < 2.5:
            return '中等'
        else:
            return '较难'
    
    def _identify_weak_categories(self, category_analysis: Dict) -> List[str]:
        weak = []
        for cat, stats in category_analysis.items():
            if stats['total'] >= 2 and stats['accuracy'] < 60:
                weak.append((cat, stats['accuracy']))
        
        weak.sort(key=lambda x: x[1])
        return [cat for cat, _ in weak[:5]]
    
    def _identify_weak_groups(self, group_analysis: Dict) -> List[str]:
        weak = []
        for group, stats in group_analysis.items():
            if stats['total'] >= 3 and stats['accuracy'] < 70:
                weak.append((group, stats['accuracy']))
        
        weak.sort(key=lambda x: x[1])
        return [group for group, _ in weak[:3]]
    
    def _generate_suggestions(self, category_analysis: Dict, group_analysis: Dict) -> List[str]:
        suggestions = []
        
        weak_cats = self._identify_weak_categories(category_analysis)
        if weak_cats:
            suggestions.append(f"建议重点复习: {', '.join(weak_cats[:3])}")
        
        weak_groups = self._identify_weak_groups(group_analysis)
        if weak_groups:
            suggestions.append(f"需要加强的领域: {', '.join(weak_groups)}")
        
        total_correct = sum(s['correct'] for s in category_analysis.values())
        total_questions = sum(s['total'] for s in category_analysis.values())
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        if overall_accuracy >= 80:
            suggestions.append("表现优秀！继续保持，可以挑战更难的题目")
        elif overall_accuracy >= 60:
            suggestions.append("表现良好，继续巩固薄弱知识点")
        else:
            suggestions.append("需要加强基础知识学习，多做练习")
        
        return suggestions
    
    def get_difficulty_for_category(self, category: str) -> float:
        stats = self.category_stats.get(category, {'correct': 0, 'total': 0})
        if stats['total'] < 3:
            return 1.0
        
        accuracy = stats['correct'] / stats['total']
        if accuracy < 0.5:
            return 1.3
        elif accuracy < 0.7:
            return 1.0
        else:
            return 0.8
    
    def update_from_session(self, session_stats: Dict):
        for category, stats in session_stats.get('category_stats', {}).items():
            if category not in self.category_stats:
                self.category_stats[category] = {'correct': 0, 'wrong': 0, 'total': 0}
            
            self.category_stats[category]['correct'] += stats.get('correct', 0)
            self.category_stats[category]['wrong'] += stats.get('wrong', 0)
            self.category_stats[category]['total'] += stats.get('total', 0)
        
        for group, stats in session_stats.get('group_stats', {}).items():
            if group not in self.group_stats:
                self.group_stats[group] = {'correct': 0, 'wrong': 0, 'total': 0}
            
            self.group_stats[group]['correct'] += stats.get('correct', 0)
            self.group_stats[group]['wrong'] += stats.get('wrong', 0)
            self.group_stats[group]['total'] += stats.get('total', 0)
    
    def _perform_higher_order_analysis(self, category_stats: Dict, 
                                        answer_history: List[Dict]) -> Dict[str, Any]:
        if len(answer_history) < 5:
            return {'status': 'insufficient_data', 'message': '数据不足，至少需要5条答题记录'}
        
        accuracy_sequence = []
        for answer in answer_history:
            is_correct = answer.get('is_correct', False)
            accuracy_sequence.append(1.0 if is_correct else 0.0)
        
        accuracy_array = np.array(accuracy_sequence)
        
        try:
            moments = self.higher_order_analyzer.compute_moments(accuracy_array)
            
            if len(accuracy_sequence) >= 10:
                skewness_ci = self.higher_order_analyzer.bootstrap_moment_confidence_interval(
                    accuracy_array, 'skewness'
                )
                kurtosis_ci = self.higher_order_analyzer.bootstrap_moment_confidence_interval(
                    accuracy_array, 'kurtosis'
                )
            else:
                skewness_ci = {'ci_lower': 0, 'ci_upper': 0}
                kurtosis_ci = {'ci_lower': 0, 'ci_upper': 0}
            
            category_accuracies = []
            category_labels = []
            for cat, stats in category_stats.items():
                if stats['total'] > 0:
                    category_accuracies.append(stats['correct'] / stats['total'])
                    category_labels.append(cat)
            
            cross_moments = {}
            if len(category_accuracies) >= 2:
                cat_array = np.array(category_accuracies)
                for i, cat1 in enumerate(category_labels):
                    for j, cat2 in enumerate(category_labels):
                        if i < j:
                            try:
                                cm = self.higher_order_analyzer.compute_cross_moments(
                                    np.array([category_accuracies[i]] * 10),
                                    np.array([category_accuracies[j]] * 10),
                                    order=(1, 1)
                                )
                                cross_moments[f"{cat1}_{cat2}"] = cm
                            except Exception:
                                pass
            
            result = {
                'status': 'success',
                'moments': {
                    'mean': float(moments['mean']),
                    'variance': float(moments['variance']),
                    'skewness': float(moments['skewness']),
                    'kurtosis': float(moments['kurtosis'])
                },
                'confidence_intervals': {
                    'skewness': skewness_ci,
                    'kurtosis': kurtosis_ci
                },
                'cross_moments': cross_moments,
                'distribution_characteristics': self._interpret_moments(moments)
            }
            
            self.moment_analysis_cache = result
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _interpret_moments(self, moments: Dict) -> Dict[str, str]:
        interpretation = {}
        
        skewness = moments['skewness']
        if skewness > 0.5:
            interpretation['skewness'] = '正偏态：正确率分布偏向较低值，存在较多低分情况'
        elif skewness < -0.5:
            interpretation['skewness'] = '负偏态：正确率分布偏向较高值，表现整体较好'
        else:
            interpretation['skewness'] = '近似对称：正确率分布较为均匀'
        
        kurtosis = moments['kurtosis']
        if kurtosis > 1.0:
            interpretation['kurtosis'] = '高峰态：正确率集中在均值附近，表现稳定'
        elif kurtosis < -1.0:
            interpretation['kurtosis'] = '低峰态：正确率分散，表现波动较大'
        else:
            interpretation['kurtosis'] = '正态峰度：正确率分布接近正态分布'
        
        return interpretation
    
    def _get_higher_order_adjustment_factor(self, category: str, 
                                             higher_order_analysis: Dict) -> float:
        if higher_order_analysis.get('status') != 'success':
            return 1.0
        
        moments = higher_order_analysis.get('moments', {})
        skewness = moments.get('skewness', 0)
        kurtosis = moments.get('kurtosis', 0)
        
        factor = 1.0
        
        if skewness > 0.5:
            factor *= 1.1
        elif skewness < -0.5:
            factor *= 0.95
        
        if kurtosis > 1.0:
            factor *= 0.95
        elif kurtosis < -1.0:
            factor *= 1.15
        
        return max(0.8, min(1.2, factor))
    
    def _perform_ergodicity_analysis(self, answer_history: List[Dict]) -> Dict[str, Any]:
        if len(answer_history) < 10:
            return {'status': 'insufficient_data', 'message': '数据不足，至少需要10条答题记录'}
        
        accuracy_sequence = []
        for answer in answer_history:
            is_correct = answer.get('is_correct', False)
            accuracy_sequence.append(1.0 if is_correct else 0.0)
        
        accuracy_array = np.array(accuracy_sequence)
        
        try:
            temporal_result = self.ergodicity_analyzer.test_temporal_ergodicity(accuracy_array)
            
            lln_result = self.ergodicity_analyzer.verify_law_of_large_numbers(accuracy_array)
            
            category_indices = {}
            for i, answer in enumerate(answer_history):
                cat = answer.get('category', '未知')
                if cat not in category_indices:
                    category_indices[cat] = []
                category_indices[cat].append(i)
            
            spatial_data = []
            spatial_labels = []
            for cat, indices in category_indices.items():
                if len(indices) >= 3:
                    cat_accuracy = np.mean([accuracy_sequence[i] for i in indices])
                    spatial_data.append(cat_accuracy)
                    spatial_labels.append(cat)
            
            spatial_result = None
            if len(spatial_data) >= 2:
                spatial_array = np.array(spatial_data).reshape(-1, 1)
                spatial_result = self.ergodicity_analyzer.test_spatial_ergodicity(spatial_array)
            
            overall_ergodicity = (
                temporal_result['is_temporally_ergodic'] and 
                lln_result['is_converged']
            )
            if spatial_result is not None:
                overall_ergodicity = overall_ergodicity and spatial_result['is_spatially_ergodic']
            
            result = {
                'status': 'success',
                'temporal_ergodicity': {
                    'is_ergodic': temporal_result['is_temporally_ergodic'],
                    'global_entropy': float(temporal_result['global_entropy']),
                    'mean_stability': float(temporal_result['mean_stability']),
                    'variance_stability': float(temporal_result['variance_stability'])
                },
                'law_of_large_numbers': {
                    'is_converged': lln_result['is_converged'],
                    'final_error': float(lln_result['final_error']),
                    'convergence_speed': float(lln_result['mean_convergence_speed']),
                    'n_for_convergence': lln_result['n_for_convergence']
                },
                'spatial_ergodicity': {
                    'is_ergodic': spatial_result['is_spatially_ergodic'] if spatial_result else None,
                    'uniformity_score': float(spatial_result['uniformity_score']) if spatial_result else None
                } if spatial_result else None,
                'overall_ergodicity': overall_ergodicity,
                'interpretation': self._interpret_ergodicity(
                    temporal_result, lln_result, spatial_result
                )
            }
            
            self.ergodicity_cache = result
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _interpret_ergodicity(self, temporal: Dict, lln: Dict, 
                               spatial: Optional[Dict]) -> Dict[str, str]:
        interpretation = {}
        
        if temporal['is_temporally_ergodic']:
            interpretation['temporal'] = '时间遍历性良好：答题表现随时间稳定'
        else:
            interpretation['temporal'] = '时间遍历性不足：答题表现存在时间波动'
        
        if lln['is_converged']:
            interpretation['convergence'] = f"大数定律收敛：答题表现趋于稳定（收敛样本数: {lln['n_for_convergence']}）"
        else:
            interpretation['convergence'] = '大数定律未收敛：需要更多答题数据'
        
        if spatial is not None:
            if spatial['is_spatially_ergodic']:
                interpretation['spatial'] = '空间遍历性良好：各分类表现均匀'
            else:
                interpretation['spatial'] = '空间遍历性不足：各分类表现差异较大'
        
        return interpretation
    
    def perform_sensitivity_analysis(self, 
                                     model_func: Optional[Callable] = None,
                                     param_bounds: Optional[List[Tuple[float, float]]] = None,
                                     param_names: Optional[List[str]] = None) -> Dict[str, Any]:
        if model_func is None:
            model_func = self._default_weight_model
        
        if param_bounds is None:
            param_bounds = [
                (0.5, 2.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0)
            ]
        
        if param_names is None:
            param_names = ['base_weight', 'accuracy_factor', 'difficulty_factor', 'momentum_factor']
        
        try:
            default_params = np.array([1.0, 0.5, 0.5, 0.5])
            
            derivative_result = self.sensitivity_analyzer.compute_partial_derivatives(
                model_func, default_params, param_names
            )
            
            sobol_result = self.sensitivity_analyzer.compute_sobol_indices(
                model_func, param_bounds, param_names
            )
            
            sensitivity_ranking = sorted(
                sobol_result['sobol_dict'].items(),
                key=lambda x: x[1]['total_order_mean'],
                reverse=True
            )
            
            result = {
                'status': 'success',
                'partial_derivatives': {
                    name: {
                        'mean': float(deriv['mean']),
                        'std': float(deriv['std']),
                        'max': float(deriv['max'])
                    }
                    for name, deriv in derivative_result['partial_derivatives'].items()
                },
                'sobol_indices': {
                    name: {
                        'first_order': float(indices['first_order_mean']),
                        'total_order': float(indices['total_order_mean'])
                    }
                    for name, indices in sobol_result['sobol_dict'].items()
                },
                'sensitivity_ranking': [s[0] for s in sensitivity_ranking],
                'most_sensitive_parameter': sensitivity_ranking[0][0] if sensitivity_ranking else None,
                'interpretation': self._interpret_sensitivity(sensitivity_ranking)
            }
            
            self.sensitivity_cache = result
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _default_weight_model(self, base_weight: float, accuracy_factor: float,
                               difficulty_factor: float, momentum_factor: float) -> float:
        accuracy = 0.7
        difficulty = 1.0
        
        adjustment = (1 - accuracy) * accuracy_factor + difficulty * difficulty_factor
        weight = base_weight * (1 + adjustment * momentum_factor)
        
        return max(0.5, min(2.0, weight))
    
    def _interpret_sensitivity(self, ranking: List[Tuple[str, Dict]]) -> Dict[str, str]:
        interpretation = {}
        
        if not ranking:
            return {'message': '无法分析灵敏度'}
        
        most_sensitive = ranking[0][0]
        interpretation['primary_factor'] = f"最敏感参数: {most_sensitive}，该参数对权重调整影响最大"
        
        if len(ranking) > 1:
            least_sensitive = ranking[-1][0]
            interpretation['secondary_factor'] = f"最不敏感参数: {least_sensitive}，该参数影响较小"
        
        interpretation['recommendation'] = "建议优先调整敏感参数以获得更显著的效果"
        
        return interpretation
    
    def verify_ergodicity(self, answer_history: List[Dict]) -> Dict[str, Any]:
        return self._perform_ergodicity_analysis(answer_history)
    
    def get_moment_summary(self) -> str:
        if self.moment_analysis_cache:
            return self.higher_order_analyzer.get_moment_summary()
        return "尚未进行高阶矩分析"
    
    def get_ergodicity_summary(self) -> str:
        if self.ergodicity_cache:
            return self.ergodicity_analyzer.get_ergodicity_summary()
        return "尚未进行遍历性分析"
    
    def get_sensitivity_summary(self) -> str:
        if self.sensitivity_cache:
            return self.sensitivity_analyzer.get_sensitivity_summary()
        return "尚未进行灵敏度分析"
