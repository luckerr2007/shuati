from advanced_math_models import *
import math

print('=== 测试排列组合决策树 ===')
tree = PermutationCombinationDecisionTree(['feature1', 'feature2', 'feature3'])

print(f'P(5,3) = {tree.permutation_count(5, 3)} (期望: 60)')
print(f'P(10,2) = {tree.permutation_count(10, 2)} (期望: 90)')

print(f'C(5,3) = {tree.combination_count(5, 3)} (期望: 10)')
print(f'C(10,5) = {tree.combination_count(10, 5)} (期望: 252)')

perms = tree.generate_feature_permutations(['a', 'b', 'c'], 2)
print(f'特征排列数: {len(perms)} (期望: 6)')

combs = tree.generate_feature_combinations(['a', 'b', 'c', 'd'], 2)
print(f'特征组合数: {len(combs)} (期望: 6)')

print()
print('=== 测试容斥原理边界修正 ===')
corrector = InclusionExclusionBoundaryCorrector()

corrector.add_set('A', {1, 2, 3, 4, 5})
corrector.add_set('B', {4, 5, 6, 7, 8})
corrector.add_set('C', {5, 6, 9, 10})

union_size = corrector.calculate_union_size(['A', 'B', 'C'])
print(f'并集大小: {union_size} (期望: 10)')

intersection_size = corrector.calculate_intersection_size(['A', 'B'])
print(f'A与B交集大小: {intersection_size} (期望: 2)')

boundaries = corrector.correct_classification_boundary({
    'A': {1, 2, 3, 4, 5},
    'B': {4, 5, 6, 7, 8}
})
print(f'边界修正结果: {boundaries}')

print()
print('=== 测试鸽巢原理分桶 ===')
adjuster = PigeonholeBucketAdjuster()

min_buckets = adjuster.calculate_min_buckets(100, 10)
print(f'最小桶数(100项,容量10): {min_buckets} (期望: 10)')

items = list(range(50))
buckets = adjuster.dynamic_adjust_buckets(items, lambda x: x)
print(f'分桶数量: {len(buckets)}')

violations = adjuster.detect_pigeonhole_violation({0: [1,2,3], 1: [4,5,6,7,8,9]}, 4)
print(f'违反容量限制的桶: {violations} (期望: [1])')

stats = adjuster.get_bucket_distribution_stats(buckets)
print('分桶统计: mean={:.2f}, variance={:.4f}'.format(stats['mean'], stats['variance']))

print()
print('=== 测试组合计数验证 ===')
validator = CombinationCountValidator()

prob = validator.calculate_combination_probability(10, 3, 5)
print(f'组合概率: {prob:.6f}')

observed = {'A': 30, 'B': 40, 'C': 30}
expected = {'A': 33.3, 'B': 33.3, 'C': 33.3}
result = validator.validate_distribution(observed, expected)
print('分布验证结果: valid={}, chi_square={:.4f}'.format(result['valid'], result['chi_square']))

multi_prob = validator.calculate_multinomial_probability(10, {'A': 4, 'B': 3, 'C': 3}, {'A': 0.4, 'B': 0.3, 'C': 0.3})
print(f'多项分布概率: {multi_prob:.6f}')

print()
print('=== 测试综合模型 ===')
model = CombinatorialDecisionModel(['f1', 'f2', 'f3'])

samples = [
    {'f1': 1, 'f2': 2, 'f3': 3, 'id': 1},
    {'f1': 4, 'f2': 5, 'f3': 6, 'id': 2},
    {'f1': 7, 'f2': 8, 'f3': 9, 'id': 3},
    {'f1': 1, 'f2': 2, 'f3': 3, 'id': 4},
]
labels = [0, 1, 1, 0]

model.fit(samples, labels)
prediction, confidence = model.predict({'f1': 1, 'f2': 2, 'f3': 3})
print('预测结果: class={}, confidence={:.4f}'.format(prediction, confidence))

print()
print('=== 所有测试通过! ===')
