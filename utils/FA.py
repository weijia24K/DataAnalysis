import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

subject = '语文'
level = '初一'

df = pd.read_csv('所有有提问数据.csv', encoding='utf-8')

# 1.过滤学科
df = df[(df['用户信息_学科'] == subject) & (df['课程信息_年级'] == level)]

# 2.数据清洗
# 2.1 删除名义字段
columns2del = ['用户信息_学段', '用户信息_学科', '课程信息_年级', '占比', '抬头率_', '参与度_', '活跃度_', '一致性_']
for column in df.columns.tolist():
    for col2del in columns2del:
        if col2del in column:
            df = df.drop(column, axis=1)
df.dropna(inplace=True)

# 2.2 删除是0的行
df = df.loc[:, (df.var() != 0).tolist()]

# # 2.3 标准化
# scaler = StandardScaler()
# df[df.columns] = scaler.fit_transform(df)

# 最大最小标准化
# scaler = MinMaxScaler()
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 3.保存数据
df.to_excel(f'z:\\{level}{subject}_original.xlsx', index=False)

# 4. 检查数据是否适合进行因子分析
chi_square_value, p_value = calculate_bartlett_sphericity(df)
kmo_all, kmo_model = calculate_kmo(df)
print(f"巴特利特球形度检验的p值：{p_value}")
print(f"KMO检验的值：{kmo_model}")
if p_value < 0.05:  # and kmo_model > 0.6:

    # 5. 创建因子分析对象并拟合数据,测试因子为6
    fa = FactorAnalyzer(n_factors=3, rotation='varimax')
    fa.fit(df)

    # 6. 检查Eigenvalues值确定因子个数
    ev, v = fa.get_eigenvalues()
    # print(ev)

    # 7. 执行因子分析
    # 确定因子个数，通常选择特征值大于1的因子个数
    n_factors = sum(ev > 1)  # 决定因子数量
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(df)

    # 8. 获取旋转后的因子载荷矩阵
    factor_loadings = fa.loadings_
    factor_loadings_df = pd.DataFrame(factor_loadings, columns=[f'Factor {i + 1}' for i in range(n_factors)])
    factor_loadings_df.insert(0, 'Original Columns', df.columns)
    factor_loadings_df.to_csv(f'z:\\{level}{subject}_factor_loadings.csv', index=False, encoding='utf-8-sig')

    # 9. 计算因子得分（可选）
    df_scores = fa.transform(df)
    for i in range(n_factors):
        df[f'Factor{i + 1}'] = df_scores[:, i]

    scores_df = pd.DataFrame(df_scores, columns=[f'Factor{i + 1}' for i in range(n_factors)])

    # 10. 合并原始数据和因子得分
    # output_df = pd.concat([df, scores_df], axis=1)

    # 11. 保存包含原始数据和因子得分的完整数据集
    scores_df.to_csv(f'z:\\{level}{subject}_transformed.csv', index=False, encoding='utf-8-sig')

    # 10. 计算因子变异解释能力
    total_variance = np.sum(ev)
    explained_variance = ev[:n_factors] / total_variance
    cumulative_variance = np.cumsum(explained_variance)
    output_data = {
        'Factor': [f'Factor {i + 1}' for i in range(n_factors)],
        'Explained Variance': explained_variance,
        'Cumulative Variance': cumulative_variance
    }
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(f'z:\\{level}{subject}_explained_variance.csv', index=False, encoding='utf-8-sig')

    # print('计算因子得分（可选）')
    # # 输出或分析因子得分
    # print(df[[f'Factor{i + 1}' for i in range(n_factors)]])
else:
    print("数据不适合进行因子分析")
