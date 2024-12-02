import pandas as pd
import numpy as np

def extract_factor_components(input_file, output_file, n_items=5):
    """
    从因子分析结果中提取每个因子最高和最低载荷的组成部分
    
    Parameters:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    n_items (int): 每个因子要提取的top/bottom项数
    """
    # 读取CSV文件
    df = pd.read_csv(input_file, encoding="utf-8")
    
    # 获取第一列(组成部分名称)和其他列(因子)
    components = df.iloc[:, 0]
    factors = df.iloc[:, 1:]
    
    # 创建结果DataFrame的列表
    results = []
    
    # 处理每个因子
    for factor in factors.columns:
        # 获取当前因子的所有载荷
        loadings = factors[factor]
        
        # 获取最高的n个载荷
        top_indices = loadings.nlargest(n_items).index
        for idx in top_indices:
            results.append({
                'Factor': factor,
                'Type': 'Top',
                'Rank': list(top_indices).index(idx) + 1,
                'Component': components[idx],
                'Loading': loadings[idx]
            })
        
        # 获取最低的n个载荷
        bottom_indices = loadings.nsmallest(n_items).index
        for idx in bottom_indices:
            results.append({
                'Factor': factor,
                'Type': 'Bottom',
                'Rank': list(bottom_indices).index(idx) + 1,
                'Component': components[idx],
                'Loading': loadings[idx]
            })
    
    # 转换结果为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    
    return results_df


# 使用示例
if __name__ == "__main__":
    # 假设输入文件名为 'factor_analysis.csv'
    # 假设输出文件名为 'factor_components.csv'
    results = extract_factor_components('z:\\factor_loadings.csv', 'z:\\factor_components.csv')
    print("处理完成！结果已保存到factor_components.csv")
