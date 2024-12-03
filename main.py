'''
Description: 用于中庆画像
Author: Weijia Ju
version: 
Date: 2024-11-18 22:25:34
LastEditors: Weijia Ju
LastEditTime: 2024-12-03 09:05:30
'''
import streamlit as st
import pandas as pd
import Config as C
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def mapsize(data, x, MIN, MAX, dim):
    '''
    @ Description: 映射数据到指定范围
    @ x: 数据
    @ MIN: 最小值
    @ MAX: 最大值
    @ dim: 维度
    '''
    x_max = np.max(data)
    x_min = np.min(data)
    k = (MAX-MIN)/(x_max-x_min)
    if dim > 4:
        return MIN+k*(x-x_min)
    else:
        return math.floor(MIN+k*(x-x_min))

def draw_cluster(data, best_labels, best_center, fig):
    '''
    @ Description: 绘制聚类图
    @ data: 聚类数据
    @ best_labels: 最佳聚类标签
    @ best_center: 最佳聚类中心
    '''
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    font = FontProperties(fname=r"SimHei.ttf", size=10)
    dimension = len(data.columns)  # 维度
    if dimension == 1:
        # st.info(len(data.index.tolist()))
        # st.info(len(data.iloc[:, 0].tolist()))
        # 绘制散点图
        # 绘制散点图
        plt.scatter(data.index, data.iloc[:, 0], c=best_labels, cmap='Dark2')
        # 绘制聚类中心
        # plt.scatter(best_center.index, best_center[:, 0], c='red', marker='x')
        plt.xlabel('序列', fontproperties= font)
        plt.ylabel(data.columns[0], fontproperties= font)
    if dimension == 2:
        # 绘制散点图
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=best_labels, cmap='Dark2')
        # 绘制聚类中心
        plt.scatter(best_center[:, 0], best_center[:, 1], c='red', marker='x')
        # plt.xticks(fontproperties= font, rotation = -45)
        plt.xlabel(data.columns[0], fontproperties= font)
        plt.ylabel(data.columns[1], fontproperties= font)
    if dimension == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=best_labels, cmap='Dark2')
        ax.scatter3D(best_center[:, 0], best_center[:, 1], best_center[:, 2], c='red', marker='x')
        ax.set_xlabel(data.columns[0], fontproperties= font)
        ax.set_ylabel(data.columns[1], fontproperties= font)
        ax.set_zlabel(data.columns[2], fontproperties= font)
    if dimension == 4:
        # 映射第四维数据到节点的大小
        sizes = []
        for x in data.iloc[:, 3]:
            sizes.append(mapsize(data.iloc[:, 3], x, 10, 100, dimension))
        ax = plt.axes(projection='3d')
        ax.scatter3D(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], s = sizes, c=best_labels, cmap='Dark2')
        ax.scatter3D(best_center[:, 0], best_center[:, 1], best_center[:, 2], c='red', marker='x')
        ax.set_xlabel(data.columns[0], fontproperties= font)
        ax.set_ylabel(data.columns[1], fontproperties= font)
        ax.set_zlabel(data.columns[2], fontproperties= font)
    if dimension == 5:
        # 映射第四维数据到节点的大小
        sizes = []
        line_widths = []
        edgecolors = []
        for x in data.iloc[:, 3]:
            sizes.append(mapsize(data.iloc[:, 3], x, 10, 100, dimension))
        for x in data.iloc[:, 4]:    
            line_widths.append(mapsize(data.iloc[:, 4], x, 0, 1.5, dimension))
            edgecolors.append('red')
        ax = plt.axes(projection='3d')
        ax.scatter3D(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], s = sizes, c=best_labels, linewidths=line_widths, edgecolors=edgecolors,cmap='Dark2')
        ax.scatter3D(best_center[:, 0], best_center[:, 1], best_center[:, 2], c='red', marker='x')
        ax.set_xlabel(data.columns[0], fontproperties= font)
        ax.set_ylabel(data.columns[1], fontproperties= font)
        ax.set_zlabel(data.columns[2], fontproperties= font)
    if dimension >= 6:
        st.warning("维度大于等于6, 无法绘制聚类图")

def draw_index(x, y, cluster_index):
    '''
    @ Description: 绘制最佳聚类指数
    @ x: 聚类个数
    @ y: 聚类指数
    '''
    font_dict = dict(fontsize=16,
                color='black',
                family='Times New Roman',
                weight='light',
                style='italic',
                )
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    plt.plot(x, # 聚类簇
            y, # 指数值
            color = 'red', # 标记颜色
            marker = 'o', # 标记形状
            linestyle = '-', # 标记线型
            linewidth = 2, # 标记线宽
            markersize = 8, # 标记大小
            markerfacecolor = 'red', # 标记颜色
            )
    plt.title(cluster_index, fontdict=font_dict)
    plt.xticks(fontname="Times New Roman", fontsize=10)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    plt.xlabel("Number of Cluster", font_dict)
    plt.ylabel('Number of Index', fontdict=font_dict)
# KNN聚类
def clustering_analysis_k_means(pca_results):
    '''
    @ Description: KNN聚类
    @ pca_results: 处理后的表格数据列表
    @ k: 聚类个数
    '''
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score  # dbi指数
    from sklearn.metrics import calinski_harabasz_score  # CH指数
    cluster_index = st.sidebar.selectbox("选择聚类指数: ", ["DBI", "CH"])
    scalerMethod = st.sidebar.selectbox("标准化方法", ["None","Standard", "MinMax"])
    subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    for grade in grades:
        rows = st.columns(len(pca_results[grade]))
        for i, col in enumerate(rows):
            subject = subjects[i]
            col.header(f'{grade}{subject}课聚类')
            component_data = pca_results[grade][subject]
            # col.info(f'共有{len(component_data)}个成分')
            for component in range(1, len(component_data)):
                data = component_data[component-1]
                if scalerMethod == "None":
                    data = data
                if scalerMethod == "Standard": 
                    # 常规的标准化
                    scaler = StandardScaler()
                    data[data.columns] = scaler.fit_transform(data)
                if scalerMethod == "MinMax":
                    # 最大最小标准化
                    scaler = MinMaxScaler()
                    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                col.info(f'第{component}个成分有{len(data.columns)}维')
                col.dataframe(data.head(5))
                import matplotlib.pyplot as plt
                # col.dataframe(data)
                dbis = []
                chs = []
                min_dbi = 100000
                max_CH = 0
                best_k = 0
                best_labels = []
                best_center = []
                fig_index = plt.figure()
                for k in range(2, 8):
                    kmeans = KMeans(n_clusters=k, random_state=123).fit(data)
                    if cluster_index == "DBI":
                        dbi = davies_bouldin_score(data, kmeans.labels_)
                        dbis.append(dbi)
                        if dbi < min_dbi:
                            min_dbi = dbi
                            best_k = k
                            best_labels = kmeans.labels_
                            best_center = kmeans.cluster_centers_
                    if cluster_index == "CH":
                        CH = calinski_harabasz_score(data, kmeans.labels_)
                        chs.append(CH)
                        if CH > max_CH:
                            max_CH = CH
                            best_k = k
                            best_labels = kmeans.labels_
                            best_center = kmeans.cluster_centers_
                if cluster_index == "DBI":
                    col.info(f'最好的聚类簇为{best_k}, DBI指数为{min_dbi}')
                    draw_index(range(2,8), dbis ,cluster_index)
                if cluster_index == "CH":
                    col.info(f'最好的聚类簇为{best_k}, CH指数为{max_CH}')
                    draw_index(range(2,8), chs, cluster_index)
                col.info('最佳聚类中心')
                col.dataframe(best_center)
                col.pyplot(fig_index)
                plt.close(fig_index)
                # 绘制聚类结果
                fig_cluster = plt.figure()
                draw_cluster(data, best_labels, best_center,fig_cluster)
                col.pyplot(fig_cluster)
                plt.close(fig_cluster)    
# DBSCAN
def clustering_analysis_dbscan(pca_results):
    '''
    @ Description: DBSCAN聚类
    @ pca_results: 处理后的表格数据列表
    '''
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score  # 轮廓系数
    subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    eps = st.sidebar.slider("请输入DBSCAN的eps值", 0, 1000, [10, 900]) # 设置半径
    min_samples = st.sidebar.slider("请输入DBSCAN的min_samples值", 0, 100, [0,90]) # 设置最小样本数
    scalerMethod = st.sidebar.selectbox("标准化方法", ["None","Standard", "MinMax"])
    eps = [i/100 for i in eps]
    for grade in grades:
        rows = st.columns(len(pca_results[grade]))
        for i, col in enumerate(rows):
            subject = subjects[i]
            col.header(f'{grade}{subject}课聚类')
            component_data = pca_results[grade][subject]
            # col.info(f'共有{len(component_data)}个成分')
            for component in range(1, len(component_data)):
                data = component_data[component-1]
                if scalerMethod == "None":
                    data = data
                if scalerMethod == "Standard": 
                    # 常规的标准化
                    scaler = StandardScaler()
                    data[data.columns] = scaler.fit_transform(data)
                if scalerMethod == "MinMax":
                    # 最大最小标准化
                    scaler = MinMaxScaler()
                    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                col.info(f'第{component}个成分有{len(data.columns)}维')
                col.dataframe(data.head(5))
                import matplotlib.pyplot as plt
                # dbis = []
                max_k = 0 # 轮廓系数最大值
                best_labels = []
                best_ratio = []
                best_eps = 0
                best_min_samples = 0
                fig_cluster = plt.figure()
                for eps_i in eps:
                    for min_samples_i in min_samples:
                        try:
                            dbscan = DBSCAN(eps=eps_i, min_samples=min_samples_i).fit(data)
                            labels = dbscan.labels_
                            k = silhouette_score(data, labels)
                            if k > max_k:
                                max_k = k
                                best_labels = labels
                                best_ratio = len(best_labels[best_labels[:]==-1])/len(best_labels)
                                best_eps = eps_i
                                best_min_samples = min_samples_i
                        except:
                            pass
                n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
                col.info(f'共有{n_clusters}个簇, 最佳半径为{best_eps},最小样本数为{best_min_samples}, 轮廓系数为{max_k}, 异常点比例为{best_ratio}')
                # col.dataframe(data.head(5))
                best_centers = []
                best_centers.append([0]*len(data.columns))
                best_centers.append([0]*len(data.columns))
                best_centers = np.array(best_centers)
                # 绘制聚类结果
                draw_cluster(data, labels, best_centers, fig_cluster)
                col.pyplot(fig_cluster)

def draw_loadings(x, y):
    '''
    @ Description: 绘制载荷图
    @ x: 载荷值
    @ y: 成分
    '''
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    font = FontProperties(fname=r"SimHei.ttf", size=10)
    plt.rcParams['axes.unicode_minus'] = False # 显示负号
    bar1 = plt.bar(x, # 成分
                   y, # 载荷值
                   color = np.where(y>0, 'red', 'blue'), # 根据x的值来设置柱形的颜色
                   width = 0.6, # 柱形宽度
                   align = 'center', # 柱形对齐方式
                   alpha = 0.8, # 柱形透明度
                   )
    plt.axhline(y=0, c='k', ls=':', lw=1) # 添加水平线
    plt.title('载荷图', fontproperties=font)
    plt.xticks(fontsize=10, fontproperties=font, rotation=45)
    plt.yticks(fontname="Times New Roman",fontsize=10)
    plt.xlabel("成分", fontproperties=font)
    plt.ylabel("载荷值", fontproperties=font)

def pca_analysis(data):
    '''
    @ Description: PCA降维流程
    @ data: 处理后的表格数据列表
    '''
    st.info("PCA累计贡献率阈值尽量不选100")
    pca_expla_theta = st.slider("请输入PCA累计贡献率阈值", 0, 100, 95)
    pca_expla_theta = pca_expla_theta / 100
    st.info("PCA载荷阈值")
    loadings_theta = st.slider("请输入PCA载荷阈值", 0, 100, 30)
    loadings_theta = [-loadings_theta / 100, loadings_theta / 100]
    st.info(f'载荷值范围{loadings_theta}')
    rows = st.columns(len(data))
    subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    scalerMethod = st.sidebar.selectbox("标准化方法", ["None","Standard", "MinMax"])
    pca_results = {}
    for grade in grades:
        pca_results[grade] = {}
        for i, col in enumerate(rows):
            subject = subjects[i]
            pca_results[grade][subject] = {}
            temp_data = data[i]
            if scalerMethod == "None":
                temp_data = temp_data
            if scalerMethod == "Standard": 
                # 常规的标准化
                scaler = StandardScaler()
                temp_data[temp_data.columns] = scaler.fit_transform(temp_data)
            if scalerMethod == "MinMax":
                # 最大最小标准化
                scaler = MinMaxScaler()
                numeric_columns = temp_data.select_dtypes(include=['float64', 'int64']).columns
                temp_data[numeric_columns] = scaler.fit_transform(temp_data[numeric_columns])
            col.header(f'{grade}{subject}课PCA降维')
            n_components = 2  # 初始成分个数
            flag = True
            while flag:
                pca = PCA(n_components=n_components)
                n_components += 1
                pca.fit(temp_data)
                pca_expla = pca.explained_variance_ratio_ # 方差贡献率
                loadings = pca.components_ # 载荷
                temp_pca = []
                t = 0
                for p in pca_expla[::-1]:
                    t += p
                    temp_pca.append(t)
                if sum(pca_expla) > pca_expla_theta:
                    flag = False
            # 将list转为array
            pca_expla = pd.DataFrame(temp_pca, columns=["累计贡献率"])
            col.info("PCA累计贡献率")
            col.table(pca_expla)
            # col.info("PCA载荷")
            loadings = pd.DataFrame(loadings, columns=temp_data.columns)
            loadings_trans = loadings.transpose()
            # col.table(loadings_trans)
            col.info("PCA载荷降维")
            for component in range(1, n_components):
                # pca_results[grade][subject][component-1] = {}
                temp_loadings = loadings_trans.loc[:, component-1]
                index_name = temp_loadings[(temp_loadings>loadings_theta[0])&(temp_loadings<loadings_theta[1])].index
                temp_loadings = temp_loadings.drop(index_name)
                col.info(f'这里是成分{component}的载荷,其属性有{len(temp_loadings.index.to_list())}')
                col.table(temp_loadings)
                x = temp_loadings.index.to_list() # 成分
                y = np.array(temp_loadings.values.tolist()) # 载荷值
                import matplotlib.pyplot as plt
                fig = plt.figure()
                # 画出载荷双标图   
                draw_loadings(x, y)
                col.pyplot(fig)
                plt.close(fig)
                # 获取PCA降维后的维度的数据
                temp_pca_data = temp_data.loc[:, temp_loadings.index.to_list()]
                # 对每一个部分保存
                pca_results[grade][subject][component-1] = temp_pca_data
            # 查看所保存的数据
            # col.json(pca_results)
    return pca_results

def factor_analysis(data):
    # TODO: 因子分析通过后确定因子个数
    '''
    @ Description: 因子分析流程
    @ data: 处理后的表格数据列表
    '''
    rows = st.columns(len(data))
    subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    scalerMethod = st.sidebar.selectbox("标准化方法", ["None","Standard", "MinMax"])
    for grade in grades:
        for i, col in enumerate(rows):
            subject = subjects[i]
            temp_data = data[i]
            if scalerMethod == "None":
                temp_data = temp_data
            if scalerMethod == "Standard": 
                # 常规的标准化
                scaler = StandardScaler()
                temp_data[temp_data.columns] = scaler.fit_transform(temp_data)
            if scalerMethod == "MinMax":
                # 最大最小标准化
                scaler = MinMaxScaler()
                numeric_columns = temp_data.select_dtypes(include=['float64', 'int64']).columns
                temp_data[numeric_columns] = scaler.fit_transform(temp_data[numeric_columns])
            col.dataframe(temp_data.head(5))
            col.header(f'{grade}{subject}课因子分析')
            # FA(temp_data, col)
            # 
            chi_square_value, p_value = calculate_bartlett_sphericity(temp_data)
            col.info("球状检验p值结果:   "+str(p_value))
            kmo_all, kmo_model = calculate_kmo(temp_data)
            col.info("KMO检验值:   "+str(kmo_model))
            if p_value < 0.05 and kmo_model > 0.6:
                col.success("因子分析通过")
            else:
                col.error("因子分析未通过")

def displaytable(data):
    '''
    @ Description: 展示原表格中的基本数据
    @ data: 处理后的表格数据
    '''
    subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    rows = st.columns(len(subjects))  # 对每个课程，创建一个列
    results = []
    for grade in grades:
        for i, col in enumerate(rows):
            subject = subjects[i]
            col.header(f'{grade}{subject}课数据')
            temp_data = data[data["用户信息_学科"]==subject]
            col.dataframe(temp_data.head(5))
            col.metric("课量", len(temp_data))
            temp_data = temp_data.drop("用户信息_学科", axis=1)
            results.append(temp_data)
    return results

def getInfo():
    # TODO: 上传文件
    '''
    @ Description: 获取数据
    @ results: 删除值为0的列后的表格数据
    '''
    with st.status("数据分析中...", expanded=True):
        st.write("获取数据...")
        file_path = "data/初中语数外数据.csv"
        raw_data = pd.read_csv(file_path, encoding='gbk')
        st.dataframe(raw_data.head(5))
        st.metric("特征量", len(raw_data.columns.tolist()))
        # for col in data.columns.tolist():
        #     print(col)
        # 对此data删除某一列的值全为0的列
        # time.sleep(1)
        st.write("去除特征量的值全为0的列...")
        columns2del = []
        for col in raw_data.columns.tolist():
            # st.text(data[col].sum())
            if raw_data[col].sum() == 0:
                columns2del.append(col)
        st.info("去除属性"+str(columns2del))
        data = raw_data.drop(columns2del, axis=1)
        st.dataframe(data.head(5))
        st.metric("特征量", len(data.columns.tolist()), -int(len(columns2del)))
        # time.sleep(1)
        st.write("删除文本数据....")
        columns2del.extend(['用户信息_学段', '课程信息_年级', '占比', '抬头率_', '参与度_', '活跃度_', '一致性_'])
        st.info("去除属性"+str(columns2del))
        for column in data.columns.tolist():
            for col2del in columns2del:
                if col2del in column:
                    data = data.drop(column, axis=1)
        st.dataframe(data.head(5))
        st.metric("特征量", len(data.columns.tolist()), -int(len(columns2del)))
        st.write("展示基本信息....")
        # time.sleep(1)
        results = displaytable(data)
        return results

def main():
    st.title("平台使用方法")
    st.info("1. 平台首先会展示基本信息，包括用户信息、课程信息、课程数据等")
    st.info("2. 用户可以选择进行因子分析，分析课程数据中的因子")
    st.info("3. 因子分析若无法使用，用户可以选择PCA分析")
    st.info("4. 降维后，可以使用聚类分析查看聚类效果")
    page = st.sidebar.radio("选择分析类型",C.side_para[1]["names"])
    results = getInfo()
    if "pca_results" not in st.session_state:
        st.session_state.pca_results = []
    if page == "因子分析":
        st.title("因子分析")
        # time.sleep(1)
        factor_analysis(results)
    if page == "PCA降维":
        st.title("PCA分析")
        # 应当返回每个成分组成元素的列值 pca_results 
        pca_results = pca_analysis(results)
        st.session_state.pca_results = pca_results
    if page == "聚类分析":
        st.title("聚类分析")
        pca_results = st.session_state.pca_results
        if pca_results == []: 
            st.warning("请先进行PCA分析")
        else:
            cluster = st.selectbox("选择聚类算法: ", C.side_para[3]["clusters"])
            if cluster == "k-means":
                clustering_analysis_k_means(pca_results)
            if cluster == "DBSCAN":
                clustering_analysis_dbscan(pca_results)

if __name__ == '__main__':
    st.set_page_config(
    page_title="中庆画像", 
    page_icon=":chart_with_upwards_trend:", 
    layout="wide"
)
    main()