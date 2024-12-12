'''
Description: 用于中庆画像
Author: Weijia Ju
version: 
Date: 2024-11-18 22:25:34
LastEditors: Weijia Ju
LastEditTime: 2024-12-12 09:29:08
'''
import streamlit as st
import copy
import pandas as pd
import Config as C
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso

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

def draw_cluster(data, best_labels, best_center, fig, col):
    '''
    @ Description: 绘制聚类图
    @ data: 聚类数据
    @ best_labels: 最佳聚类标签
    @ best_center: 最佳聚类中心
    '''
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
        col.warning("维度大于等于6, 无法绘制聚类图")

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
def clustering_analysis_k_means(pca_results, subjects):
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
    # subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    for grade in grades:
        st.write(len(pca_results[grade]))
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
                    selected_data = st.session_state["classify_data"]
                    selected_data = selected_data.to_dict()["importance"]
                    for column in data.columns:
                        data[column] = data[column].multiply(selected_data[column])
                if scalerMethod == "MinMax":
                    # 最大最小标准化
                    scaler = MinMaxScaler()
                    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                    if not st.session_state["classify_data"].empty():
                        selected_data = st.session_state["classify_data"]
                        selected_data = selected_data.to_dict()["importance"]
                        for column in data.columns:
                            data[column] = data[column].multiply(selected_data[column])
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
                # col.info('最佳聚类中心')
                # col.dataframe(best_center)
                col.pyplot(fig_index)
                plt.close(fig_index)
                # 绘制聚类结果
                fig_cluster = plt.figure()
                draw_cluster(data, best_labels, best_center,fig_cluster,col)
                col.pyplot(fig_cluster)
                plt.close(fig_cluster)    
# DBSCAN
def clustering_analysis_dbscan(pca_results, subjects):
    '''
    @ Description: DBSCAN聚类
    @ pca_results: 处理后的表格数据列表
    '''
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score  # 轮廓系数
    # subjects = C.side_para[2]["subjects"] # 课程
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
                    # if not st.session_state["classify_data"].empty():
                    selected_data = st.session_state["classify_data"]
                    selected_data = selected_data.to_dict()["importance"]
                    for column in data.columns:
                        data[column] = data[column].multiply(selected_data[column])
                if scalerMethod == "MinMax":
                    # 最大最小标准化
                    scaler = MinMaxScaler()
                    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                    # if not st.session_state["classify_data"].empty():
                    selected_data = st.session_state["classify_data"]
                    selected_data = selected_data.to_dict()["importance"]
                    for column in data.columns:
                        data[column] = data[column].multiply(selected_data[column])
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
                draw_cluster(data, labels, best_centers, fig_cluster,col)
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

def pca_analysis(data, subjects):
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
    # subjects = C.side_para[2]["subjects"] # 课程
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
                # if not st.session_state["classify_data"].empty():
                selected_data = st.session_state["classify_data"]
                selected_data = selected_data.to_dict()["importance"]
                for column in temp_data.columns:
                    temp_data[column] = temp_data[column].multiply(selected_data[column])
            if scalerMethod == "MinMax":
                # 最大最小标准化
                scaler = MinMaxScaler()
                numeric_columns = temp_data.select_dtypes(include=['float64', 'int64']).columns
                temp_data[numeric_columns] = scaler.fit_transform(temp_data[numeric_columns])
                # if not st.session_state["classify_data"].empty():
                selected_data = st.session_state["classify_data"]
                selected_data = selected_data.to_dict()["importance"]
                for column in temp_data.columns:
                    temp_data[column] = temp_data[column].multiply(selected_data[column])
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
                for p in pca_expla:
                    t += p
                    temp_pca.append(t)
                if sum(pca_expla) > pca_expla_theta:
                    flag = False
            # 将list转为array
            pca_data = []
            pca_data.append(pca_expla)
            pca_data.append(temp_pca)
            pca_explas = pd.DataFrame(np.array(pca_data).transpose(), columns=["方差贡献率","累计贡献率"])
            col.info("PCA累计贡献率")
            col.dataframe(pca_explas)
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
                # 降序排列
                temp_loadings = temp_loadings.sort_values(ascending=False)
                col.table(temp_loadings)
                # col.dataframe(temp_loadings)
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

def factor_analysis(data, subjects):
    # TODO: 因子分析通过后确定因子个数
    '''
    @ Description: 因子分析流程
    @ data: 处理后的表格数据列表
    '''
    rows = st.columns(len(data))
    # subjects = C.side_para[2]["subjects"] # 课程
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
                # if not st.session_state["classify_data"].empty():
                selected_data = st.session_state["classify_data"]
                selected_data = selected_data.to_dict()["importance"]
                for column in temp_data.columns:
                    temp_data[column] = temp_data[column].multiply(selected_data[column])
            if scalerMethod == "MinMax":
                # 最大最小标准化
                scaler = MinMaxScaler()
                numeric_columns = temp_data.select_dtypes(include=['float64', 'int64']).columns
                temp_data[numeric_columns] = scaler.fit_transform(temp_data[numeric_columns])
                # if not st.session_state["classify_data"].empty():
                selected_data = st.session_state["classify_data"]
                selected_data = selected_data.to_dict()["importance"]
                for column in temp_data.columns:
                    temp_data[column] = temp_data[column].multiply(selected_data[column])
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

def displaytable(data, subjects):
    '''
    @ Description: 展示原表格中的基本数据
    @ data: 处理后的表格数据
    '''
    # subjects = C.side_para[2]["subjects"] # 课程
    grades = C.side_para[2]["grades"] # 年级
    rows = st.columns(len(subjects))  # 对每个课程，创建一个列
    results = []
    for grade in grades:
        for i, col in enumerate(rows):
            subject = subjects[i]
            # subject_id = C.side_para[4][subject]
            col.header(f'{grade}{subject}课数据')
            temp_data = data[data["用户信息_学科"]==subject]
            col.metric("课量", len(temp_data))
            temp_data = temp_data.drop("用户信息_学科", axis=1)
            results.append(temp_data)
            col.dataframe(temp_data.head(3))
    return results

def displayCorr(corr):
    '''
    @ Description: 展示相关系数矩阵
    '''
    font = FontProperties(fname=r"SimHei.ttf", size=10)
    plt.imshow(corr)
    plt.title("相关系数矩阵", fontproperties=font)
    plt.title("相关性热力图", fontproperties=font)

def selectfeature(data):
    skb = SelectKBest(f_classif, k=len(data.columns)-1)
    skb.fit(data, data['用户信息_学科'])
    scores = np.log10(skb.scores_)
    score_data = pd.DataFrame(scores, index=data.columns, columns=["score"])
    score_data = score_data.sort_values(by='score', ascending=False)
    # 去除值为空的列
    # score_data = score_data[score_data['score'] > 0]
    st.bar_chart(score_data)
    return score_data

def myClassifier(data, classifier, n_feature, threshold):
    '''
    @ Description: 分类器
    '''
    if classifier == "随机森林":
        classifier = RandomForestClassifier(n_estimators=100, max_features=n_feature, random_state=0)
    # elif classifier == "支持向量机":
    #     classifier = svm.SVC(kernel='sigmoid', C=1)
    # elif classifier == "K近邻":
    #     classifier = KNeighborsClassifier(n_neighbors=3)
    # elif classifier == "朴素贝叶斯":
    #     classifier = nb.GaussianNB()
    X = data.drop("用户信息_学科", axis=1)
    y = data["用户信息_学科"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    # 特征重要性
    importances = classifier.feature_importances_
    # 特征量和重要性的值组成pandas的dataframe
    classify_data = pd.DataFrame(importances, index=X_train.columns, columns=["importance"])
    # selected_data = classify_data[classify_data['importance'] > threshold]
    # selected_feature = selected_data.index.tolist()
    return accuracy, classify_data

def getInfo(subjects):
    '''
    @ Description: 获取数据
    @ results: 删除值为0的列后的表格数据
    '''
    with st.status("数据分析中...", expanded=True):
        st.html("<h1 style='text-align: center'>----------------获    取    数    据---------------</h1>")
        file = st.file_uploader("", type=["csv"])
        if file is None:
            st.warning("请先上传一个csv文件")
        else:
            feature_list = []  # 特征量列表
            feature_delta_list = [] # 特征量变化列表
            raw_data = pd.read_csv(file, encoding='gbk')
            st.dataframe(raw_data.head(3))
            # 初始特征量
            feature_list.append(len(raw_data.columns.tolist()))
            feature_delta_list.append(0)
            columns2del = []
            for col in raw_data.columns.tolist():
                # st.text(data[col].sum())
                if raw_data[col].sum() == 0:
                    columns2del.append(col)
            st.info("去除特征量的值全为0的列..."+str(columns2del))
            data = raw_data.drop(columns2del, axis=1)
            st.dataframe(data.head(3))
            # 增加第二次特征量
            feature_list.append(len(data.columns.tolist()))
            feature_delta_list.append(-int(len(columns2del)))
            columns2del.extend(['用户信息_学段', '课程信息_年级', '占比', '抬头率_', '参与度_', '活跃度_', '一致性_'])
            st.info("删除文本数据...."+str(columns2del))
            for column in data.columns.tolist():
                for col2del in columns2del:
                    if col2del in column:
                        data = data.drop(column, axis=1)
            st.dataframe(data.head(3))
            # 增加第三次特征量
            feature_list.append(len(data.columns.tolist()))
            feature_delta_list.append(-int(len(columns2del)))
            corr_on = st.toggle("相关性分析")
            if corr_on:
                # 相关性分析，去除高相关的特征
                corr_theta = st.slider("选择去除相关系数的阈值", 0.0, 1.0, 0.8)
                corr_data = copy.deepcopy(data) # 用于做相关性分析
                corr_data = corr_data.drop("用户信息_学科", axis=1)
                # 去除方差非常小的特征
                var_thresh = VarianceThreshold(threshold=0.001)
                var_thresh.fit(corr_data)
                var_thresh.transform(corr_data)
                columns_remained = var_thresh.get_feature_names_out()
                columns_var_min = list(set(corr_data.columns)-set(columns_remained))
                columns2del.extend(columns_var_min)
                st.info(f'去除方差较小的特征...{columns_var_min}')
                # 增加第四次特征量
                feature_list.append(len(corr_data.columns.tolist()))
                feature_delta_list.append(-int(len(columns2del)))
                corr_matric = corr_data.corr()
                st.dataframe(corr_matric)
                corr_on = st.toggle("是否展示热力图")
                if corr_on:
                    fig_corr = plt.figure()
                    displayCorr(corr_matric)
                    st.pyplot(fig_corr)
                # 取出相关系数大于阈值的特征
                corr_matric = corr_matric[corr_matric > corr_theta]
                st.dataframe(corr_matric)
                columns2del.extend(corr_matric.columns.tolist())
            classify_on = st.toggle("训练学科分类器")
            if classify_on:
                st.html("<h1 style='text-align: center'>------------对  语  数  外  打  标  签------------</h1>")
                data.replace(C.side_para[4], inplace = True)
                st.dataframe(data.head(3))
                # 去除值为空的列
                # feature_on = st.toggle("训练分类器以查看最佳特征数")
                max_accuracy = 0
                best_features = []
                if "classify_data" not in st.session_state: # 经过分类筛选后的特征维度及其重要性
                    st.session_state["classify_data"] = []
                if "max_accuracy" not in st.session_state:
                    st.session_state["max_accuracy"] = max_accuracy
                if "accuracy_list" not in st.session_state:
                    st.session_state["accuracy_list"] = []
                # if feature_on:
                threshold = st.slider("选择阈值", 0.0, 1.0, 0.25)
                threshold = threshold / 10
                st.html("<h1 style='text-align: center'>----------------训  练  分  类  器---------------</h1>")
                if st.session_state["max_accuracy"] == 0:
                    my_bar = st.progress(0, "Classifying...")
                    classify = st.selectbox("选择分类器", C.side_para[5]["classifier"])
                    accuracy_list = []
                    for i in range(5, 30):
                        text = "Test " + str(i) + "th classify feature nums..."
                        my_bar.progress(i/30, text)
                        accuracy, selected_data = myClassifier(data, classify, i, threshold)
                        accuracy_list.append(accuracy)
                        if accuracy > max_accuracy:
                            max_accuracy = accuracy
                            st.session_state["classify_data"] = selected_data
                            st.session_state["max_accuracy"] = max_accuracy
                            st.session_state["accuracy_list"] = accuracy_list
                # st.metric("最佳特征数量", n_feature)
                max_accuracy = st.session_state["max_accuracy"]
                accuracy_list = st.session_state["accuracy_list"]
                selected_data = st.session_state["classify_data"]
                st.info("最佳训练分类器的特征")
                best_feature_on = st.toggle("展示最佳特征")
                if best_feature_on:
                    # 降序排序
                    selected_data_display = selected_data.sort_values(by='importance', ascending=False)
                    st.table(selected_data_display)
                    # st.json(selected_data.to_dict())
                st.metric("最佳准确率", max_accuracy)
                st.bar_chart(accuracy_list)
                st.html("<h1 style='text-align: center'>----------------选  择  特  征  拟  合---------------</h1>")
                # 从分类器中选择合适的特征
                selected_data = selected_data[selected_data['importance'] > threshold]
                best_features = selected_data.index.tolist()
                columns = set(data.columns)
                data = data[best_features]
                new_columns = set(data.columns)
                columns2del.extend(list(columns-new_columns))
                st.info("去除特征..."+str(columns2del))
                # st.dataframe(data) # 根据阈值选择的特征
                feature_list.append(len(data.columns.tolist()))
                feature_delta_list.append(-int(len(columns2del)))
                 # 从raw_data中向data增加用户信息_学科这一列数据
                data = data.merge(raw_data[["用户信息_学科"]], left_index=True, right_index=True)
            st.html("<h1 style='text-align: center'>----------------特  征  量  变  化  趋  势---------------</h1>")
            st.bar_chart(feature_list)
            # st.dataframe(data)
            if subjects == []:
                st.warning("请先选择课程")
            else:
                st.html("<h1 style='text-align: center'>----------------展  示  基  本  信  息---------------</h1>")
                results = displaytable(data, subjects)
                return results
            # score_data = selectfeature(data)
            # columns2del.extend(score_data.index[n_feature+1:])
            # st.info("去除特征..."+str(columns2del))
            # data = data[col_remains]
            # accuracy = myClassifier(data, classify, n_feature)
            # st.metric("分类准确率", accuracy)
            # 增加第四次特征量
            # st.dataframe(score_data)

def classifyclass(data, subjects):
    st.info("选择分类器")
    classifier = st.selectbox("选择分类器", C.side_para[5]["classifier"])
    temp_data = data[0]
    labels = st.multiselect("选择标签", temp_data.columns.tolist(), default=["ST分析_师生行为转换率"])
    rows = st.columns(len(data))
    grades = C.side_para[2]["grades"] # 年级
    for grade in grades:
        for i, col in enumerate(rows):
            subject = subjects[i]
            col.header(f'{grade}{subject}课程分类')
            temp_data = data[i]
            X = temp_data.drop(labels, axis=1)
            Y = temp_data[labels]
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            if classifier == "随机森林":
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("分类准确率", accuracy)
                

def main():
    st.title("平台使用方法")
    st.info("1. 平台首先会展示基本信息，包括用户信息、课程信息、课程数据等")
    st.info("2. 用户可以选择进行因子分析，分析课程数据中的因子")
    st.info("3. 因子分析若无法使用，用户可以选择PCA分析")
    st.info("4. 降维后，可以使用聚类分析查看聚类效果")
    subjects = st.multiselect("选择课程", C.side_para[2]["subjects"], default=C.side_para[2]["subjects"][0])
    page = st.sidebar.radio("分析步骤",C.side_para[1]["names"])
    results = getInfo(subjects)
    if "pca_results" not in st.session_state:
        st.session_state.pca_results = []
    if page == "因子分析":
        st.title("因子分析")
        # time.sleep(1)
        factor_analysis(results, subjects)
    if page == "PCA降维":
        st.title("PCA分析")
        # 应当返回每个成分组成元素的列值 pca_results 
        pca_results = pca_analysis(results, subjects)
        st.session_state.pca_results = pca_results
    if page == "聚类分析":
        st.title("聚类分析")
        pca_results = st.session_state.pca_results
        if pca_results == []: 
            st.warning("请先进行PCA分析")
        else:
            cluster = st.selectbox("选择聚类算法: ", C.side_para[3]["clusters"])
            if cluster == "k-means":
                clustering_analysis_k_means(pca_results, subjects)
            if cluster == "DBSCAN":
                clustering_analysis_dbscan(pca_results, subjects)
    if page == "分类分析":
        st.title("分类分析")
        classifyclass(results, subjects) # 对每一种课程类型，使用某一标签进行分类

if __name__ == '__main__':
    st.set_page_config(
    page_title="中庆画像", 
    page_icon=":chart_with_upwards_trend:", 
    layout="wide"
)
    main()