import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'Dataset.xlsx'
sheet_name = 'Dataset'
data = pd.read_excel(file_path, sheet_name=sheet_name, keep_default_na=False)

target_column1 = '负荷/bar'
target_column2 = '热效率'
categorical_features = ['进气方式', '负荷类型', '缸数', '密度测试温度/K']
continuous_features = ['压缩比', '径程比', '单缸排量/mm3', "燃烧室容积/mm3", "活塞平均速度/(m/s)", '转速/rpm', '单缸循环供油量/mg',
                       '喷油时刻（°CA BTDC）', 'EGR/%', '密度（kg/m3）', 'CN', 'LHV（MJ/kg）', "含氧量wt%", "YSI-Das"]
additional_features = ['进气温度/K', '进气压力/bar']

X_cont = data[continuous_features]
X_cat = data[categorical_features]
X_add = data[additional_features].replace("-", np.nan)
y1 = data[target_column1]
y2 = data[target_column2]


encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_cat), columns=encoder.get_feature_names_out())

X_combined = pd.concat([X_cont.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=12)
data['initial_cluster'] = kmeans.fit_predict(X_combined)

final_clusters = []

for cluster_id in data['initial_cluster'].unique():
    cluster_data = data[data['initial_cluster'] == cluster_id]
    cluster_X_combined = X_combined.loc[cluster_data.index]

    if not X_add.loc[cluster_data.index].isnull().all().all():
        refined_X = pd.concat([cluster_X_combined, X_add.loc[cluster_data.index].fillna(0)], axis=1)
        kmeans_refined = KMeans(n_clusters=2, random_state=12)
        sub_clusters = kmeans_refined.fit_predict(refined_X)
        cluster_data = cluster_data.copy()
        cluster_data['final_cluster'] = cluster_data['initial_cluster'].astype(str) + '_' + sub_clusters.astype(str)
    else:
        cluster_data['final_cluster'] = f"{cluster_id}_base"

    final_clusters.append(cluster_data)

data = pd.concat(final_clusters)

if pd.api.types.is_numeric_dtype(data[target_column1]):
    data['label1'] = pd.qcut(data[target_column1], q=4, labels=['低', '中低', '中高', '高'])
else:
    data['label1'] = data[target_column1].astype(str)

if pd.api.types.is_numeric_dtype(data[target_column2]):
    data['label2'] = pd.qcut(data[target_column2], q=4, labels=['低', '中低', '中高', '高'])
else:
    data['label2'] = data[target_column2].astype(str)

data['combined_label'] = data['label1'].astype(str) + '_' + data['label2'].astype(str)

n_folds = 6
data['fold'] = -1

for cluster in data['final_cluster'].unique():
    cluster_data = data[data['final_cluster'] == cluster].copy()

    if len(cluster_data) < n_folds:
        print(f"Cluster {cluster} 样本数量少于 {n_folds}，将所有样本分配到一个折。")
        data.loc[cluster_data.index, 'fold'] = 1
        continue

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_number, (_, test_index) in enumerate(skf.split(cluster_data, cluster_data['combined_label']), 1):
        fold_indices = cluster_data.iloc[test_index].index
        data.loc[fold_indices, 'fold'] = fold_number

unassigned = data[data['fold'] == -1]
if not unassigned.empty:
    print("存在未分配折的样本，将这些样本随机分配到1到6的折中。")
    data.loc[unassigned.index, 'fold'] = np.random.randint(1, n_folds + 1, size=len(unassigned))

assert (data['fold'] != -1).all(), "存在未分配折的样本"

print("\n=== 全局组合标签分布 ===")
print(data['combined_label'].value_counts(normalize=True))

print("\n=== 各折组合标签分布 ===")
fold_combined_label_dist = data.groupby('fold')['combined_label'].value_counts(normalize=True).unstack()
print(fold_combined_label_dist)

plt.figure(figsize=(12, 8))
bar_width = 0.15
labels = fold_combined_label_dist.columns
x = np.arange(len(labels))

for i, fold in enumerate(sorted(data['fold'].unique())):
    plt.bar(x + i * bar_width, fold_combined_label_dist.loc[fold], width=bar_width, label=f'Fold {fold}')

global_dist = data['combined_label'].value_counts(normalize=True).sort_index()
plt.plot(x + (len(data['fold'].unique()) / 2) * bar_width, global_dist, color='black', marker='o', linestyle='--', label='全局分布')

plt.title('各折的组合标签分布与全局分布对比')
plt.xlabel('组合标签')
plt.ylabel('比例')
plt.xticks(x + (len(data['fold'].unique()) / 2) * bar_width, labels, rotation=45)
plt.legend()
plt.tight_layout()
# plt.show()

continuous_labels = ['负荷/bar', '热效率']

for label in continuous_labels:
    plt.figure(figsize=(12, 8))

    sns.kdeplot(data[label], label='全局分布', color='black', lw=2)

    for fold in sorted(data['fold'].unique()):
        fold_data = data.loc[data['fold'] == fold, label]
        sns.kdeplot(fold_data, label=f'Fold {fold}', fill=True, alpha=0.1)

    plt.title(f'各折的真实标签 "{label}" 概率密度分布与全局分布对比')
    plt.xlabel(label)
    plt.ylabel('概率密度')
    plt.legend()
    plt.tight_layout()
    # plt.show()

selected_features = ['压缩比', '径程比', '单缸排量/mm3', "燃烧室容积/mm3", "活塞平均速度/(m/s)", '转速/rpm', '单缸循环供油量/mg',
                       '喷油时刻（°CA BTDC）', 'EGR/%', '密度（kg/m3）', 'CN', 'LHV（MJ/kg）', "含氧量wt%", "YSI-Das"]

for feature in selected_features:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=data, x=feature, label='全局', color='black', lw=2)

    for fold in sorted(data['fold'].unique()):
        fold_data = data[data['fold'] == fold]
        sns.kdeplot(data=fold_data, x=feature, label=f'Fold {fold}', fill=True, alpha=0.1)
    plt.title(f'特征 "{feature}" 在各折中的分布与全局分布对比')
    plt.xlabel(feature)
    plt.ylabel('概率密度')
    plt.legend()
    plt.tight_layout()
    # plt.show()

selected_cat_features = ['进气方式', '负荷类型', '热效率类型', '缸数', '密度测试温度/K']
bar_width = 0.15

for feature in selected_cat_features:
    plt.figure(figsize=(12, 6))

    global_dist = data[feature].value_counts(normalize=True).sort_index()
    labels = global_dist.index
    x = np.arange(len(labels))

    fold_label_dist = data.groupby('fold')[feature].value_counts(normalize=True).unstack().fillna(0)
    for i, fold in enumerate(sorted(data['fold'].unique())):
        plt.bar(x + i * bar_width, fold_label_dist.loc[fold], width=bar_width, label=f'Fold {fold}')

    plt.plot(x + (len(data['fold'].unique()) / 2) * bar_width, global_dist, color='black', marker='o', linestyle='--', lw=2, label='全局分布')

    plt.title(f'类别特征 "{feature}" 在各折中的分布与全局分布对比')
    plt.xlabel(feature)
    plt.ylabel('比例')
    plt.xticks(x + (len(data['fold'].unique()) / 2) * bar_width, labels)
    plt.legend()
    plt.tight_layout()
    # plt.show()

additional_features = ['进气温度/K', '进气压力/bar']

data['进气温度/K'] = pd.to_numeric(data['进气温度/K'], errors='coerce')
data['进气压力/bar'] = pd.to_numeric(data['进气压力/bar'], errors='coerce')

for feature in additional_features:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=data, x=feature, label='全局', color='black', lw=2)

    for fold in sorted(data['fold'].unique()):
        fold_data = data[data['fold'] == fold]
        sns.kdeplot(data=fold_data, x=feature, label=f'Fold {fold}', fill=True, alpha=0.1)
    plt.title(f'附加特征 "{feature}" 在各折中的分布与全局分布对比')
    plt.xlabel(feature)
    plt.ylabel('概率密度')
    plt.legend()
    plt.tight_layout()
    # plt.show()

plt.show()

output_path = 'result.xlsx'

data.to_excel(output_path, index=False)
