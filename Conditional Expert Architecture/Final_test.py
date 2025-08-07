import numbers
import os
import pickle
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from TabTransformer import TabTransformer
from MultiTaskMLP import MultiTaskMLP
from UserDataset import UserDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = 'saved_models_ensemble'
scaler_path = 'saved_scalars_ensemble/scalers.pkl'
encoder_path = 'saved_encoders_ensemble/label_encoders.pkl'

file_path = '6 folds.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

required_columns = ['负荷类型', '密度测试温度/K', '缸数', '径程比', '压缩比', '单缸排量/mm3', '燃烧室容积/mm3',
                    '转速/rpm', '活塞平均速度/(m/s)', '单缸循环供油量/mg', '喷油时刻（°CA BTDC）', 'EGR/%',
                    '单缸循环供能/J', '密度（kg/m3）', 'CN', 'LHV（MJ/kg）', '含氧量wt%', 'YSI-Das',
                    '进气方式', '进气温度/K', '进气压力/bar', '负荷/bar', '热效率']

columns_to_scale = ['径程比', '压缩比', '单缸排量/mm3', '燃烧室容积/mm3', '转速/rpm', '活塞平均速度/(m/s)',
                    '单缸循环供油量/mg', '喷油时刻（°CA BTDC）', 'EGR/%', '单缸循环供能/J', '密度（kg/m3）',
                    'CN', 'LHV（MJ/kg）', '含氧量wt%', 'YSI-Das', '进气温度/K', '进气压力/bar', '负荷/bar', '热效率']

# 指定类别变量的列
categorical_columns = ['负荷类型', '密度测试温度/K', '缸数', '进气方式']

# 加载scaler和encoder
scalers = joblib.load(scaler_path)
with open(encoder_path, 'rb') as f:
    label_encoders = pickle.load(f)

# 加载测试集
test_df = pd.read_excel(file_path, sheet_name=sheet_names[-1], keep_default_na=False)[required_columns].replace('', np.nan)

# 编码分类变量
for col in categorical_columns:
    le = label_encoders[col]
    test_df[col] = le.transform(test_df[col])

# 归一化
scaled_df = test_df.copy()
for col in columns_to_scale:
    scaled_df[col] = scalers[col].transform(test_df[[col]])

# 构建 DataLoader
test_dataset = UserDataset(scaled_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def load_model_fold(fold_idx):
    # TabTransformer
    num_categories = [2, 2, 3]
    category_embedding_dim = 12
    d_model = 64
    nhead = 4
    num_transformer_layers = 3
    dropout = 0.05007310142201917

    model = TabTransformer(
        num_categories=num_categories,
        category_embedding_dim=category_embedding_dim,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout
    ).to('cuda')

    # MLP
    extra_features_dim = 2
    category_features_dim = 3
    continuous_features_dim = 15

    ST_MLP = MultiTaskMLP(
        input_size=d_model * (category_features_dim + continuous_features_dim) + extra_features_dim,
        shared_layer_sizes=[128, 64],
        task1_sizes=[128, 64],
        task2_sizes=[32, 16],
        output_sizes=(1, 1)
    ).to('cuda')
    NA_MLP = MultiTaskMLP(
        input_size=d_model * (category_features_dim + continuous_features_dim),
        shared_layer_sizes=[64, 32],
        task1_sizes=[64, 32],
        task2_sizes=[32, 16],
        output_sizes=(1, 1)
    ).to('cuda')
    ET_MLP = MultiTaskMLP(
        input_size=d_model * (category_features_dim + continuous_features_dim),
        shared_layer_sizes=[128, 64],
        task1_sizes=[128, 64],
        task2_sizes=[128, 64],
        output_sizes=(1, 1)
    ).to('cuda')

    # 加载权重
    model_path = os.path.join(model_dir, f'fold{fold_idx + 1}.pth')
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['TabTransformer'])
    ST_MLP.load_state_dict(checkpoint['ST_MLP'])
    NA_MLP.load_state_dict(checkpoint['NA_MLP'])
    ET_MLP.load_state_dict(checkpoint['ET_MLP'])

    model.eval(), ST_MLP.eval(), NA_MLP.eval(), ET_MLP.eval()
    return model, ST_MLP, NA_MLP, ET_MLP

def calculate_physics_bias(MEP, Q_in, V_disp, Efficiency_pred):
    efficiency_calc = (MEP * V_disp / Q_in) * 1e-4
    return abs(efficiency_calc - Efficiency_pred)

models = [load_model_fold(i) for i in range(5)]

final_preds_mep, final_preds_eff = [], []
true_mep, true_eff = [], []

record_rows = []

for i, (cat, geo, op, fuel, mode, extra, true_m, true_e) in enumerate(test_loader):
    biases, mep_preds, eff_preds = [], [], []

    row_record = {
        'Sample Index': i,
        'True MEP': test_df.iloc[i]["负荷/bar"],
        'True Efficiency': test_df.iloc[i]["热效率"]
    }

    for fold_idx, (model, ST, NA, ET) in enumerate(models):
        with torch.no_grad():
            cont = torch.cat([geo, op, fuel], dim=1).to(device)
            embed = model(cat.to(device), cont)

            mode_id = mode.item()
            extra = extra.to(device)
            if mode_id == 2:
                feat = torch.cat([embed.view(1, -1), extra], dim=-1)
                mep, eff = ST(feat)
            elif mode_id == 1:
                mep, eff = NA(embed.view(1, -1))
            else:
                mep, eff = ET(embed.view(1, -1))

            mep = mep.cpu().numpy()[0][0]
            eff = eff.cpu().numpy()[0][0]

            mep_original = mep * (scalers['负荷/bar'].data_max_ - scalers['负荷/bar'].data_min_) + scalers['负荷/bar'].data_min_
            eff_original = eff * (scalers['热效率'].data_max_ - scalers['热效率'].data_min_) + scalers['热效率'].data_min_
            V_disp = geo[:, 2].item() * (scalers["单缸排量/mm3"].data_max_ - scalers["单缸排量/mm3"].data_min_) + scalers["单缸排量/mm3"].data_min_
            Q_in = op[:, -1].item() * (scalers["单缸循环供能/J"].data_max_ - scalers["单缸循环供能/J"].data_min_) + scalers["单缸循环供能/J"].data_min_

            bias = calculate_physics_bias(mep_original, Q_in, V_disp, eff_original)

            row_record[f'Fold{fold_idx + 1}_MEP'] = mep_original
            row_record[f'Fold{fold_idx + 1}_Efficiency'] = eff_original
            row_record[f'Fold{fold_idx + 1}_Bias'] = bias

            mep_preds.append(mep_original)
            eff_preds.append(eff_original)
            biases.append(bias)

    best_idx = np.argmin(biases)
    row_record['Selected Fold'] = best_idx + 1
    row_record['Selected MEP'] = mep_preds[best_idx]
    row_record['Selected Efficiency'] = eff_preds[best_idx]
    row_record['Selected Bias'] = biases[best_idx]
    row_record['Selected MEP Residual'] = mep_preds[best_idx] - row_record['True MEP']
    row_record['Selected Efficiency Residual'] = eff_preds[best_idx] - row_record['True Efficiency']

    final_preds_mep.append(mep_preds[best_idx])
    final_preds_eff.append(eff_preds[best_idx])
    true_mep.append(test_df.iloc[i]["负荷/bar"])
    true_eff.append(test_df.iloc[i]["热效率"])

    record_rows.append(row_record)

def extract_scalar(x):
    if isinstance(x, (list, np.ndarray)) and len(x) == 1:
        return x[0]
    elif isinstance(x, np.generic):
        return x.item()
    elif isinstance(x, numbers.Number):
        return x
    else:
        return x

record_df = pd.DataFrame(record_rows)

record_df = record_df.applymap(extract_scalar)

record_df.to_csv("ensemble_prediction_with_physics_bias.csv", index=False)

def mean_deviation(y_true, y_pred):
    return np.mean(y_pred - y_true)

def mean_relative_error(y_true, y_pred):
    return np.mean((y_pred - y_true) / y_true) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

true_mep = np.array(true_mep)
final_preds_mep = np.array(final_preds_mep).flatten()
true_eff = np.array(true_eff)
final_preds_eff = np.array(final_preds_eff).flatten()

# MEP
mse_mep = mean_squared_error(true_mep, final_preds_mep)
r2_mep = r2_score(true_mep, final_preds_mep)
plt.figure(figsize=(6, 6))
plt.scatter(true_mep, final_preds_mep, alpha=0.5)
plt.plot([min(true_mep), max(true_mep)], [min(true_mep), max(true_mep)], 'r--')
plt.title(f"MEP - Test Set\nMSE: {mse_mep:.4f}, R²: {r2_mep:.4f}")
plt.xlabel("True MEP")
plt.ylabel("Predicted MEP")
plt.grid(True)
plt.show()

# Efficiency
mse_eff = mean_squared_error(true_eff, final_preds_eff)
r2_eff = r2_score(true_eff, final_preds_eff)
plt.figure(figsize=(6, 6))
plt.scatter(true_eff, final_preds_eff, alpha=0.5)
plt.plot([min(true_eff), max(true_eff)], [min(true_eff), max(true_eff)], 'r--')
plt.title(f"Thermal Efficiency - Test Set\nMSE: {mse_eff:.6f}, R²: {r2_eff:.4f}")
plt.xlabel("True Efficiency")
plt.ylabel("Predicted Efficiency")
plt.grid(True)
plt.show()