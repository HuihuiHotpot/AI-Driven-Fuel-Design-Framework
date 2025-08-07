import os
import pickle
import random
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from LDS import LDS
from LDSWeights import LDSWeights
from MultiTaskMLP import MultiTaskMLP
from PhysicsConLoss import PhysicsConLoss
from TabTransformer import TabTransformer
from UserDataset import UserDataset
from evaluate import evaluate

seed = 42

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

file_path = '6 folds.xlsx'
required_columns = ['负荷类型', '密度测试温度/K', '缸数', '径程比', '压缩比', '单缸排量/mm3', '燃烧室容积/mm3',
                    '转速/rpm', '活塞平均速度/(m/s)', '单缸循环供油量/mg', '喷油时刻（°CA BTDC）', 'EGR/%',
                    '单缸循环供能/J', '密度（kg/m3）', 'CN', 'LHV（MJ/kg）', '含氧量wt%', 'YSI-Das',
                    '进气方式', '进气温度/K', '进气压力/bar', '负荷/bar', '热效率']

sheet_names = pd.ExcelFile(file_path).sheet_names
folds = [pd.read_excel(file_path, sheet_name=sheet, keep_default_na=False)[required_columns].replace('', np.nan) for sheet in sheet_names]

fold_test = folds[-1]
train_folds = folds[:-1]

train_folds_combined = pd.concat(train_folds, axis=0)

columns_to_scale = ['径程比', '压缩比', '单缸排量/mm3', '燃烧室容积/mm3', '转速/rpm', '活塞平均速度/(m/s)',
                    '单缸循环供油量/mg', '喷油时刻（°CA BTDC）', 'EGR/%', '单缸循环供能/J', '密度（kg/m3）',
                    'CN', 'LHV（MJ/kg）', '含氧量wt%', 'YSI-Das', '进气温度/K', '进气压力/bar', '负荷/bar', '热效率']

columns_to_keep = [col for col in train_folds_combined.columns if col not in columns_to_scale]
print("需要归一化的列范围:", columns_to_scale)

scalers = {}
for column in columns_to_scale:
    scaler = MinMaxScaler()
    scaler.fit(train_folds_combined[[column]])  # 只在整个训练 folds 上 fit
    scalers[column] = scaler

scaler_save_path = 'saved_scalars_ensemble/scalers.pkl'
joblib.dump(scalers, scaler_save_path)

categorical_columns = ['负荷类型', '密度测试温度/K', '缸数', '进气方式']  # 替换为你的类别特征列名

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    train_folds_combined[col] = le.fit_transform(train_folds_combined[col])
    label_encoders[col] = le

with open('saved_encoders_ensemble/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

MEP_numpy = train_folds_combined["负荷/bar"]
Efficiency_numpy = train_folds_combined["热效率"]
MEP_hist, MEP_bin_centers, MEP_bin_edges, MEP_smoothed_hist = LDS(MEP_numpy, kernel_std=1.8838976821296132, n_bins=22)
Efficiency_hist, Efficiency_bin_centers, Efficiency_bin_edges, Efficiency_smoothed_hist = LDS(Efficiency_numpy, kernel_std=0.9173494080962382, n_bins=20)
lds_weights = LDSWeights()

for val_fold_idx in range(5):
    val_fold = train_folds[val_fold_idx]
    train_folds_cv = pd.concat([train_folds[i] for i in range(5) if i != val_fold_idx], axis=0)

    fold_train_scaled_df = pd.DataFrame(index=train_folds_cv.index)
    fold_val_scaled_df = pd.DataFrame(index=val_fold.index)

    for column in columns_to_scale:
        fold_train_scaled_df[column] = scalers[column].transform(train_folds_cv[[column]])
        fold_val_scaled_df[column] = scalers[column].transform(val_fold[[column]])

    columns_to_keep = [col for col in train_folds_cv.columns if col not in columns_to_scale]
    fold_train_final = pd.concat([train_folds_cv[columns_to_keep], fold_train_scaled_df], axis=1)
    fold_val_final = pd.concat([val_fold[columns_to_keep], fold_val_scaled_df], axis=1)

    for col in categorical_columns:
        fold_train_final[col] = label_encoders[col].transform(fold_train_final[col])
        fold_val_final[col] = label_encoders[col].transform(fold_val_final[col])

    fold_train_final = fold_train_final[train_folds_combined.columns]
    fold_val_final = fold_val_final[val_fold.columns]

    train_dataset = UserDataset(fold_train_final)
    val_dataset = UserDataset(fold_val_final)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=len(fold_val_final), shuffle=False)

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

    loss_basic = nn.MSELoss(reduction='none')

    loss_physics = PhysicsConLoss()

    optimizer = optim.Adam(
        list(model.parameters()) +
        list(ST_MLP.parameters()) +
        list(NA_MLP.parameters()) +
        list(ET_MLP.parameters()),
        lr=0.0004940951650795987
    )

    num_epochs = 180

    progress_bar = tqdm(range(num_epochs), desc=f"Fold {val_fold_idx + 1}/5",
                        leave=True)

    for epoch in range(num_epochs):
        model.train()
        ST_MLP.train()
        NA_MLP.train()
        ET_MLP.train()

        total_loss = 0
        total_samples = 0
        for (batch_category_features,
             batch_geometry_features,
             batch_operating_features,
             batch_fuel_features,
             batch_intake_mode,
             batch_intake_extra,
             batch_TrueMEP,
             batch_TrueEfficiency) in train_loader:

            optimizer.zero_grad()

            batch_continuous_features = torch.cat([
                batch_geometry_features,
                batch_operating_features,
                batch_fuel_features
            ], dim=1)

            batch_contextual_embeddings = model(batch_category_features, batch_continuous_features)

            current_batch_size = batch_category_features.shape[0]
            MEP_predict = torch.zeros(current_batch_size, 1).to('cuda')
            Efficiency_predict = torch.zeros(current_batch_size, 1).to('cuda')

            for i in range(current_batch_size):
                mode = batch_intake_mode[i].item()
                embedding = batch_contextual_embeddings[i]
                flattened_embedding = embedding.view(1, -1)
                extra = batch_intake_extra[i].unsqueeze(0)

                if mode == 2:  # ST
                    combined_features = torch.cat([flattened_embedding, extra], dim=-1)
                    MEP_predict[i], Efficiency_predict[i] = ST_MLP(combined_features)
                elif mode == 1:  # NA
                    combined_features = torch.cat([flattened_embedding], dim=-1)
                    MEP_predict[i], Efficiency_predict[i] = NA_MLP(combined_features)
                elif mode == 0:  # ET
                    combined_features = torch.cat([flattened_embedding], dim=-1)
                    MEP_predict[i], Efficiency_predict[i] = ET_MLP(combined_features)

            loss_MEP = loss_basic(MEP_predict, batch_TrueMEP.unsqueeze(1))
            loss_Efficiency = loss_basic(Efficiency_predict, batch_TrueEfficiency.unsqueeze(1))

            MEP_batch_weights = lds_weights(batch_TrueMEP, MEP_bin_edges, MEP_smoothed_hist)
            Efficiency_batch_weights = lds_weights(batch_TrueEfficiency, Efficiency_bin_edges,
                                                   Efficiency_smoothed_hist)

            loss_supervised = (loss_MEP * MEP_batch_weights).mean() + (
                    7.838193575296774 * loss_Efficiency * Efficiency_batch_weights).mean()

            热效率_min_tensor = torch.tensor(scalers["热效率"].data_min_, dtype=torch.float32,
                                             device=MEP_predict.device).unsqueeze(0)
            热效率_max_tensor = torch.tensor(scalers["热效率"].data_max_, dtype=torch.float32,
                                             device=MEP_predict.device).unsqueeze(0)

            负荷_min_tensor = torch.tensor(scalers["负荷/bar"].data_min_, dtype=torch.float32,
                                           device=MEP_predict.device).unsqueeze(0)
            负荷_max_tensor = torch.tensor(scalers["负荷/bar"].data_max_, dtype=torch.float32,
                                           device=MEP_predict.device).unsqueeze(0)

            单缸排量_min_tensor = torch.tensor(scalers["单缸排量/mm3"].data_min_, dtype=torch.float32,
                                               device=MEP_predict.device).unsqueeze(0)
            单缸排量_max_tensor = torch.tensor(scalers["单缸排量/mm3"].data_max_, dtype=torch.float32,
                                               device=MEP_predict.device).unsqueeze(0)

            单缸循环供能_min_tensor = torch.tensor(scalers["单缸循环供能/J"].data_min_, dtype=torch.float32,
                                                   device=MEP_predict.device).unsqueeze(0)
            单缸循环供能_max_tensor = torch.tensor(scalers["单缸循环供能/J"].data_max_, dtype=torch.float32,
                                                   device=MEP_predict.device).unsqueeze(0)

            MEP_original_tensor = MEP_predict * (负荷_max_tensor - 负荷_min_tensor) + 负荷_min_tensor  # [52, 1]

            Capacity_original_tensor = batch_geometry_features[:, 2].unsqueeze(1) * (
                    单缸排量_max_tensor - 单缸排量_min_tensor) + 单缸排量_min_tensor  # [52, 1]

            Qin_original_tensor = batch_operating_features[:, -1].unsqueeze(1) * (
                    单缸循环供能_max_tensor - 单缸循环供能_min_tensor) + 单缸循环供能_min_tensor  # [52, 1]

            CalculatedEfficiency_original_tensor = ((MEP_original_tensor * Capacity_original_tensor) / Qin_original_tensor) * 1e-4
            CalculatedEfficiency = (CalculatedEfficiency_original_tensor - 热效率_min_tensor) / (
                    热效率_max_tensor - 热效率_min_tensor)

            assert CalculatedEfficiency.requires_grad, "PredEfficiency 的计算未包含梯度信息！"

            loss_physics_value = loss_physics(CalculatedEfficiency, Efficiency_predict)

            batch_avg_loss = loss_supervised + 2.7959235669896216 * loss_physics_value

            batch_avg_loss.backward()
            optimizer.step()

            total_loss += batch_avg_loss.item() * current_batch_size
            total_samples += current_batch_size

        progress_bar.update(1)
        progress_bar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Fold": f"{val_fold_idx + 1}/5",
            "Loss": total_loss / total_samples
        })

    progress_bar.close()

    val_MEP_avg_loss, val_Efficiency_avg_loss, val_MEP_preds, val_MEP_true, val_TrueE, val_PredE = evaluate(
        val_loader, model, ST_MLP, NA_MLP, ET_MLP, scalers)

    r2_1 = r2_score(val_MEP_true, val_MEP_preds)
    r2_2 = r2_score(val_TrueE, val_PredE)

    print(f"Fold {val_fold_idx + 1} - R2_MEP: {r2_1:.4f}, R2_ThermalEfficiency: {r2_2:.4f}")

    model_save_path = f'saved_models_ensemble/fold{val_fold_idx + 1}.pth'
    torch.save({
        'TabTransformer': model.state_dict(),
        'ST_MLP': ST_MLP.state_dict(),
        'NA_MLP': NA_MLP.state_dict(),
        'ET_MLP': ET_MLP.state_dict(),
    }, model_save_path)