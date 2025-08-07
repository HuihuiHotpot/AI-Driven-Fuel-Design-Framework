import warnings
import optuna
import optuna.visualization as vis
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.pruners import MedianPruner
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from LDS import LDS
from LDSWeights import LDSWeights
from PhysicsConLoss import PhysicsConLoss
from TabTransformer import TabTransformer
from MultiTaskMLP import MultiTaskMLP
from UserDataset import UserDataset
from evaluate import evaluate

warnings.filterwarnings("ignore")

file_path = '6 folds.xlsx'

required_columns = ['负荷类型', '密度测试温度/K', '缸数',
                    '径程比', '压缩比', "单缸排量/mm3", '燃烧室容积/mm3',
                    '转速/rpm', "活塞平均速度/(m/s)", '单缸循环供油量/mg','喷油时刻（°CA BTDC）','EGR/%', "单缸循环供能/J",
                    '密度（kg/m3）', 'CN', 'LHV（MJ/kg）', "含氧量wt%", "YSI-Das",
                    '进气方式', '进气温度/K', '进气压力/bar',
                    '负荷/bar',
                    '热效率']

sheet_names = pd.ExcelFile(file_path).sheet_names

folds = [
    pd.read_excel(file_path, sheet_name=sheet, keep_default_na=False)[required_columns].replace('', np.nan)
    for sheet in sheet_names
]

fold_test = folds[-1]
train_folds = folds[:-1]

n_trials = 30

def objective(trial):
    seed = 42

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    kernel_std_1 = trial.suggest_uniform('kernel_std_1', 1.0, 2.0)
    n_bins_1 = trial.suggest_int('n_bins_1', 12, 24, step=2)
    kernel_std_2 = trial.suggest_uniform('kernel_std_2', 0.8, 1.5)
    n_bins_2 = trial.suggest_int('n_bins_2', 10, 20, step=2)

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 7e-4)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 150, 200, step=10)
    dropout = trial.suggest_uniform('dropout', 0.05, 0.2)

    num_transformer_layers = trial.suggest_int('num_transformer_layers', 2, 4, step=1)
    transformer_d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    transformer_nhead = trial.suggest_categorical('nhead', [4, 8])
    category_embedding_dim = trial.suggest_int('category_embedding_dim', 8, 16, step=4)

    ST_shared_layer_sizes = trial.suggest_categorical('ST_shared_layer_sizes',[[64, 32], [128, 64], [256, 128]])
    ST_task1_layer_sizes = trial.suggest_categorical('ST_task1_layer_sizes', [[32, 16], [64, 32], [128, 64]])
    ST_task2_layer_sizes = trial.suggest_categorical('ST_task2_layer_sizes', [[32, 16], [64, 32], [128, 64]])

    ET_shared_layer_sizes = trial.suggest_categorical('ET_shared_layer_sizes', [[64, 32], [128, 64], [256, 128]])
    ET_task1_layer_sizes = trial.suggest_categorical('ET_task1_layer_sizes', [[32, 16], [64, 32], [128, 64]])
    ET_task2_layer_sizes = trial.suggest_categorical('ET_task2_layer_sizes', [[32, 16], [64, 32], [128, 64]])

    NA_shared_layer_sizes = trial.suggest_categorical('NA_shared_layer_sizes', [[64, 32], [128, 64], [256, 128]])
    NA_task1_layer_sizes = trial.suggest_categorical('NA_task1_layer_sizes', [[32, 16], [64, 32], [128, 64]])
    NA_task2_layer_sizes = trial.suggest_categorical('NA_task2_layer_sizes', [[32, 16], [64, 32], [128, 64]])

    physics_loss_weight = trial.suggest_uniform('physics_loss_weight', 2.5, 6.0)
    efficiency_loss_weight = trial.suggest_uniform('efficiency_loss_weight', 5.0, 8.0)

    r2_1_scores = []
    r2_2_scores = []

    train_folds_combined = pd.concat(train_folds, axis=0)

    columns_to_scale = [
        '径程比', '压缩比', '单缸排量/mm3', '燃烧室容积/mm3',

        '转速/rpm', '活塞平均速度/(m/s)', '单缸循环供油量/mg',
        '喷油时刻（°CA BTDC）', 'EGR/%', '单缸循环供能/J',

        '密度（kg/m3）', 'CN', 'LHV（MJ/kg）', '含氧量wt%', 'YSI-Das',

        '进气温度/K', '进气压力/bar',

        "负荷/bar", '热效率'
    ]

    scalers = {}
    for column in columns_to_scale:
        scaler = MinMaxScaler()
        scaler.fit(train_folds_combined[[column]])
        scalers[column] = scaler

    MEP_numpy = train_folds_combined["负荷/bar"]
    Efficiency_numpy = train_folds_combined["热效率"]
    MEP_hist, MEP_bin_centers, MEP_bin_edges, MEP_smoothed_hist = LDS(MEP_numpy, kernel_std=kernel_std_1,
                                                                      n_bins=n_bins_1)
    Efficiency_hist, Efficiency_bin_centers, Efficiency_bin_edges, Efficiency_smoothed_hist = LDS(Efficiency_numpy,
                                                                                                  kernel_std=kernel_std_2,
                                                                                                  n_bins=n_bins_2)
    lds_weights = LDSWeights()

    for val_fold_idx in range(5):
        val_fold = train_folds[val_fold_idx]
        train_folds_cv = pd.concat([train_folds[i] for i in range(5) if i != val_fold_idx], axis=0)  # 其他 folds 作为训练集

        fold_train_scaled_df = pd.DataFrame(index=train_folds_cv.index)
        fold_val_scaled_df = pd.DataFrame(index=val_fold.index)

        for column in columns_to_scale:
            fold_train_scaled_df[column] = scalers[column].transform(train_folds_cv[[column]])
            fold_val_scaled_df[column] = scalers[column].transform(val_fold[[column]])

        columns_to_keep = [col for col in train_folds_cv.columns if col not in columns_to_scale]
        fold_train_final = pd.concat([train_folds_cv[columns_to_keep], fold_train_scaled_df], axis=1)
        fold_val_final = pd.concat([val_fold[columns_to_keep], fold_val_scaled_df], axis=1)

        categorical_columns = ['负荷类型', '密度测试温度/K', '缸数', '进气方式']  # 替换为你的类别特征列名

        label_encoders = {}

        for col in categorical_columns:
            le = LabelEncoder()
            fold_train_final[col] = le.fit_transform(fold_train_final[col])
            fold_val_final[col] = le.transform(fold_val_final[col])
            label_encoders[col] = le

        fold_train_final = fold_train_final[train_folds_combined.columns]
        fold_val_final = fold_val_final[val_fold.columns]

        train_dataset = UserDataset(fold_train_final)
        val_dataset = UserDataset(fold_val_final)

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_dataset, batch_size=len(fold_val_final), shuffle=False)

        num_categories = [2, 2, 3]

        # TabTransformer
        model = TabTransformer(
            num_categories=num_categories,
            category_embedding_dim=category_embedding_dim,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout
        ).to('cuda')

        # MLP
        extra_features_dim = 2
        category_features_dim = 3
        continuous_features_dim = 15

        ST_MLP = MultiTaskMLP(
            input_size=transformer_d_model * (category_features_dim + continuous_features_dim) + extra_features_dim,
            shared_layer_sizes=ST_shared_layer_sizes,
            task1_sizes = ST_task1_layer_sizes,
            task2_sizes = ST_task2_layer_sizes,
            output_sizes = (1, 1)
        ).to('cuda')
        NA_MLP = MultiTaskMLP(
            input_size=transformer_d_model * (category_features_dim + continuous_features_dim),
            shared_layer_sizes=NA_shared_layer_sizes,
            task1_sizes = NA_task1_layer_sizes,
            task2_sizes = NA_task2_layer_sizes,
            output_sizes = (1, 1)
        ).to('cuda')
        ET_MLP = MultiTaskMLP(
            input_size=transformer_d_model * (category_features_dim + continuous_features_dim),
            shared_layer_sizes=ET_shared_layer_sizes,
            task1_sizes = ET_task1_layer_sizes,
            task2_sizes = ET_task2_layer_sizes,
            output_sizes = (1, 1)
        ).to('cuda')

        loss_basic = nn.MSELoss(reduction='none')

        loss_physics = PhysicsConLoss()

        optimizer = optim.Adam(
            list(model.parameters()) +
            list(ST_MLP.parameters()) +
            list(NA_MLP.parameters()) +
            list(ET_MLP.parameters()),
            lr=learning_rate
        )

        progress_bar = tqdm(range(num_epochs), desc=f"Trial {trial.number + 1} | Fold {val_fold_idx + 1}/5",
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
                            efficiency_loss_weight * loss_Efficiency * Efficiency_batch_weights).mean()

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

                batch_avg_loss = loss_supervised + physics_loss_weight * loss_physics_value

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

        Efficiency_true_all = []
        Efficiency_pred_all = []

        model.eval()
        ST_MLP.eval()
        NA_MLP.eval()
        ET_MLP.eval()

        with torch.no_grad():
            for (batch_category_features,
                 batch_geometry_features,
                 batch_operating_features,
                 batch_fuel_features,
                 batch_intake_mode,
                 batch_intake_extra,
                 _,
                 batch_TrueEfficiency) in train_loader:

                batch_continuous_features = torch.cat([
                    batch_geometry_features,
                    batch_operating_features,
                    batch_fuel_features
                ], dim=1)

                batch_contextual_embeddings = model(batch_category_features, batch_continuous_features)
                current_batch_size = batch_category_features.shape[0]
                Efficiency_predict = torch.zeros(current_batch_size, 1).to('cuda')

                for i in range(current_batch_size):
                    mode = batch_intake_mode[i].item()
                    embedding = batch_contextual_embeddings[i]
                    flattened_embedding = embedding.view(1, -1)
                    extra = batch_intake_extra[i].unsqueeze(0)

                    if mode == 2:
                        combined_features = torch.cat([flattened_embedding, extra], dim=-1)
                        _, Efficiency_predict[i] = ST_MLP(combined_features)
                    elif mode == 1:
                        combined_features = torch.cat([flattened_embedding], dim=-1)
                        _, Efficiency_predict[i] = NA_MLP(combined_features)
                    elif mode == 0:
                        combined_features = torch.cat([flattened_embedding], dim=-1)
                        _, Efficiency_predict[i] = ET_MLP(combined_features)

                Efficiency_true_all.append(batch_TrueEfficiency.cpu().numpy())
                Efficiency_pred_all.append(Efficiency_predict.cpu().numpy())

        Efficiency_true_all = np.concatenate(Efficiency_true_all)
        Efficiency_pred_all = np.concatenate(Efficiency_pred_all)

        r2_epoch = r2_score(Efficiency_true_all, Efficiency_pred_all)

        trial.report(r2_epoch, step=epoch)

        if epoch >= 30 and trial.should_prune():
            print(f"[Early Stop] Trial {trial.number} pruned at epoch {epoch}, R² = {r2_epoch:.4f}")
            raise optuna.exceptions.TrialPruned()

        val_MEP_avg_loss, val_Efficiency_avg_loss, val_MEP_preds, val_MEP_true, val_TrueE, val_PredE = evaluate(
            val_loader, model, ST_MLP, NA_MLP, ET_MLP, scalers)

        r2_1 = r2_score(val_MEP_true, val_MEP_preds)
        r2_2 = r2_score(val_TrueE, val_PredE)

        print(f"Fold {val_fold_idx + 1} - R2_MEP: {r2_1:.4f}, R2_ThermalEfficiency: {r2_2:.4f}")

        r2_1_scores.append(r2_1)
        r2_2_scores.append(r2_2)

    composite_score = 0.35 * np.mean(r2_1_scores) + 0.65 * np.mean(r2_2_scores)
    trial.set_user_attr("R2_MEP", np.mean(r2_1_scores))
    trial.set_user_attr("R2_Efficiency", np.mean(r2_2_scores))

    return composite_score

study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=30)
)

def callback(study, trial):
    composite_r2 = trial.value
    r2_mep = trial.user_attrs.get("R2_MEP", None)
    r2_eff = trial.user_attrs.get("R2_Efficiency", None)

    print(f"Trial {trial.number + 1} finished with Composite R² = {composite_r2:.4f}")
    if r2_mep is not None and r2_eff is not None:
        print(f"    R²_MEP = {r2_mep:.4f}, R²_ThermalEfficiency = {r2_eff:.4f}")
    print(f"    Parameters: {trial.params}")

study.optimize(objective, n_trials=n_trials, callbacks=[callback])

print("\nOptimization Completed!")
print("Best Trial:")
best_trial = study.best_trial
print(f"  Trial {best_trial.number}: Composite R² = {best_trial.value:.4f}")
print(f"  R²_MEP = {best_trial.user_attrs['R2_MEP']:.4f}, R²_ThermalEfficiency = {best_trial.user_attrs['R2_Efficiency']:.4f}")
print(f"  Params: {best_trial.params}")

history_data = []
for trial in study.trials:
    trial_data = {
        "Trial Number": trial.number,
        "Composite R2": trial.value,
        "R2_MEP": trial.user_attrs.get("R2_MEP", None),
        "R2_ThermalEfficiency": trial.user_attrs.get("R2_Efficiency", None),
        **trial.params,
    }
    history_data.append(trial_data)

history_df = pd.DataFrame(history_data)
history_df.to_csv("optuna result/optimization_history.csv", index=False)
print("Optimization history saved to 'optimization_history.csv'")

fig_score = vis.plot_optimization_history(study, target_name="Composite R2")
fig_score.write_html("optuna result/optimization_history_composite.html")
fig_score.show()