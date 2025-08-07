import os
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from deap import base, creator, tools, algorithms
from tqdm import tqdm

from TabTransformer import TabTransformer
from MultiTaskMLP import MultiTaskMLP

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dir = 'saved_models_ensemble'
scaler_path = 'saved_scalars_ensemble/scalers.pkl'

scalers = joblib.load(scaler_path)

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

print("Models loaded successfully!")

def predict_NA(MassOfFuel, Qin, rho, CN, LHV, O2, YSI, scalers):
    fixed_category_feature_IB = torch.zeros((1, 1), dtype=torch.long).to('cuda')
    fixed_category_feature_rhoTem = torch.ones((1, 1), dtype=torch.long).to('cuda')
    fixed_category_feature_CylinderNum = torch.ones((1, 1), dtype=torch.long).to('cuda')
    fixed_continuous_features = {
        "径程比": torch.tensor([[1.018181818]], dtype=torch.float32),
        "压缩比": torch.tensor([[19]], dtype=torch.float32),
        "单缸排量/mm3": torch.tensor([[1083723.802]], dtype=torch.float32),
        "燃烧室容积/mm3": torch.tensor([[60206.87788]], dtype=torch.float32),
        "转速/rpm": torch.tensor([[1800]], dtype=torch.float32),
        "活塞平均速度/(m/s)": torch.tensor([[6.6]], dtype=torch.float32),
        "喷油时刻（°CA BTDC）": torch.tensor([[8]], dtype=torch.float32),
        "EGR/%": torch.tensor([[0]], dtype=torch.float32),
    }
    for key in fixed_continuous_features:
        feature_df = pd.DataFrame(fixed_continuous_features[key].cpu().numpy(), columns=[key])
        fixed_continuous_features[key] = torch.tensor(
            scalers[key].transform(feature_df),
            dtype=torch.float32
        ).to('cuda')

    input_continuous_features = {
        "单缸循环供油量/mg": torch.tensor([[MassOfFuel]], dtype=torch.float32),
        "单缸循环供能/J": torch.tensor([[Qin]], dtype=torch.float32),
        "密度（kg/m3）": torch.tensor([[rho]], dtype=torch.float32),
        "CN": torch.tensor([[CN]], dtype=torch.float32),
        "LHV（MJ/kg）": torch.tensor([[LHV]], dtype=torch.float32),
        "含氧量wt%": torch.tensor([[O2]], dtype=torch.float32),
        "YSI-Das": torch.tensor([[YSI]], dtype=torch.float32),
    }
    for key in input_continuous_features:
        feature_df = pd.DataFrame(input_continuous_features[key].cpu().numpy(), columns=[key])
        input_continuous_features[key] = torch.tensor(
            scalers[key].transform(feature_df),
            dtype=torch.float32
        ).to('cuda')

    category_features = torch.cat([fixed_category_feature_IB, fixed_category_feature_rhoTem, fixed_category_feature_CylinderNum], dim=1)

    continuous_features = torch.cat(
        [fixed_continuous_features["径程比"],fixed_continuous_features["压缩比"],
         fixed_continuous_features['单缸排量/mm3'],fixed_continuous_features["燃烧室容积/mm3"],
         fixed_continuous_features["转速/rpm"], fixed_continuous_features["活塞平均速度/(m/s)"],
         input_continuous_features["单缸循环供油量/mg"],fixed_continuous_features["喷油时刻（°CA BTDC）"],
          fixed_continuous_features["EGR/%"], input_continuous_features["单缸循环供能/J"],
         input_continuous_features["密度（kg/m3）"], input_continuous_features["CN"],
         input_continuous_features["LHV（MJ/kg）"], input_continuous_features["含氧量wt%"], input_continuous_features["YSI-Das"]],
        dim=1
    )

    biases, mep_preds, eff_preds = [], [], []

    for fold_idx, (model, ST, NA, ET) in enumerate(models):
        with torch.no_grad():
            contextual_embeddings = model(category_features, continuous_features)

            flattened_embedding = contextual_embeddings.view(1, -1)
            combined_features = torch.cat([flattened_embedding], dim=-1)
            mep, eff = NA(combined_features)

            mep = mep.cpu().numpy()[0][0]
            eff = eff.cpu().numpy()[0][0]

            mep_original = mep * (scalers['负荷/bar'].data_max_ - scalers['负荷/bar'].data_min_) + scalers[
                '负荷/bar'].data_min_
            eff_original = eff * (scalers['热效率'].data_max_ - scalers['热效率'].data_min_) + scalers[
                '热效率'].data_min_
            V_disp = scalers["单缸排量/mm3"].inverse_transform(
        pd.DataFrame(fixed_continuous_features["单缸排量/mm3"].cpu().detach().numpy().reshape(-1, 1), columns=["单缸排量/mm3"])
    )
            Q_in = scalers["单缸循环供能/J"].inverse_transform(
        pd.DataFrame(input_continuous_features["单缸循环供能/J"].cpu().detach().numpy().reshape(-1, 1), columns=["单缸循环供能/J"])
    )

            bias = calculate_physics_bias(mep_original, Q_in, V_disp, eff_original)

            mep_preds.append(mep_original)
            eff_preds.append(eff_original)
            biases.append(bias)

    best_idx = np.argmin(biases)
    diffEfficiency = biases[best_idx]

    final_preds_mep = mep_preds[best_idx]
    final_preds_eff = eff_preds[best_idx]

    return final_preds_mep, final_preds_eff, diffEfficiency

TARGET_MEP = 7.0
TOLERANCE = 0.01

def find_mass_of_fuel(target_MEP, rho, CN, LHV, O2, YSI, scalers, tolerance=0.01, max_iter=20):
    low, high = 10.0, 80.0
    iter_count = 0

    while iter_count < max_iter:
        mass_of_fuel = (low + high) / 2
        Qin = mass_of_fuel * LHV
        MEP_pred, Efficiency_pred, diffEfficiency = predict_NA(
            MassOfFuel=mass_of_fuel, Qin=Qin, rho=rho, CN=CN, LHV=LHV, O2=O2, YSI=YSI, scalers=scalers
        )
        MEP_pred_val = MEP_pred.item()

        if abs(MEP_pred_val - target_MEP) < tolerance:
            return mass_of_fuel, MEP_pred_val, Efficiency_pred, diffEfficiency
        elif MEP_pred_val < target_MEP:
            low = mass_of_fuel
        else:
            high = mass_of_fuel

        iter_count += 1

    return mass_of_fuel, MEP_pred_val, Efficiency_pred, diffEfficiency

def fitness_function(x):
    rho, CN, LHV, O2, YSI = x

    if not (750<= rho <= 850):
        return (9999.0, 0.0)
    if not (40 <= CN <= 65):
        return (9999.0, 0.0)
    if not (38 <= LHV <= 48):
        return (9999.0, 0.0)
    if not (0 <= O2 <= 20):
        return (9999.0, 0.0)
    if not (10 <= YSI <= 200):
        return (9999.0, 0.0)

    MassOfFuel, MEP_pred_val, Efficiency_pred, diffEfficiency = find_mass_of_fuel(
        target_MEP=TARGET_MEP, rho=rho, CN=CN, LHV=LHV, O2=O2, YSI=YSI, scalers=scalers, tolerance=TOLERANCE
    )

    Eff_pred_val = Efficiency_pred.item()
    diff_val = diffEfficiency.item()

    return (diff_val, Eff_pred_val)

def random_combined_feature():
    rho_range = (750, 850)
    CN_range = (40, 55)
    LHV_range = (38, 45)
    O2_range = (0, 20)
    YSI_range = (10, 120)

    return [
        np.random.uniform(*rho_range),
        np.random.uniform(*CN_range),
        np.random.uniform(*LHV_range),
        np.random.uniform(*O2_range),
        np.random.uniform(*YSI_range),
    ]

POP_SIZE = 200
NGEN = 50
CXPB = 0.7
MUTPB = 0.3

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("individual", tools.initIterate, creator.Individual, random_combined_feature)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# use tools in deap to creat our application
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.5)

toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", fitness_function) # commit our evaluate

pop = toolbox.population(n=POP_SIZE)

for gen in tqdm(range(1, NGEN+1), desc="NSGA-II进化"):
    offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits_off = map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fits_off):
        ind.fitness.values = fit
    pop = toolbox.select(pop + offspring, k=POP_SIZE)

final_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

saved_info = []
for ind in final_front:
    rho, CN, LHV, O2, YSI = ind
    MassOfFuel, MEP_pred_val, Efficiency_pred, diffEfficiency = find_mass_of_fuel(
        target_MEP=TARGET_MEP, rho=rho, CN=CN, LHV=LHV, O2=O2, YSI=YSI, scalers=scalers, tolerance=TOLERANCE
    )

    Efficiency_pred = Efficiency_pred.item()
    diffEfficiency = diffEfficiency.item()

    saved_info.append({
        "MassOfFuel": MassOfFuel,
        "rho": rho,
        "CN": CN,
        "LHV": LHV,
        "O2": O2,
        "YSI": YSI,
        "MEP": MEP_pred_val,
        "Efficiency": Efficiency_pred,
        "diffEfficiency": diffEfficiency
    })

df_results = pd.DataFrame(saved_info)
df_results.to_csv("optimized_fuel_characteristics-1.csv", index=False, encoding='utf-8')