import numpy as np
import torch

def evaluate(loader, model, ST_MLP, NA_MLP, ET_MLP, scalers):
    model.eval()
    ST_MLP.eval()
    NA_MLP.eval()
    ET_MLP.eval()

    MEP_total_loss = 0
    Efficiency_total_loss = 0
    total_samples = 0
    MEP_preds = []
    MEP_true = []
    TrueE = []
    PredE = []

    with torch.no_grad():
        for (batch_category_features,
             batch_geometry_features,
             batch_operating_features,
             batch_fuel_features,
             batch_intake_mode,
             batch_intake_extra,
             batch_TrueMEP,
             batch_TrueEfficiency) in loader:

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

            热效率_min_tensor = torch.tensor(scalers["热效率"].data_min_, dtype=torch.float32,
                                             device=MEP_predict.device).unsqueeze(0)
            热效率_max_tensor = torch.tensor(scalers["热效率"].data_max_, dtype=torch.float32,
                                             device=MEP_predict.device).unsqueeze(0)

            负荷_min_tensor = torch.tensor(scalers["负荷/bar"].data_min_, dtype=torch.float32,
                                           device=MEP_predict.device).unsqueeze(0)
            负荷_max_tensor = torch.tensor(scalers["负荷/bar"].data_max_, dtype=torch.float32,
                                           device=MEP_predict.device).unsqueeze(0)

            TrueEfficiency = batch_TrueEfficiency.unsqueeze(1)
            TrueMEP_original_tensor = batch_TrueMEP.unsqueeze(1)

            TrueEfficiency_original_tensor = TrueEfficiency * (热效率_max_tensor - 热效率_min_tensor) + 热效率_min_tensor  # [52, 1]
            Efficiency_predict = Efficiency_predict * (热效率_max_tensor - 热效率_min_tensor) + 热效率_min_tensor  # [52, 1]

            TrueMEP_original_tensor = TrueMEP_original_tensor * (负荷_max_tensor - 负荷_min_tensor) + 负荷_min_tensor
            MEP_predict = MEP_predict * (负荷_max_tensor - 负荷_min_tensor) + 负荷_min_tensor  # [52, 1]

            MEP_total_loss += torch.sum((MEP_predict - TrueMEP_original_tensor) ** 2).cpu().numpy()
            Efficiency_total_loss += torch.sum((Efficiency_predict - TrueEfficiency_original_tensor) ** 2).cpu().numpy()
            total_samples += current_batch_size

            MEP_preds.append(MEP_predict.cpu().numpy())
            MEP_true.append(TrueMEP_original_tensor.cpu().numpy())
            TrueE.append(TrueEfficiency_original_tensor.cpu().numpy())
            PredE.append(Efficiency_predict.cpu().numpy())

    MEP_avg_loss = MEP_total_loss / total_samples
    Efficiency_avg_loss = Efficiency_total_loss / total_samples
    MEP_preds = np.concatenate(MEP_preds, axis=0)
    MEP_true = np.concatenate(MEP_true, axis=0)
    TrueE = np.concatenate(TrueE, axis=0)
    PredE = np.concatenate(PredE, axis=0)
    return MEP_avg_loss, Efficiency_avg_loss, MEP_preds, MEP_true, TrueE, PredE