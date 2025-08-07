import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, dataframe):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.category_features = torch.tensor(
            dataframe[['负荷类型', '密度测试温度/K', '缸数']].values,
            dtype=torch.long, device=device
        )

        self.geometry_features = torch.tensor(
            dataframe[['径程比', '压缩比', '单缸排量/mm3', '燃烧室容积/mm3']].values,
            dtype=torch.float32, device=device
        )

        self.operating_features = torch.tensor(
            dataframe[['转速/rpm', '活塞平均速度/(m/s)', '单缸循环供油量/mg',
                       '喷油时刻（°CA BTDC）', 'EGR/%', '单缸循环供能/J']].values,
            dtype=torch.float32, device=device
        )

        self.fuel_features = torch.tensor(
            dataframe[['密度（kg/m3）', 'CN', 'LHV（MJ/kg）', '含氧量wt%', 'YSI-Das']].values,
            dtype=torch.float32, device=device
        )

        self.intake_mode = torch.tensor(
            dataframe['进气方式'].values,
            dtype=torch.long, device=device
        )

        self.intake_extra = torch.tensor(
            dataframe[['进气温度/K', '进气压力/bar']].values,
            dtype=torch.float32, device=device
        )

        self.TrueMEP = torch.tensor(
            dataframe['负荷/bar'].values,
            dtype=torch.float32, device=device
        )
        self.TrueEfficiency = torch.tensor(
            dataframe['热效率'].values,
            dtype=torch.float32, device=device
        )

    def __len__(self):
        return len(self.TrueMEP)

    def __getitem__(self, idx):
        return (
            self.category_features[idx],
            self.geometry_features[idx],
            self.operating_features[idx],
            self.fuel_features[idx],
            self.intake_mode[idx],
            self.intake_extra[idx],
            self.TrueMEP[idx],
            self.TrueEfficiency[idx]
        )