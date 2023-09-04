import torch
from torch import nn


class AdaptiveParameterAdjustmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor_beta = 3 * torch.ones(3)
        self.cov_lat = 1
        self.cov_lon = 5
        self.cov_up = 10
        self.pseudo_measurement_covariance_base = torch.tensor([self.cov_lat, self.cov_lon, self.cov_up])

        self.cov_net = nn.Sequential(
            nn.Conv1d(6, 32, 5),
            nn.ReplicationPad1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(32, 32, 5, dilation=3),
            nn.ReplicationPad1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.cov_lin = nn.Sequential(
            nn.Linear(32, 3),
            nn.Tanh(),
        )

        self.cov_lin[0].bias.data[:] /= 100
        self.cov_lin[0].weight.data[:] /= 100

    def forward(self, normalized_phone_data):
        if not torch.is_tensor(normalized_phone_data):
            normalized_phone_data = torch.from_numpy(normalized_phone_data)

        factor_cnn_input = normalized_phone_data.t().unsqueeze(0).float()
        factor_cnn = self.cov_net(factor_cnn_input).transpose(0, 2).squeeze()
        factor_lin = self.cov_lin(factor_cnn)
        factor_power = self.factor_beta.unsqueeze(0) * factor_lin
        pseudo_measurement_covariance = self.pseudo_measurement_covariance_base.unsqueeze(0) * (10 ** factor_power)
        return pseudo_measurement_covariance
