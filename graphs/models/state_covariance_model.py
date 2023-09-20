import torch
from torch import nn

from utils.constant_utils import TENSOR_EYE2, TENSOR_EYE3


class StateCovarianceModel(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.state_dim = 21
        self.init_state_covariance_nav_rotation_matrix = 1e-6
        self.init_state_covariance_nav_velocity = 1e-1
        self.init_state_covariance_nav_gyroscope_bias = 1e-8
        self.init_state_covariance_nav_accelerometer_bias = 1e-3
        self.init_state_covariance_car_rotation_matrix = 1e-5
        self.init_state_covariance_car_position = 1e-2
        self.fc = nn.Linear(1, 6, bias=False)
        self.fc.weight.data[:] /= 10
        self.tanh = torch.nn.Tanh()

    def forward(self, x=torch.ones(1)):
        state_covariance_factor = self.fc(x)
        state_covariance_factor = self.tanh(state_covariance_factor)
        state_covariance_factor = 10 ** state_covariance_factor
        state_covariance = torch.zeros(self.state_dim, self.state_dim, dtype=torch.float64).to(self.device)
        state_covariance[:2, :2] = self.init_state_covariance_nav_rotation_matrix * state_covariance_factor[0] * TENSOR_EYE2.to(self.device)
        state_covariance[3:5, 3:5] = self.init_state_covariance_nav_velocity * state_covariance_factor[1] * TENSOR_EYE2.to(self.device)
        state_covariance[9:12, 9:12] = self.init_state_covariance_nav_gyroscope_bias * state_covariance_factor[3] * TENSOR_EYE3.to(self.device)
        state_covariance[12:15, 12:15] = self.init_state_covariance_nav_accelerometer_bias * state_covariance_factor[4] * TENSOR_EYE3.to(self.device)
        state_covariance[15:18, 15:18] = self.init_state_covariance_car_rotation_matrix * state_covariance_factor[5] * TENSOR_EYE3.to(self.device)
        state_covariance[18:21, 18:21] = self.init_state_covariance_car_position * state_covariance_factor[5] * TENSOR_EYE3.to(self.device)
        return state_covariance
