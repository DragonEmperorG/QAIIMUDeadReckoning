import torch
from torch import nn

from utils.constant_utils import TENSOR_EYE3


class NoiseCovarianceModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.noise_dim = 18
        self.init_noise_covariance_imu_gyroscope = 2e-4
        self.init_noise_covariance_imu_accelerometer_bias = 1e-3
        self.init_noise_covariance_imu_gyroscope_bias = 1e-8
        self.init_noise_covariance_imu_accelerometer_bias = 1e-6
        self.init_noise_covariance_car_rotation_matrix = 1e-8
        self.init_noise_covariance_car_position = 1e-8
        self.fc = nn.Linear(1, 6, bias=False)
        self.fc.weight.data[:] /= 10
        self.tanh = nn.Tanh()

    def forward(self, x=torch.ones(1)):
        noise_covariance_factor = self.fc(x)
        noise_covariance_factor = self.tanh(noise_covariance_factor)
        noise_covariance_factor = 10 ** noise_covariance_factor
        noise_covariance = torch.zeros(self.noise_dim, self.noise_dim, dtype=torch.float64)
        noise_covariance[:3, :3] = self.init_noise_covariance_imu_gyroscope * noise_covariance_factor[0] * TENSOR_EYE3
        noise_covariance[3:6, 3:6] = self.init_noise_covariance_imu_accelerometer_bias * noise_covariance_factor[1] * TENSOR_EYE3
        noise_covariance[6:9, 6:9] = self.init_noise_covariance_imu_gyroscope_bias * noise_covariance_factor[2] * TENSOR_EYE3
        noise_covariance[9:12, 9:12] = self.init_noise_covariance_imu_accelerometer_bias * noise_covariance_factor[3] * TENSOR_EYE3
        noise_covariance[12:15, 12:15] = self.init_noise_covariance_car_rotation_matrix * noise_covariance_factor[4] * TENSOR_EYE3
        noise_covariance[15:18, 15:18] = self.init_noise_covariance_car_position * noise_covariance_factor[5] * TENSOR_EYE3
        return noise_covariance
