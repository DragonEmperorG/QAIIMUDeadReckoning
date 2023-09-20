import os

import torch
from torch import nn

from graphs.models.adaptive_parameter_adjustment_model import AdaptiveParameterAdjustmentModel
from graphs.models.noise_covariance_model import NoiseCovarianceModel
from graphs.models.state_covariance_model import StateCovarianceModel
from utils.constant_utils import TENSOR_EYE3, TENSOR_EYE12, TENSOR_EYE21, TIMESTAMP, FILTER_NAV_ROTATION_MATRIX, \
    FILTER_NAV_VELOCITY, FILTER_NAV_POSITION
from utils.lie_group_utils import so3exp, skew, sen3exp


class InvariantExtendedKalmanFilter(nn.Module):
    SEQUENCE_LEN = 'SEQUENCE_LEN'
    INIT_NAV_ROTATION_MATRIX = 'INIT_NAV_ROTATION_MATRIX'
    INIT_NAV_VELOCITY = 'INIT_NAV_VELOCITY'
    INIT_NAV_POSITION = 'INIT_NAV_POSITION'

    PROPAGATE_STATE_NAV_ROTATION_MATRIX = 'PROPAGATE_STATE_NAV_ROTATION_MATRIX'
    PROPAGATE_STATE_NAV_VELOCITY = 'PROPAGATE_STATE_NAV_VELOCITY'
    PROPAGATE_STATE_NAV_POSITION = 'PROPAGATE_STATE_NAV_POSITION'
    PROPAGATE_STATE_IMU_GYROSCOPE_BIAS = 'PROPAGATE_STATE_IMU_GYROSCOPE_BIAS'
    PROPAGATE_STATE_IMU_ACCELEROMETER_BIAS = 'PROPAGATE_STATE_IMU_ACCELEROMETER_BIAS'
    PROPAGATE_STATE_CAR_ROTATION_MATRIX = 'PROPAGATE_STATE_CAR_ROTATION_MATRIX'
    PROPAGATE_STATE_CAR_POSITION = 'PROPAGATE_STATE_CAR_POSITION'
    PROPAGATE_STATE_COVARIANCE = 'PROPAGATE_STATE_COVARIANCE'

    UPDATE_STATE_NAV_ROTATION_MATRIX = 'UPDATE_STATE_NAV_ROTATION_MATRIX'
    UPDATE_STATE_NAV_VELOCITY = 'UPDATE_STATE_NAV_VELOCITY'
    UPDATE_STATE_NAV_POSITION = 'UPDATE_STATE_NAV_POSITION'
    UPDATE_STATE_IMU_GYROSCOPE_BIAS = 'UPDATE_STATE_IMU_GYROSCOPE_BIAS'
    UPDATE_STATE_IMU_ACCELEROMETER_BIAS = 'UPDATE_STATE_IMU_ACCELEROMETER_BIAS'
    UPDATE_STATE_CAR_ROTATION_MATRIX = 'UPDATE_STATE_CAR_ROTATION_MATRIX'
    UPDATE_STATE_CAR_POSITION = 'UPDATE_STATE_CAR_POSITION'
    UPDATE_STATE_COVARIANCE = 'UPDATE_STATE_COVARIANCE'

    def __init__(self, device):
        super(InvariantExtendedKalmanFilter, self).__init__()
        self.device = device
        self.state_covariance_model = StateCovarianceModel(self.device)
        self.noise_covariance_model = NoiseCovarianceModel(self.device)
        self.adaptive_parameter_adjustment_model = AdaptiveParameterAdjustmentModel(self.device)
        self.state_dim = 21
        self.noise_dim = 18
        self.g = torch.tensor([0, 0, -9.80665]).to(self.device)
        self.sequence_len = None
        self.sequence_time = None
        self.measurement_delta_time = None
        self.measurement_gyroscope = None
        self.measurement_accelerometer = None
        self.measurement_car_velocity_forward = None
        self.state_nav_rotation_matrix = None
        self.state_nav_velocity = None
        self.state_nav_position = None
        self.state_imu_gyroscope_bias = None
        self.state_imu_accelerometer_bias = None
        self.state_car_rotation_matrix = None
        self.state_car_position = None
        self.state_covariance = None
        self.noise_covariance = None
        self.measurement_covariance = None

    def forward(self, sequence_data):
        self.filter_init(sequence_data)
        self.filter_loop()
        predicted_data_dic = {
            TIMESTAMP: self.sequence_time,
            FILTER_NAV_ROTATION_MATRIX: self.state_nav_rotation_matrix,
            FILTER_NAV_VELOCITY: self.state_nav_velocity,
            FILTER_NAV_POSITION: self.state_nav_position,
        }
        return predicted_data_dic

    @staticmethod
    def get_model_path(model_file_name):
        experiments_folder_name = 'experiments'
        experiments_folder_path = os.path.join(os.path.abspath('.'), experiments_folder_name)
        if not os.path.isdir(experiments_folder_path):
            os.mkdir(experiments_folder_path)

        checkpoints_folder_name = 'checkpoints'
        checkpoints_folder_path = os.path.join(experiments_folder_path, checkpoints_folder_name)
        if not os.path.isdir(checkpoints_folder_path):
            os.mkdir(checkpoints_folder_path)

        model_file_path = os.path.join(checkpoints_folder_path, model_file_name)
        return model_file_path

    def load_filter(self, model_file_name):
        self.load_state_dict(torch.load(InvariantExtendedKalmanFilter.get_model_path(model_file_name)))

    def save_filter(self, model_file_name):
        torch.save(self.state_dict(), InvariantExtendedKalmanFilter.get_model_path(model_file_name))

    def filter_init(self, sequence_data):
        self.prepare_filter(sequence_data)
        self.filter_init_state(sequence_data)

    def prepare_filter(self, sequence_data):
        self.sequence_len = sequence_data.timestamp.shape[0]
        self.sequence_time = torch.from_numpy(sequence_data.timestamp).to(self.device)
        self.measurement_delta_time = torch.zeros(self.sequence_len).to(self.device)
        self.measurement_delta_time[:-1] = self.sequence_time[1:] - self.sequence_time[:-1]
        self.measurement_gyroscope = torch.from_numpy(sequence_data.phone_measurement_gyroscope).to(self.device)
        self.measurement_accelerometer = torch.from_numpy(sequence_data.phone_measurement_accelerometer).to(self.device)
        self.measurement_car_velocity_forward = torch.from_numpy(sequence_data.pseudo_measurement_car_velocity_forward).to(self.device)
        self.measurement_covariance = self.adaptive_parameter_adjustment_model(torch.from_numpy(sequence_data.phone_measurement_normalized).to(self.device))

        self.state_nav_rotation_matrix = torch.zeros(self.sequence_len, 3, 3, dtype=torch.float64).to(self.device)
        self.state_nav_velocity = torch.zeros(self.sequence_len, 3, dtype=torch.float64).to(self.device)
        self.state_nav_position = torch.zeros(self.sequence_len, 3, dtype=torch.float64).to(self.device)
        self.state_imu_gyroscope_bias = torch.zeros(self.sequence_len, 3, dtype=torch.float64).to(self.device)
        self.state_imu_accelerometer_bias = torch.zeros(self.sequence_len, 3, dtype=torch.float64).to(self.device)
        self.state_car_rotation_matrix = torch.zeros(self.sequence_len, 3, 3, dtype=torch.float64).to(self.device)
        self.state_car_position = torch.zeros(self.sequence_len, 3, dtype=torch.float64).to(self.device)

    def filter_init_state(self, sequence_data):
        state_nav_rotation_matrix_init = sequence_data.ground_truth_nav_rotation_matrix[0, :, :]
        state_nav_velocity_init = sequence_data.ground_truth_nav_velocity[0, :]
        state_nav_position_init = sequence_data.ground_truth_nav_position[0, :]
        self.state_nav_rotation_matrix[0] = torch.from_numpy(state_nav_rotation_matrix_init)
        self.state_nav_velocity[0] = torch.from_numpy(state_nav_velocity_init)
        self.state_nav_position[0] = torch.from_numpy(state_nav_position_init)
        self.state_car_rotation_matrix[0] = torch.eye(3)
        self.filter_init_state_covariance()
        self.filter_init_noise_covariance()

    def filter_init_state_covariance(self):
        self.state_covariance = self.state_covariance_model(torch.ones(1).to(self.device))

    def filter_init_noise_covariance(self):
        self.noise_covariance = self.noise_covariance_model(torch.ones(1).to(self.device))

    def filter_loop(self):
        for i in range(1, self.sequence_len):
            propagated_state = self.filter_propagate(i)
            update_state = self.filter_update(i, propagated_state)

            self.state_nav_rotation_matrix[i, :, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_NAV_ROTATION_MATRIX]
            self.state_nav_velocity[i, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_NAV_VELOCITY]
            self.state_nav_position[i, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_NAV_POSITION]
            self.state_imu_gyroscope_bias[i, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_IMU_GYROSCOPE_BIAS]
            self.state_imu_accelerometer_bias[i, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_IMU_ACCELEROMETER_BIAS]
            self.state_car_rotation_matrix[i, :, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_CAR_ROTATION_MATRIX]
            self.state_car_position[i, :] = update_state[InvariantExtendedKalmanFilter.UPDATE_STATE_CAR_POSITION]

    def filter_propagate(self, i):
        base_state_nav_rotation_matrix = self.state_nav_rotation_matrix[i - 1, :, :].clone()

        imu_gyroscope = self.measurement_gyroscope[i - 1, :] - self.state_imu_gyroscope_bias[i - 1, :]
        delta_nav_rotation_matrix = imu_gyroscope * self.measurement_delta_time[i - 1]
        propagate_state_nav_rotation_matrix = base_state_nav_rotation_matrix.mm(so3exp(delta_nav_rotation_matrix))
        imu_accelerometer = self.measurement_accelerometer[i - 1, :] - self.state_imu_accelerometer_bias[i - 1, :]
        nav_accelerometer = base_state_nav_rotation_matrix.mv(imu_accelerometer) + self.g
        propagate_state_nav_velocity = self.state_nav_velocity[i - 1, :] + nav_accelerometer * self.measurement_delta_time[i - 1]
        propagate_state_nav_position = self.state_nav_position[i - 1, :] + (self.state_nav_velocity[i - 1, :].clone() + propagate_state_nav_velocity.clone()) * self.measurement_delta_time[i - 1] * 0.5
        propagate_state_imu_gyroscope_bias = self.state_imu_gyroscope_bias[i - 1, :]
        propagate_state_imu_accelerometer_bias = self.state_imu_accelerometer_bias[i - 1, :]
        propagate_state_car_rotation_matrix = self.state_car_rotation_matrix[i - 1, :, :]
        propagate_state_car_position = self.state_car_position[i - 1, :]

        f_state_jacobian_matrix = torch.zeros(self.state_dim, self.state_dim).to(self.device)
        f_state_jacobian_matrix[3:6, :3] = skew(self.g)
        f_state_jacobian_matrix[6:9, 3:6] = TENSOR_EYE3
        f_state_jacobian_matrix[:3, 9:12] = -base_state_nav_rotation_matrix
        f_state_jacobian_matrix[3:6, 9:12] = -skew(self.state_nav_velocity[i - 1, :]).mm(base_state_nav_rotation_matrix)
        f_state_jacobian_matrix[6:9, 9:12] = -skew(self.state_nav_position[i - 1, :]).mm(base_state_nav_rotation_matrix)
        f_state_jacobian_matrix[3:6, 12:15] = -base_state_nav_rotation_matrix
        f_state_jacobian_matrix = f_state_jacobian_matrix * self.measurement_delta_time[i - 1]
        f_state_jacobian_matrix_square = f_state_jacobian_matrix.mm(f_state_jacobian_matrix)
        f_state_jacobian_matrix_cube = f_state_jacobian_matrix_square.mm(f_state_jacobian_matrix)
        f_state_jacobian_matrix_phi = TENSOR_EYE21.to(self.device) + f_state_jacobian_matrix + 1.0 / 2 * f_state_jacobian_matrix_square + 1.0 / 6 * f_state_jacobian_matrix_cube

        f_noise_jacobian_matrix = torch.zeros(self.state_dim, self.noise_dim, dtype=torch.float64).to(self.device)
        f_noise_jacobian_matrix[:9, :6] = -f_state_jacobian_matrix[:9, 9:15]
        f_noise_jacobian_matrix[9:21, 6:18] = TENSOR_EYE12
        f_noise_jacobian_matrix = f_noise_jacobian_matrix * self.measurement_delta_time[i - 1]

        propagate_state_covariance = f_state_jacobian_matrix_phi \
            .mm(self.state_covariance + f_noise_jacobian_matrix.mm(self.noise_covariance).mm(f_noise_jacobian_matrix.t())) \
            .mm(f_state_jacobian_matrix_phi.t())

        propagate_state = {
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_ROTATION_MATRIX: propagate_state_nav_rotation_matrix.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_VELOCITY: propagate_state_nav_velocity.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_POSITION: propagate_state_nav_position.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_IMU_GYROSCOPE_BIAS: propagate_state_imu_gyroscope_bias.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_IMU_ACCELEROMETER_BIAS: propagate_state_imu_accelerometer_bias.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_ROTATION_MATRIX: propagate_state_car_rotation_matrix.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_POSITION: propagate_state_car_position.clone(),
            InvariantExtendedKalmanFilter.PROPAGATE_STATE_COVARIANCE: propagate_state_covariance,
        }
        return propagate_state

    def filter_update(self, i, propagated_state):
        propagate_measurement_gyroscope = self.measurement_gyroscope[i - 1, :] - propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_IMU_GYROSCOPE_BIAS]
        propagate_measurement_gyroscope_skew = skew(propagate_measurement_gyroscope)
        propagate_state_nav_car_rotation_matrix = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_ROTATION_MATRIX].mm(
            propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_ROTATION_MATRIX])
        propagate_state_imu_velocity = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_ROTATION_MATRIX].t().mv(
            propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_VELOCITY])
        propagate_state_car_velocity = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_ROTATION_MATRIX].t().mv(propagate_state_imu_velocity) + skew(
            propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_POSITION]).mv(propagate_measurement_gyroscope)

        h_function_car_velocity = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_ROTATION_MATRIX].t().mm(skew(propagate_state_imu_velocity))
        h_function_car_position = -skew(propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_POSITION])
        filter_function_h = torch.zeros(3, self.state_dim, dtype=torch.float64).to(self.device)
        filter_function_h[:, 3:6] = propagate_state_nav_car_rotation_matrix.t()
        filter_function_h[:, 9:12] = h_function_car_position
        filter_function_h[:, 15:18] = h_function_car_velocity
        filter_function_h[:, 18:21] = - propagate_measurement_gyroscope_skew
        measurement_covariance_r = torch.diag(self.measurement_covariance[i, :]).double().to(self.device)
        filter_function_s = filter_function_h.mm(propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_COVARIANCE]).mm(filter_function_h.t()) + measurement_covariance_r
        filter_gain_k_t = torch.linalg.solve(filter_function_s, propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_COVARIANCE].mm(filter_function_h.t()).t())
        filter_gain_k = filter_gain_k_t.t()

        measurement_state_car_velocity = torch.zeros(3, dtype=torch.float64).to(self.device)
        measurement_state_car_velocity[1] = self.measurement_car_velocity_forward[i, :]
        measurement_state_car_velocity_error = measurement_state_car_velocity - propagate_state_car_velocity
        delta_filter_state = filter_gain_k.mv(measurement_state_car_velocity_error.view(-1))

        delta_filter_state_nav_rotation_matrix, delta_filter_state_nav_velocity, delta_filter_state_nav_position = sen3exp(delta_filter_state[:9])
        update_state_nav_rotation_matrix = delta_filter_state_nav_rotation_matrix.mm(propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_ROTATION_MATRIX])
        update_state_nav_velocity = delta_filter_state_nav_rotation_matrix.mv(
            propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_VELOCITY]) + delta_filter_state_nav_velocity
        update_state_nav_position = delta_filter_state_nav_rotation_matrix.mv(
            propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_NAV_POSITION]) + delta_filter_state_nav_position
        update_state_imu_gyroscope_bias = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_IMU_GYROSCOPE_BIAS] + delta_filter_state[9:12]
        update_state_imu_accelerometer_bias = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_IMU_ACCELEROMETER_BIAS] + delta_filter_state[12:15]

        delta_filter_state_car_rotation_matrix = so3exp(delta_filter_state[15:18])
        update_state_car_rotation_matrix = delta_filter_state_car_rotation_matrix.mm(propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_ROTATION_MATRIX])
        update_state_car_position = propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_CAR_POSITION] + delta_filter_state[18:21]

        function_i_kh = TENSOR_EYE21.to(self.device) - filter_gain_k.mm(filter_function_h)
        update_state_covariance_raw = function_i_kh.mm(propagated_state[InvariantExtendedKalmanFilter.PROPAGATE_STATE_COVARIANCE]).mm(function_i_kh.t()) + filter_gain_k.mm(
            measurement_covariance_r).mm(filter_gain_k.t())
        update_state_covariance = (update_state_covariance_raw + update_state_covariance_raw.t()) * 0.5

        update_state = {
            InvariantExtendedKalmanFilter.UPDATE_STATE_NAV_ROTATION_MATRIX: update_state_nav_rotation_matrix,
            InvariantExtendedKalmanFilter.UPDATE_STATE_NAV_VELOCITY: update_state_nav_velocity,
            InvariantExtendedKalmanFilter.UPDATE_STATE_NAV_POSITION: update_state_nav_position,
            InvariantExtendedKalmanFilter.UPDATE_STATE_IMU_GYROSCOPE_BIAS: update_state_imu_gyroscope_bias,
            InvariantExtendedKalmanFilter.UPDATE_STATE_IMU_ACCELEROMETER_BIAS: update_state_imu_accelerometer_bias,
            InvariantExtendedKalmanFilter.UPDATE_STATE_CAR_ROTATION_MATRIX: update_state_car_rotation_matrix,
            InvariantExtendedKalmanFilter.UPDATE_STATE_CAR_POSITION: update_state_car_position,
            InvariantExtendedKalmanFilter.UPDATE_STATE_COVARIANCE: update_state_covariance,
        }
        return update_state
