import math
import os

import numpy as np
import torch

from datasets.sequence_dataset import SequenceDataset
from utils.constant_utils import SAMPLE_INDEX, TIMESTAMP, PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD, \
    PHONE_MEASUREMENT_GYROSCOPE, PHONE_MEASUREMENT_ACCELEROMETER, PHONE_MEASUREMENT_NORMALIZED,\
    GROUND_TRUTH_NAV_ROTATION_MATRIX, GROUND_TRUTH_NAV_VELOCITY, GROUND_TRUTH_NAV_POSITION, \
    FILTER_NAV_ROTATION_MATRIX, FILTER_NAV_VELOCITY, FILTER_NAV_POSITION, \
    FILTER_IMU_GYROSCOPE_BIAS, FILTER_IMU_ACCELEROMETER_BIAS, \
    FILTER_CAR_ROTATION_MATRIX, FILTER_CAR_POSITION, \
    FILTER_STATE_COVARIANCE, FILTER_NOISE_COVARIANCE, FILTER_MEASUREMENT_COVARIANCE
from utils.data_utils import resample_data
from utils.files.preprocessing_colleted_file_util import load_preprocessing_collected_data
from utils.files.preprocessing_pseudo_measurement_file_util import load_preprocessing_pseudo_measurement_data
from utils.logs.log_utils import get_logger


class TrackDataset:

    VDR_DATASET_FOLDER_NAME = "DATASET_QAIIMUDEADRECKONING"
    VDR_DATASET_FILE_NAME = "QAIIMUDeadReckoningTrainData.npz"
    VDR_DATASET_RESULT_FILE_NAME = "QAIIMUDeadReckoningResultData.npz"

    PHONE_GYROSCOPE_X = 'PHONE_GYROSCOPE_X'
    PHONE_GYROSCOPE_Y = 'PHONE_GYROSCOPE_Y'
    PHONE_GYROSCOPE_Z = 'PHONE_GYROSCOPE_Z'
    PHONE_ACCELEROMETER_X = 'PHONE_ACCELEROMETER_X'
    PHONE_ACCELEROMETER_Y = 'PHONE_ACCELEROMETER_Y'
    PHONE_ACCELEROMETER_Z = 'PHONE_ACCELEROMETER_Z'
    PHONE_PRESSURE = 'PHONE_PRESSURE'

    RELATIVE_SEQUENCE_LENGTH = [50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    def __init__(self, track_phone_folder_path, normalize_factors):
        self.track_phone_folder_path = track_phone_folder_path
        track_folder_path, self.phone_folder_name = os.path.split(track_phone_folder_path)
        _, self.track_folder_name = os.path.split(track_folder_path)
        track_dataset_dic = TrackDataset.load_data(track_phone_folder_path)
        self.timestamp = track_dataset_dic[TIMESTAMP]
        self.phone_measurement_gyroscope = track_dataset_dic[PHONE_MEASUREMENT_GYROSCOPE]
        self.phone_measurement_accelerometer = track_dataset_dic[PHONE_MEASUREMENT_ACCELEROMETER]
        self.pseudo_measurement_car_velocity_forward = track_dataset_dic[PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD]
        self.ground_truth_nav_rotation_matrix = track_dataset_dic[GROUND_TRUTH_NAV_ROTATION_MATRIX].reshape(self.timestamp.shape[0], 3, 3)
        self.ground_truth_nav_velocity = track_dataset_dic[GROUND_TRUTH_NAV_VELOCITY]
        self.ground_truth_nav_position = track_dataset_dic[GROUND_TRUTH_NAV_POSITION]

        self.sample_time_interval = np.mean(self.timestamp[1:] - self.timestamp[:-1])
        self.sample_rate = 1 / self.sample_time_interval

        self.ground_truth_relative_translation = self.prepare_ground_truth_relative_translation()

        phone_measurement = np.concatenate(
            (self.phone_measurement_gyroscope, self.phone_measurement_accelerometer),
            axis=1
        )
        self.phone_measurement_normalized = (phone_measurement - normalize_factors['mean']) / normalize_factors['std']

        self.train_loss_min = np.finfo(np.float64).max

    def get_track_name(self):
        return self.track_folder_name

    def get_phone_name(self):
        return self.phone_folder_name

    @staticmethod
    def load_data(folder_path):

        vdr_dataset_folder_path = os.path.join(
            folder_path,
            TrackDataset.VDR_DATASET_FOLDER_NAME
        )

        if os.path.isdir(vdr_dataset_folder_path):
            vdr_dataset_file_path = os.path.join(vdr_dataset_folder_path, TrackDataset.VDR_DATASET_FILE_NAME)
            if os.path.isfile(vdr_dataset_file_path):
                vdr_dataset = np.load(vdr_dataset_file_path)
            else:
                vdr_dataset = TrackDataset.load_preprocessing_data(folder_path)
                np.savez(
                    vdr_dataset_file_path,
                    TIMESTAMP=vdr_dataset[TIMESTAMP],
                    PHONE_MEASUREMENT_GYROSCOPE=vdr_dataset[PHONE_MEASUREMENT_GYROSCOPE],
                    PHONE_MEASUREMENT_ACCELEROMETER=vdr_dataset[PHONE_MEASUREMENT_ACCELEROMETER],
                    PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD=vdr_dataset[PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD],
                    GROUND_TRUTH_NAV_ROTATION_MATRIX=vdr_dataset[GROUND_TRUTH_NAV_ROTATION_MATRIX],
                    GROUND_TRUTH_NAV_VELOCITY=vdr_dataset[GROUND_TRUTH_NAV_VELOCITY],
                    GROUND_TRUTH_NAV_POSITION=vdr_dataset[GROUND_TRUTH_NAV_POSITION]
                )
        else:
            vdr_dataset = TrackDataset.load_preprocessing_data(folder_path)
            os.mkdir(vdr_dataset_folder_path)
            vdr_dataset_file_path = os.path.join(vdr_dataset_folder_path, TrackDataset.VDR_DATASET_FILE_NAME)
            np.savez(
                vdr_dataset_file_path,
                TIMESTAMP=vdr_dataset[TIMESTAMP],
                PHONE_MEASUREMENT_GYROSCOPE=vdr_dataset[PHONE_MEASUREMENT_GYROSCOPE],
                PHONE_MEASUREMENT_ACCELEROMETER=vdr_dataset[PHONE_MEASUREMENT_ACCELEROMETER],
                PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD=vdr_dataset[PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD],
                GROUND_TRUTH_NAV_ROTATION_MATRIX=vdr_dataset[GROUND_TRUTH_NAV_ROTATION_MATRIX],
                GROUND_TRUTH_NAV_VELOCITY=vdr_dataset[GROUND_TRUTH_NAV_VELOCITY],
                GROUND_TRUTH_NAV_POSITION=vdr_dataset[GROUND_TRUTH_NAV_POSITION]
            )

        loaded_vdr_dataset = TrackDataset.preprocessing_integral_head_tail_data(vdr_dataset)

        return loaded_vdr_dataset

    @staticmethod
    def preprocessing_integral_head_tail_data(raw_vdr_dataset):
        timestamp = raw_vdr_dataset[TIMESTAMP]
        head_time_ceil = math.ceil(timestamp[0])
        head_time_ceil_index, = np.where(timestamp == head_time_ceil)
        tail_time_floor = math.floor(timestamp[-1])
        tail_time_floor_index, = np.where(timestamp == tail_time_floor)
        slice_index = np.arange(head_time_ceil_index, tail_time_floor_index+1)
        preprocessed_vdr_dataset = {
            TIMESTAMP: raw_vdr_dataset[TIMESTAMP][slice_index],
            PHONE_MEASUREMENT_GYROSCOPE: raw_vdr_dataset[PHONE_MEASUREMENT_GYROSCOPE][slice_index, :],
            PHONE_MEASUREMENT_ACCELEROMETER: raw_vdr_dataset[PHONE_MEASUREMENT_ACCELEROMETER][slice_index, :],
            PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD: raw_vdr_dataset[PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD],
            GROUND_TRUTH_NAV_ROTATION_MATRIX: raw_vdr_dataset[GROUND_TRUTH_NAV_ROTATION_MATRIX][slice_index, :],
            GROUND_TRUTH_NAV_VELOCITY: raw_vdr_dataset[GROUND_TRUTH_NAV_VELOCITY][slice_index, :],
            GROUND_TRUTH_NAV_POSITION: raw_vdr_dataset[GROUND_TRUTH_NAV_POSITION][slice_index, :],
        }
        return preprocessed_vdr_dataset

    @staticmethod
    def load_preprocessing_data(folder_path):
        timestamp, phone_measurement_gyroscope, phone_measurement_accelerometer, ground_truth_nav_rotation_matrix, ground_truth_nav_velocity, ground_truth_nav_position = load_preprocessing_collected_data(folder_path)
        pseudo_measurement_timestamp, pseudo_measurement_car_velocity_forward = load_preprocessing_pseudo_measurement_data(folder_path)
        resampled_pseudo_measurement_car_velocity_forward = resample_data(pseudo_measurement_timestamp, pseudo_measurement_car_velocity_forward, timestamp)
        preprocessing_data_dic = {
            TIMESTAMP: timestamp,
            PHONE_MEASUREMENT_GYROSCOPE: phone_measurement_gyroscope,
            PHONE_MEASUREMENT_ACCELEROMETER: phone_measurement_accelerometer,
            PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD: resampled_pseudo_measurement_car_velocity_forward,
            GROUND_TRUTH_NAV_ROTATION_MATRIX: ground_truth_nav_rotation_matrix,
            GROUND_TRUTH_NAV_VELOCITY: ground_truth_nav_velocity,
            GROUND_TRUTH_NAV_POSITION: ground_truth_nav_position,
        }
        return preprocessing_data_dic

    def prepare_random_sample_sequence(self, train_random_sample_seq_len):
        sequence_index_head_tail = self.random_sample_sequence_index(train_random_sample_seq_len)
        slice_index = np.arange(sequence_index_head_tail[0], sequence_index_head_tail[1])
        random_sampled_sequence_dic = {
            SAMPLE_INDEX: sequence_index_head_tail,
            TIMESTAMP: self.timestamp[slice_index],
            PHONE_MEASUREMENT_GYROSCOPE: self.phone_measurement_gyroscope[slice_index, :],
            PHONE_MEASUREMENT_ACCELEROMETER: self.phone_measurement_accelerometer[slice_index, :],
            PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD: self.pseudo_measurement_car_velocity_forward[slice_index],
            GROUND_TRUTH_NAV_ROTATION_MATRIX: self.ground_truth_nav_rotation_matrix[slice_index, :, :],
            GROUND_TRUTH_NAV_VELOCITY: self.ground_truth_nav_velocity[slice_index, :],
            GROUND_TRUTH_NAV_POSITION: self.ground_truth_nav_position[slice_index, :],
            PHONE_MEASUREMENT_NORMALIZED: self.phone_measurement_normalized[slice_index, :]
        }
        return SequenceDataset(random_sampled_sequence_dic)

    def save_test_data(self, test_data):
        vdr_dataset_folder_path = os.path.join(
            self.track_phone_folder_path,
            TrackDataset.VDR_DATASET_FOLDER_NAME
        )
        vdr_dataset_file_path = os.path.join(vdr_dataset_folder_path, TrackDataset.VDR_DATASET_RESULT_FILE_NAME)
        np.savez(
            vdr_dataset_file_path,
            TIMESTAMP=test_data[TIMESTAMP].detach().cpu().numpy(),
            FILTER_NAV_ROTATION_MATRIX=test_data[FILTER_NAV_ROTATION_MATRIX].detach().cpu().numpy(),
            FILTER_NAV_VELOCITY=test_data[FILTER_NAV_VELOCITY].detach().cpu().numpy(),
            FILTER_NAV_POSITION=test_data[FILTER_NAV_POSITION].detach().cpu().numpy(),
            FILTER_IMU_GYROSCOPE_BIAS=test_data[FILTER_IMU_GYROSCOPE_BIAS].detach().cpu().numpy(),
            FILTER_IMU_ACCELEROMETER_BIAS=test_data[FILTER_IMU_ACCELEROMETER_BIAS].detach().cpu().numpy(),
            FILTER_CAR_ROTATION_MATRIX=test_data[FILTER_CAR_ROTATION_MATRIX].detach().cpu().numpy(),
            FILTER_CAR_POSITION=test_data[FILTER_CAR_POSITION].detach().cpu().numpy(),
            FILTER_STATE_COVARIANCE=test_data[FILTER_STATE_COVARIANCE].detach().cpu().numpy(),
            FILTER_NOISE_COVARIANCE=test_data[FILTER_NOISE_COVARIANCE].detach().cpu().numpy(),
            FILTER_MEASUREMENT_COVARIANCE=test_data[FILTER_MEASUREMENT_COVARIANCE].detach().cpu().numpy()
        )
        logger = get_logger()
        logger_str = 'Save file path {}'.format(vdr_dataset_file_path)
        logger.info(logger_str)

    def random_sample_sequence_index(self, train_random_sample_seq_len):
        sequence_head_index = 0
        sequence_tail_index = self.timestamp.shape[0]
        if train_random_sample_seq_len is not None:
            train_random_sample_seq_time_duration = train_random_sample_seq_len * self.sample_time_interval
            train_random_sample_seq_time_duration_ceil = math.ceil(train_random_sample_seq_time_duration)
            sequence_timestamp_head = np.random.randint(
                self.timestamp[0],
                self.timestamp[-1] - train_random_sample_seq_time_duration_ceil + 1
            )
            sequence_head_index_raw = (sequence_timestamp_head - self.timestamp[0]) * self.sample_rate
            sequence_head_index = int(sequence_head_index_raw)
            sequence_tail_index = sequence_head_index + train_random_sample_seq_len

        return sequence_head_index, sequence_tail_index

    def prepare_test_sequence(self):
        random_sampled_sequence_dic = {
            SAMPLE_INDEX: [0, self.timestamp.shape[0]],
            TIMESTAMP: self.timestamp,
            PHONE_MEASUREMENT_GYROSCOPE: self.phone_measurement_gyroscope,
            PHONE_MEASUREMENT_ACCELEROMETER: self.phone_measurement_accelerometer,
            PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD: self.pseudo_measurement_car_velocity_forward,
            GROUND_TRUTH_NAV_ROTATION_MATRIX: self.ground_truth_nav_rotation_matrix,
            GROUND_TRUTH_NAV_VELOCITY: self.ground_truth_nav_velocity,
            GROUND_TRUTH_NAV_POSITION: self.ground_truth_nav_position,
            PHONE_MEASUREMENT_NORMALIZED: self.phone_measurement_normalized
        }
        # slice_index = np.arange(4000, self.timestamp.shape[0])
        # random_sampled_sequence_dic = {
        #     SAMPLE_INDEX: slice_index,
        #     TIMESTAMP: self.timestamp[slice_index],
        #     PHONE_MEASUREMENT_GYROSCOPE: self.phone_measurement_gyroscope[slice_index, :],
        #     PHONE_MEASUREMENT_ACCELEROMETER: self.phone_measurement_accelerometer[slice_index, :],
        #     PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD: self.pseudo_measurement_car_velocity_forward[slice_index],
        #     GROUND_TRUTH_NAV_ROTATION_MATRIX: self.ground_truth_nav_rotation_matrix[slice_index, :, :],
        #     GROUND_TRUTH_NAV_VELOCITY: self.ground_truth_nav_velocity[slice_index, :],
        #     GROUND_TRUTH_NAV_POSITION: self.ground_truth_nav_position[slice_index, :],
        #     PHONE_MEASUREMENT_NORMALIZED: self.phone_measurement_normalized[slice_index, :]
        # }
        return SequenceDataset(random_sampled_sequence_dic)

    def prepare_ground_truth_relative_translation(self):
        collection_relative_translation = [[], [], []]

        odometer = np.zeros(self.timestamp.shape[0])
        delta_nav_position = self.ground_truth_nav_position[1:] - self.ground_truth_nav_position[:-1]
        delta_nav_distance = np.linalg.norm(delta_nav_position, axis=1)
        odometer[1:] = delta_nav_distance.cumsum()
        step_size = self.sample_rate
        step_len = int(self.timestamp.shape[0] / step_size) - 1
        for k in range(0, step_len):
            sequence_head_index = int(k * step_size)
            for sequence_length in TrackDataset.RELATIVE_SEQUENCE_LENGTH:
                if sequence_length + odometer[sequence_head_index] > odometer[-1]:
                    continue
                sequence_shift = np.searchsorted(odometer[sequence_head_index:], odometer[sequence_head_index] + sequence_length)
                sequence_tail_index = sequence_head_index + sequence_shift

                collection_relative_translation[0].append(sequence_head_index)
                collection_relative_translation[1].append(sequence_tail_index)

        relative_translation_head_index = collection_relative_translation[0]
        relative_translation_tail_index = collection_relative_translation[1]
        delta_translation = self.ground_truth_nav_position[relative_translation_tail_index] - self.ground_truth_nav_position[relative_translation_head_index]
        delta_translation_expand = np.expand_dims(delta_translation, -1)
        relative_rotation = self.ground_truth_nav_rotation_matrix[relative_translation_head_index, :, :].transpose(0, 2, 1)
        relative_translation = np.matmul(relative_rotation, delta_translation_expand)
        collection_relative_translation[2] = relative_translation.squeeze()

        return collection_relative_translation

    def prepare_sample_relative_translation(self, sampled_sequence, predicted_sequence, device):
        sample_index = sampled_sequence.sample_index
        sample_len = sampled_sequence.timestamp.shape[0]

        relative_translation_head_index = torch.tensor(self.ground_truth_relative_translation[0]).clone().long() - sample_index[0]
        relative_translation_tail_index = torch.tensor(self.ground_truth_relative_translation[1]).clone().long() - sample_index[0]
        relative_translation = self.ground_truth_relative_translation[2]
        sample_relative_translation_index = torch.zeros(relative_translation_head_index.shape[0], dtype=torch.bool)
        sample_relative_translation_index[:] = True
        sample_relative_translation_index[relative_translation_head_index < 0] = False
        sample_relative_translation_index[relative_translation_tail_index >= sample_len] = False

        sampled_relative_translation_head_index = relative_translation_head_index[sample_relative_translation_index]
        sampled_relative_translation_tail_index = relative_translation_tail_index[sample_relative_translation_index]
        sampled_relative_translation = torch.tensor(relative_translation[sample_relative_translation_index]).clone().to(device)

        if len(sampled_relative_translation_head_index) == 0:
            return None, None
        else:
            predicted_nav_rotation_matrix = predicted_sequence[FILTER_NAV_ROTATION_MATRIX]
            predicted_nav_translation = predicted_sequence[FILTER_NAV_POSITION]
            predicted_delta_translation = predicted_nav_translation[sampled_relative_translation_tail_index] - predicted_nav_translation[sampled_relative_translation_head_index]
            predicted_relative_translation = predicted_nav_rotation_matrix[sampled_relative_translation_head_index].transpose(-1, -2).matmul(predicted_delta_translation.unsqueeze(-1)).squeeze()
            distance = sampled_relative_translation.norm(dim=1).unsqueeze(-1)
            return sampled_relative_translation.double() / distance.double(), predicted_relative_translation.double() / distance.double(),
