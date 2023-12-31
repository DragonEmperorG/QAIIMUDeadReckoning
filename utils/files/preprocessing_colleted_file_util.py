import os

import numpy as np
import pandas as pd

from utils.logs.log_utils import get_logger

DATA_TIMESTAMP = 'DATA_TIMESTAMP'
PHONE_ACCELEROMETER_X = 'PHONE_ACCELEROMETER_X'
PHONE_ACCELEROMETER_Y = 'PHONE_ACCELEROMETER_Y'
PHONE_ACCELEROMETER_Z = 'PHONE_ACCELEROMETER_Z'
PHONE_GYROSCOPE_X = 'PHONE_GYROSCOPE_X'
PHONE_GYROSCOPE_Y = 'PHONE_GYROSCOPE_Y'
PHONE_GYROSCOPE_Z = 'PHONE_GYROSCOPE_Z'
GROUND_TRUTH_ACCELEROMETER_X = 'GROUND_TRUTH_ACCELEROMETER_X'
GROUND_TRUTH_ACCELEROMETER_Y = 'GROUND_TRUTH_ACCELEROMETER_Y'
GROUND_TRUTH_ACCELEROMETER_Z = 'GROUND_TRUTH_ACCELEROMETER_Z'
GROUND_TRUTH_GYROSCOPE_X = 'GROUND_TRUTH_GYROSCOPE_X'
GROUND_TRUTH_GYROSCOPE_Y = 'GROUND_TRUTH_GYROSCOPE_Y'
GROUND_TRUTH_GYROSCOPE_Z = 'GROUND_TRUTH_GYROSCOPE_Z'
GROUND_TRUTH_POSE_ROTATION_MATRIX_11 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_11'
GROUND_TRUTH_POSE_ROTATION_MATRIX_12 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_12'
GROUND_TRUTH_POSE_ROTATION_MATRIX_13 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_13'
GROUND_TRUTH_POSE_ROTATION_MATRIX_21 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_21'
GROUND_TRUTH_POSE_ROTATION_MATRIX_22 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_22'
GROUND_TRUTH_POSE_ROTATION_MATRIX_23 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_23'
GROUND_TRUTH_POSE_ROTATION_MATRIX_31 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_31'
GROUND_TRUTH_POSE_ROTATION_MATRIX_32 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_32'
GROUND_TRUTH_POSE_ROTATION_MATRIX_33 = 'GROUND_TRUTH_POSE_ROTATION_MATRIX_33'
GROUND_TRUTH_POSE_POSITION_X = 'GROUND_TRUTH_POSE_POSITION_X'
GROUND_TRUTH_POSE_POSITION_Y = 'GROUND_TRUTH_POSE_POSITION_Y'
GROUND_TRUTH_POSE_POSITION_Z = 'GROUND_TRUTH_POSE_POSITION_Z'
GROUND_TRUTH_VELOCITY_X = 'GROUND_TRUTH_VELOCITY_X'
GROUND_TRUTH_VELOCITY_Y = 'GROUND_TRUTH_VELOCITY_Y'
GROUND_TRUTH_VELOCITY_Z = 'GROUND_TRUTH_VELOCITY_Z'

PREPROCESSING_COLLECTED_FOLDER_NAME = 'dayZeroOClockAlign'
PREPROCESSING_COLLECTED_FILE_NAME = 'TrackSynchronized.csv'
PREPROCESSING_COLLECTED_FILE_DATA_NAME_LIST = [
    DATA_TIMESTAMP,
    PHONE_GYROSCOPE_X,
    PHONE_GYROSCOPE_Y,
    PHONE_GYROSCOPE_Z,
    PHONE_ACCELEROMETER_X,
    PHONE_ACCELEROMETER_Y,
    PHONE_ACCELEROMETER_Z,
    GROUND_TRUTH_GYROSCOPE_X,
    GROUND_TRUTH_GYROSCOPE_Y,
    GROUND_TRUTH_GYROSCOPE_Z,
    GROUND_TRUTH_ACCELEROMETER_X,
    GROUND_TRUTH_ACCELEROMETER_Y,
    GROUND_TRUTH_ACCELEROMETER_Z,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_11,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_12,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_13,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_21,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_22,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_23,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_31,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_32,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_33,
    GROUND_TRUTH_POSE_POSITION_X,
    GROUND_TRUTH_POSE_POSITION_Y,
    GROUND_TRUTH_POSE_POSITION_Z,
    GROUND_TRUTH_VELOCITY_X,
    GROUND_TRUTH_VELOCITY_Y,
    GROUND_TRUTH_VELOCITY_Z
]

PHONE_MEASUREMENT_GYROSCOPE_DATA_NAME_LIST = [
    PHONE_GYROSCOPE_X,
    PHONE_GYROSCOPE_Y,
    PHONE_GYROSCOPE_Z,
]

PHONE_MEASUREMENT_ACCELEROMETER_DATA_NAME_LIST = [
    PHONE_ACCELEROMETER_X,
    PHONE_ACCELEROMETER_Y,
    PHONE_ACCELEROMETER_Z,
]

GROUND_TRUTH_NAV_ROTATION_MATRIX_DATA_NAME_LIST = [
    GROUND_TRUTH_POSE_ROTATION_MATRIX_11,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_12,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_13,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_21,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_22,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_23,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_31,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_32,
    GROUND_TRUTH_POSE_ROTATION_MATRIX_33
]

GROUND_TRUTH_NAV_VELOCITY_DATA_NAME_LIST = [
    GROUND_TRUTH_VELOCITY_X,
    GROUND_TRUTH_VELOCITY_Y,
    GROUND_TRUTH_VELOCITY_Z
]

GROUND_TRUTH_NAV_POSITION_DATA_NAME_LIST = [
    GROUND_TRUTH_POSE_POSITION_X,
    GROUND_TRUTH_POSE_POSITION_Y,
    GROUND_TRUTH_POSE_POSITION_Z
]

PREPROCESSING_COLLECTED_PHONE_IMU_FILE_NAME = 'TrackSynchronizedPhoneIMU.npy'
# PHONE_MEASUREMENT_IMU_DATA_NAME_LIST = [
#     PHONE_GYROSCOPE_X,
#     PHONE_GYROSCOPE_Y,
#     PHONE_GYROSCOPE_Z,
#     PHONE_ACCELEROMETER_X,
#     PHONE_ACCELEROMETER_Y,
#     PHONE_ACCELEROMETER_Z
# ]


def get_track_phone_folder_path(root_folder_path, track_phone_folder_list):
    datasets_collector_data_time_folder_name = track_phone_folder_list[1]
    datasets_collector_data_time_folder_path = os.path.join(
        root_folder_path,
        datasets_collector_data_time_folder_name
    )

    datasets_preprocess_reorganized_folder_name = "Reorganized"
    datasets_preprocess_reorganized_folder_path = os.path.join(
        datasets_collector_data_time_folder_path,
        datasets_preprocess_reorganized_folder_name
    )

    datasets_collector_track_folder_name = track_phone_folder_list[2]
    datasets_collector_track_folder_path = os.path.join(
        datasets_preprocess_reorganized_folder_path,
        datasets_collector_track_folder_name
    )

    dataset_collector_phone_folder_name = track_phone_folder_list[3]
    dataset_collector_phone_folder_path = os.path.join(
        datasets_collector_track_folder_path,
        dataset_collector_phone_folder_name
    )

    return dataset_collector_phone_folder_path


def get_preprocessing_collected_phone_imu_data_file_path(folder_path):
    return os.path.join(
        folder_path,
        PREPROCESSING_COLLECTED_FOLDER_NAME,
        PREPROCESSING_COLLECTED_PHONE_IMU_FILE_NAME
    )


def load_preprocessing_collected_phone_imu_data(folder_path):
    preprocessing_collected_phone_imu_file_path = get_preprocessing_collected_phone_imu_data_file_path(folder_path)
    if os.path.isfile(preprocessing_collected_phone_imu_file_path):
        return np.load(preprocessing_collected_phone_imu_file_path)
    else:
        loaded_preprocessing_collected_data = load_preprocessing_collected_data(folder_path)
        phone_measurement_imu = np.concatenate(
            (loaded_preprocessing_collected_data[1], loaded_preprocessing_collected_data[2]),
            axis=1
        )
        np.save(preprocessing_collected_phone_imu_file_path, phone_measurement_imu)
        return phone_measurement_imu


def load_preprocessing_collected_data(folder_path):
    preprocessing_collected_file_path = os.path.join(folder_path, PREPROCESSING_COLLECTED_FOLDER_NAME,
                                                     PREPROCESSING_COLLECTED_FILE_NAME)
    if os.path.isfile(preprocessing_collected_file_path):
        raw_data = pd.read_csv(
            preprocessing_collected_file_path,
            header=None,
            names=PREPROCESSING_COLLECTED_FILE_DATA_NAME_LIST
        )

        timestamp = raw_data.loc[:, DATA_TIMESTAMP].to_numpy()
        phone_measurement_gyroscope = raw_data.loc[:, PHONE_MEASUREMENT_GYROSCOPE_DATA_NAME_LIST].to_numpy()
        phone_measurement_accelerometer = raw_data.loc[:, PHONE_MEASUREMENT_ACCELEROMETER_DATA_NAME_LIST].to_numpy()
        ground_truth_nav_rotation_matrix = raw_data.loc[:, GROUND_TRUTH_NAV_ROTATION_MATRIX_DATA_NAME_LIST].to_numpy()
        ground_truth_nav_velocity = raw_data.loc[:, GROUND_TRUTH_NAV_VELOCITY_DATA_NAME_LIST].to_numpy()
        ground_truth_raw_position = raw_data.loc[:, GROUND_TRUTH_NAV_POSITION_DATA_NAME_LIST].to_numpy()
        ground_truth_nav_position = ground_truth_raw_position - ground_truth_raw_position[0, :]
        return timestamp, phone_measurement_gyroscope, phone_measurement_accelerometer, ground_truth_nav_rotation_matrix, ground_truth_nav_velocity, ground_truth_nav_position
    else:
        logger = get_logger()
        logger.error("Not have file {}", preprocessing_collected_file_path)
        return ()
