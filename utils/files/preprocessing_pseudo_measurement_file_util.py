import math
import os

import numpy as np
import pandas as pd

from utils.logs.log_utils import get_logger

PREPROCESSING_PSEUDO_MEASUREMENT_FOLDER_NAME = 'DATASET_DEEPODO'
PREPROCESSING_PSEUDO_MEASUREMENT_FILE_NAME = 'DeepOdoPredictData.txt'
PREPROCESSING_PSEUDO_MEASUREMENT_TRAIN_FILE_NAME = 'DeepOdoTrainData.npy'


def load_preprocessing_pseudo_measurement_data(folder_path):
    preprocessing_pseudo_measurement_file_path = os.path.join(
        folder_path,
        PREPROCESSING_PSEUDO_MEASUREMENT_FOLDER_NAME,
        PREPROCESSING_PSEUDO_MEASUREMENT_FILE_NAME
    )
    if os.path.isfile(preprocessing_pseudo_measurement_file_path):
        raw_data = pd.read_csv(preprocessing_pseudo_measurement_file_path, header=None)
        pseudo_measurement_car_forward_velocity = raw_data.to_numpy()

        pseudo_measurement_train_data = load_preprocessing_pseudo_measurement_train_data(folder_path)
        pseudo_measurement_time_data = get_preprocessing_pseudo_measurement_data_time(pseudo_measurement_train_data[:, 0])

        return pseudo_measurement_time_data, pseudo_measurement_car_forward_velocity
    else:
        logger = get_logger()
        logger.error("Not have file {}", preprocessing_pseudo_measurement_file_path)
        return False


def load_preprocessing_pseudo_measurement_train_data(folder_path):
    dataset_deepodo_numpy_file_path = os.path.join(
        folder_path,
        PREPROCESSING_PSEUDO_MEASUREMENT_FOLDER_NAME,
        PREPROCESSING_PSEUDO_MEASUREMENT_TRAIN_FILE_NAME
    )
    loaded_deepodo_raw_data = []
    if os.path.isfile(dataset_deepodo_numpy_file_path):
        loaded_deepodo_raw_data = np.load(dataset_deepodo_numpy_file_path)
    else:
        logger = get_logger()
        logger.error("Not have file {}", dataset_deepodo_numpy_file_path)

    return loaded_deepodo_raw_data


def get_preprocessing_pseudo_measurement_data_time(raw_data_time):
    raw_data_length = len(raw_data_time)
    raw_data_head_time = raw_data_time[0]
    raw_data_tail_time = raw_data_time[raw_data_length - 1]
    pseudo_measurement_data_head_time = math.ceil(raw_data_head_time) + 1
    pseudo_measurement_data_tail_time = math.floor(raw_data_tail_time)
    return np.arange(pseudo_measurement_data_head_time, pseudo_measurement_data_tail_time + 1)
