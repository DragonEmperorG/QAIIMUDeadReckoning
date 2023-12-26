import os

import numpy as np

from datasets.parking_dataset import ParkingDataset
from utils.files.preprocessing_colleted_file_util import load_preprocessing_collected_phone_imu_data, get_track_phone_folder_path

DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL = 'TRAIN'
DEEPODO_DATALOADER_CONFIG_TEST_LABEL = 'TEST'


def load(root_folder_path, args):
    train_list = []
    train_preprocessor = []
    test_list = []

    arg_train_test_config = args.datasets_train_test_config
    for i in range(len(arg_train_test_config)):

        dataset_collector_phone_folder_path = get_track_phone_folder_path(root_folder_path, arg_train_test_config[i])

        if DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL in arg_train_test_config[i][0]:
            train_list.append(dataset_collector_phone_folder_path)
            temp_phone_imu_data = load_preprocessing_collected_phone_imu_data(dataset_collector_phone_folder_path)
            train_preprocessor.append(temp_phone_imu_data)

        if DEEPODO_DATALOADER_CONFIG_TEST_LABEL in arg_train_test_config[i][0]:
            test_list.append(dataset_collector_phone_folder_path)

    normalize_factors = {
        'mean': 0,
        'std': 1,
    }

    if len(train_preprocessor) != 0:
        train_preprocessor_concatenate = np.concatenate(train_preprocessor)
        train_preprocessor_mean = np.mean(train_preprocessor_concatenate, axis=0)
        train_preprocessor_std = np.std(train_preprocessor_concatenate, axis=0)
        normalize_factors = {
            'mean': train_preprocessor_mean,
            'std': train_preprocessor_std,
        }

    train_dataset = ParkingDataset(train_list, normalize_factors)
    test_dataset = ParkingDataset(test_list, normalize_factors)

    return train_dataset, test_dataset
