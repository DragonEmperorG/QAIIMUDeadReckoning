import os

import numpy as np

from datasets.parking_dataset import ParkingDataset
from utils.files.preprocessing_colleted_file_util import load_preprocessing_collected_phone_imu_data, get_track_phone_folder_path

DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL = 'TRAIN'
DEEPODO_DATALOADER_CONFIG_TEST_LABEL = 'TEST'

DEEPODO_DATALOADER_CONFIG = [
    ['TRAIN', '2023_04_10', '0008', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0009', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0010', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0011', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0012', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0013', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0014', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0015', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0016', 'HUAWEI_Mate30'],
    ['TRAIN', '2023_04_10', '0017', 'HUAWEI_Mate30'],
    ['TEST', '2023_04_10', '0018', 'HUAWEI_Mate30'],
]


def load(root_folder_path):
    train_list = []
    train_preprocessor = []
    test_list = []

    for i in range(len(DEEPODO_DATALOADER_CONFIG)):

        dataset_collector_phone_folder_path = get_track_phone_folder_path(root_folder_path, DEEPODO_DATALOADER_CONFIG[i])

        if DEEPODO_DATALOADER_CONFIG[i][0] == DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL:
            train_list.append(dataset_collector_phone_folder_path)
            temp_phone_imu_data = load_preprocessing_collected_phone_imu_data(dataset_collector_phone_folder_path)
            train_preprocessor.append(temp_phone_imu_data)
        elif DEEPODO_DATALOADER_CONFIG[i][0] == DEEPODO_DATALOADER_CONFIG_TEST_LABEL:
            test_list.append(dataset_collector_phone_folder_path)

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
