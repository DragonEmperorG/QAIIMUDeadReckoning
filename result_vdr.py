import os

import numpy as np
from matplotlib import pyplot as plt

from datasets.track_dataset import TrackDataset
from utils.constant_utils import GROUND_TRUTH_NAV_POSITION, FILTER_NAV_POSITION
from utils.files.preprocessing_colleted_file_util import get_track_phone_folder_path
from utils.logs.log_utils import get_logger


def result_filter(args):

    root_folder_path = args.datasets_base_folder_path

    logger_result = get_logger()

    result_data_config = [
        ['RESULT', '2023_04_10', '0018', 'HUAWEI_Mate30'],
    ]

    for i in range(len(result_data_config)):

        dataset_collector_phone_folder_path = get_track_phone_folder_path(root_folder_path, result_data_config[i])

        dataset_vdr_folder_path = os.path.join(
            dataset_collector_phone_folder_path,
            TrackDataset.VDR_DATASET_FOLDER_NAME
        )

        dataset_vdr_train_file_path = os.path.join(
            dataset_vdr_folder_path,
            TrackDataset.VDR_DATASET_FILE_NAME
        )

        if not os.path.isfile(dataset_vdr_train_file_path):
            logger_result.error("Not have train file")
            continue

        dataset_train_dic = np.load(dataset_vdr_train_file_path)
        dataset_train_position = dataset_train_dic[GROUND_TRUTH_NAV_POSITION]

        dataset_vdr_result_file_path = os.path.join(
            dataset_vdr_folder_path,
            TrackDataset.VDR_DATASET_RESULT_FILE_NAME
        )

        if not os.path.isfile(dataset_vdr_result_file_path):
            logger_result.error("Not have result file")
            continue

        dataset_result_dic = np.load(dataset_vdr_result_file_path)
        dataset_result_position = dataset_result_dic[FILTER_NAV_POSITION]

        fig, ax = plt.subplots(figsize=(9 / 2.54, 6.75 / 2.54), dpi=600)
        ax.plot(dataset_train_position[:, 0], dataset_train_position[:, 1], color="red", linestyle="-")
        ax.plot(dataset_result_position[:, 0], dataset_result_position[:, 1], color="blue", linestyle="--")
        ax.axis('equal')
        ax.set(xlabel='x (m)', ylabel='y (m)')
        ax.legend(['Ground Truth', 'Proposed'], loc='upper right')

        plt.tight_layout()
        # fig.savefig(os.path.join(os.path.abspath('.'), "DeepOdoPredictData.png"))
        plt.show()
