import os
import shutil

from datasets.parking_dataloader import DEEPODO_DATALOADER_CONFIG
from datasets.track_dataset import TrackDataset
from main_vdr import ARGS_INPUT_MODE
from utils.arg_utils import load_args
from utils.files.preprocessing_colleted_file_util import get_preprocessing_collected_phone_imu_data_file_path

if __name__ == '__main__':

    loaded_args = load_args(ARGS_INPUT_MODE)

    reference_folder_path = os.path.abspath('.')
    move_to_data_base_folder_path = os.path.normpath(os.path.join(reference_folder_path, '..', 'datas'))
    #
    datasets_preprocess_reorganized_folder_name = "Reorganized"
    for i in range(len(DEEPODO_DATALOADER_CONFIG)):
        move_from_data_folder_path = os.path.join(
            loaded_args.datasets_base_folder_path,
            DEEPODO_DATALOADER_CONFIG[i][1],
            datasets_preprocess_reorganized_folder_name,
            DEEPODO_DATALOADER_CONFIG[i][2],
            DEEPODO_DATALOADER_CONFIG[i][3],
        )

        move_to_data_folder_path = os.path.join(
            move_to_data_base_folder_path,
            DEEPODO_DATALOADER_CONFIG[i][1],
            datasets_preprocess_reorganized_folder_name,
            DEEPODO_DATALOADER_CONFIG[i][2],
            DEEPODO_DATALOADER_CONFIG[i][3],
        )

        move_from_imu_data_file_path = get_preprocessing_collected_phone_imu_data_file_path(move_from_data_folder_path)
        move_to_imu_data_file_path = get_preprocessing_collected_phone_imu_data_file_path(move_to_data_folder_path)
        if not os.path.isfile(move_to_imu_data_file_path):
            move_to_imu_data_folder_path = os.path.dirname(move_to_imu_data_file_path)
            if not os.path.isdir(move_to_imu_data_folder_path):
                os.makedirs(move_to_imu_data_folder_path)
            shutil.move(move_from_imu_data_file_path, move_to_imu_data_file_path)

        move_from_vdr_dataset_file_path = os.path.join(move_from_data_folder_path, TrackDataset.VDR_DATASET_FOLDER_NAME, TrackDataset.VDR_DATASET_FILE_NAME)
        move_to_vdr_dataset_file_path = os.path.join(move_to_data_folder_path, TrackDataset.VDR_DATASET_FOLDER_NAME, TrackDataset.VDR_DATASET_FILE_NAME)
        if not os.path.isfile(move_to_vdr_dataset_file_path):
            move_to_vdr_dataset_folder_path = os.path.dirname(move_to_vdr_dataset_file_path)
            if not os.path.isdir(move_to_vdr_dataset_folder_path):
                os.makedirs(move_to_vdr_dataset_folder_path)
            shutil.move(move_from_vdr_dataset_file_path, move_to_vdr_dataset_file_path)

