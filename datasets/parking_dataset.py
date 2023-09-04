from torch.utils.data import Dataset

from datasets.track_dataset import TrackDataset


class ParkingDataset(Dataset):

    def __init__(self, file_path_list, normalize_factors):
        self.track_len = len(file_path_list)
        self.track_list = []
        for i in range(self.track_len):
            track_dataset = TrackDataset(file_path_list[i], normalize_factors)
            self.track_list.append(track_dataset)

    def __len__(self):
        return self.track_len

    def __getitem__(self, idx):
        return self.track_list[idx]
