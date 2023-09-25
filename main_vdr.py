import torch

from datasets.parking_dataloader import load
from graphs.models.invariant_extended_kalman_filter import InvariantExtendedKalmanFilter
from train_vdr import train_filter
from utils.arg_utils import load_args
from utils.logs.log_utils import init_logger, get_logger

ARGS_INPUT_MODE = 0


def main(args):
    train_dataset, test_dataset = load(args.datasets_base_folder_path)

    model = InvariantExtendedKalmanFilter(args.device)
    if args.continue_training:
        model.load_filter(args.model_file_name)
    model.to(args.device)

    criterion = torch.nn.MSELoss(reduction="sum")

    learn_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    train_filter(args, train_dataset, model, criterion, optimizer)
    logger_main = get_logger()
    logger_main.info("Done!")


if __name__ == '__main__':
    #
    init_logger()
    logger = get_logger()
    logger.info("An info")

    loaded_args = load_args(ARGS_INPUT_MODE)

    main(loaded_args)
