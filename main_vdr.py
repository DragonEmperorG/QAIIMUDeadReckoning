import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.parking_dataloader import load
from graphs.models.invariant_extended_kalman_filter import InvariantExtendedKalmanFilter
from result_vdr import result_filter
from test_vdr import test_filter
from train_vdr import train_filter
from utils.arg_utils import load_args
from utils.logs.log_utils import init_logger, get_logger

ARGS_INPUT_MODE = 0


def main(args):
    logger_main = get_logger()

    layout = {
        "SDC2023": {
            "loss": ["Multiline", ["Loss/train", "Loss/test"]],
        },
    }
    writer = SummaryWriter()
    writer.add_custom_scalars(layout)

    train_dataset, test_dataset = load(args.datasets_base_folder_path, args)

    model = InvariantExtendedKalmanFilter(args.device)

    criterion = torch.nn.MSELoss(reduction="sum")

    if args.train_filter:
        logger_main.info("Start running train task")
        if args.continue_training:
            model.load_filter(args.model_file_name)
        model.to(args.device)

        learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_filter(args, train_dataset, test_dataset, model, criterion, optimizer, writer)

    if args.test_filter:
        logger_main.info("Start running test task")
        model.load_filter(args.model_file_name)
        model.to(args.device)

        test_filter(args, test_dataset, model, criterion)

    if args.result_filter:
        logger_main.info("Start running result task")

        result_filter(args)

    logger_main.info("Finished task!")


if __name__ == '__main__':
    #
    init_logger()
    logger = get_logger()

    loaded_args = load_args(ARGS_INPUT_MODE)
    logger.info("Loaded task configuration")
    if loaded_args.device == "cuda":
        cuda_device_count = torch.cuda.device_count()
        cuda_current_device = torch.cuda.current_device()
        cuda_current_device_name = torch.cuda.get_device_name(cuda_current_device)
        logger.info("Device: {}:{} | {}, total {}",
                    loaded_args.device,
                    cuda_current_device,
                    cuda_current_device_name,
                    cuda_device_count
                    )
    else:
        logger.info("Device: {}", loaded_args.device)

    main(loaded_args)
