import os

import torch


class ScriptArgs:
    # datasets_base_folder_path = "E:\\DoctorRelated\\20230410重庆VDR数据采集"
    reference_folder_path = os.path.abspath('.')
    datasets_base_folder_path = os.path.normpath(os.path.join(reference_folder_path, 'datas'))

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_filter = True
    epochs = 500
    seq_len = 12000
    max_loss = 20
    max_grad_norm = 50
    continue_training = True
    model_file_name = "filter_schedule_20231025_120725_epoch_360_500_loss_27985337.p"

    test_filter = False

    result_filter = False


def load_args(args_input_mode):
    if args_input_mode == 1:
        return load_terminal_args()
    else:
        return load_default_args()


def load_default_args():
    return ScriptArgs()


def load_terminal_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--max_loss', type=float, default=2e1)
    parser.add_argument('--max_grad_norm', type=float, default=1e0)

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()
    return args
