import time

import numpy as np
import torch
from torch import nn

from datasets.track_dataset import TrackDataset
from utils.log_utils import get_logger


def train_filter(args, datasets, model, loss_fn, optimizer):

    train_epochs = args.epochs
    train_random_sample_seq_len = args.seq_len
    train_max_loss = args.max_loss
    train_max_grad_norm = args.max_grad_norm
    datasets_len = len(datasets)
    logger = get_logger()

    for train_epoch_num in range(train_epochs):

        loss_datasets = 0

        for i, dataset_track in enumerate(datasets):
            #
            sampled_sequence = dataset_track.prepare_random_sample_sequence(train_random_sample_seq_len)
            #
            head_time = time.time()
            predicted_sequence = model(sampled_sequence)
            tail_time = time.time()
            delta_time = tail_time - head_time
            logger.info('{}', delta_time)
            #
            ground_truth_relative_translation, predicted_relative_translation = dataset_track.prepare_sample_relative_translation(sampled_sequence, predicted_sequence)

            if ground_truth_relative_translation is None:
                logger.error('{}', delta_time)
            else:
                #
                loss_sequence = loss_fn(ground_truth_relative_translation, predicted_relative_translation)
                if torch.isnan(loss_sequence):
                    continue
                elif loss_sequence > train_max_loss:
                    continue
                else:
                    loss_datasets += loss_sequence

        if loss_datasets == 0:
            logger.error('{}', 1)
        else:
            loss_datasets.backward()
            # loss_train.cuda().backward()
            g_norm = nn.utils.clip_grad_norm_(model.parameters(), train_max_grad_norm)
            if np.isnan(g_norm) or g_norm > 3 * train_max_grad_norm:
                optimizer.zero_grad()

            else:
                optimizer.step()
                optimizer.zero_grad()

