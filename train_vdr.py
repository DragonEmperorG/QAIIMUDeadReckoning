import math
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

    model.train()
    train_min_loss = np.finfo(np.float64).max
    schedule_name = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

    for train_epoch_num in range(train_epochs):

        loss_datasets = 0
        optimizer.zero_grad()

        log_epoch = "| Epoch {}".format(train_epoch_num)
        for i, dataset_track in enumerate(datasets):
            #
            dataset = datasets[i]
            log_track = "{} | Track {}".format(log_epoch, dataset.get_track_name())

            sampled_sequence = dataset_track.prepare_random_sample_sequence(train_random_sample_seq_len)
            #
            head_time = time.time()
            predicted_sequence = model(sampled_sequence)
            tail_time = time.time()
            delta_time = tail_time - head_time
            #
            ground_truth_relative_translation, predicted_relative_translation = dataset_track.prepare_sample_relative_translation(sampled_sequence, predicted_sequence, args.device)
            if ground_truth_relative_translation is None:
                logger.warning('{} | Not have relative translation', log_track)
            else:
                #
                loss_sequence = loss_fn(ground_truth_relative_translation, predicted_relative_translation)
                log_loss = "{} | Loss {:.3f}".format(log_track, loss_sequence)
                if torch.isnan(loss_sequence):
                    logger.warning('{} | Nan loss', log_loss)
                    continue
                elif loss_sequence > train_max_loss:
                    logger.warning('{} | Max loss', log_loss)
                    continue
                else:
                    logger.info(log_loss)
                    loss_datasets += loss_sequence

        log_total_loss = "{} | Total loss {:.3f}".format(log_epoch, loss_datasets)
        if loss_datasets == 0:
            logger.warning('{} | Zero loss', log_total_loss)
        else:
            # loss_datasets.backward()
            loss_datasets.cuda().backward()
            g_norm = nn.utils.clip_grad_norm_(model.parameters(), train_max_grad_norm)
            log_total_norm = "{} | Total norm {:.3f}".format(log_total_loss, g_norm)
            if np.isnan(g_norm) or g_norm > 3 * train_max_grad_norm:
                logger.warning('{} | Max norm', log_total_norm)
            else:
                if loss_datasets < train_min_loss:
                    train_min_loss = loss_datasets
                    file_name_loss = math.floor(train_min_loss * 1e6)
                    file_name = "filter_schedule_{}_epoch_{}_{}_loss_{}.p".format(schedule_name, train_epoch_num, train_epochs, file_name_loss)
                    model.save_filter(file_name)

                logger.info(log_total_norm)
                optimizer.step()

