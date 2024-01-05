import math
import os.path
import time

import numpy as np
import torch
from torch import nn

from test_vdr import test_only
from utils.logs.log_utils import get_logger


def train_filter(args, train_datasets, test_dataset, model, loss_fn, optimizer, writer):

    train_epochs = args.epochs
    train_random_sample_seq_len = args.seq_len
    train_max_loss = args.max_loss
    train_max_grad_norm = args.max_grad_norm
    logger = get_logger()

    model.train()
    train_sum_loss_min = np.finfo(np.float64).max
    train_avg_loss_min = np.finfo(np.float64).max
    schedule_name = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

    train_log_file_path = os.path.join('.', 'experiments', 'train_log.log')
    with open(train_log_file_path, 'a') as file:
        new_log_head = "New train start at {}, schedule epochs {}".format(schedule_name, train_epochs)
        track_sequence = 'Epochs'
        for i, dataset_track in enumerate(train_datasets):
            track_sequence += ', T' + dataset_track.track_folder_name
        file.write("\n")
        file.write(new_log_head)
        file.write("\n")
        file.write(track_sequence)

    for train_epoch_num in range(train_epochs):

        model.train()
        optimizer.zero_grad()

        log_epoch = "| Epoch {}".format(train_epoch_num)

        # 配置模型保存的条件
        # 条件1：任意一条轨迹的loss更新最小值
        # 条件2：每轮epoch的有效loss求和更新最小值
        # 条件3：每轮epoch的有效loss平均更新最小值
        save_model_flag = False
        epoch_sum_loss = 0
        epoch_num_loss = 0
        log_file_epoch = '{:6d}'.format(train_epoch_num)

        for i, dataset_track in enumerate(train_datasets):
            #
            dataset = train_datasets[i]
            log_track = "{} | Track {}".format(log_epoch, dataset.get_track_name())

            for j in range(8):
                log_subtrack = "{} | Sample {}".format(log_track, j)
                sampled_sequence = dataset_track.prepare_random_sample_sequence(train_random_sample_seq_len)
                #

                head_time = time.time()
                predicted_sequence = model(sampled_sequence)
                tail_time = time.time()
                delta_time = tail_time - head_time
                #
                ground_truth_relative_translation, predicted_relative_translation = (
                    dataset_track.prepare_sample_relative_translation(sampled_sequence, predicted_sequence, args.device)
                )
                if ground_truth_relative_translation is None:
                    logger.warning('{} | Not have relative translation', log_subtrack)
                    log_file_epoch += ', {:6.3f}'.format(-1)
                else:
                    #
                    loss_sequence = loss_fn(ground_truth_relative_translation, predicted_relative_translation)

                    log_loss = '{} | Loss {:6.3f}'.format(log_subtrack, loss_sequence)
                    log_file_epoch += ', {:6.3f}'.format(loss_sequence)

                    if torch.isnan(loss_sequence):
                        logger.warning('{} | Nan loss', log_loss)
                        continue
                    elif loss_sequence > train_max_loss:
                        logger.warning('{} | Max loss', log_loss)
                        continue
                    else:
                        logger.info(log_loss)
                        epoch_sum_loss += loss_sequence
                        epoch_num_loss += 1

                    # 满足条件1保存本轮模型
                    if loss_sequence < dataset_track.train_loss_min:
                        dataset_track.train_loss_min = loss_sequence
                        if not save_model_flag:
                            save_model_flag = True

        with open(train_log_file_path, 'a') as file:
            file.write("\n")
            file.write(log_file_epoch)

        log_total_loss = "{} | Total loss {:.3f}".format(log_epoch, epoch_sum_loss)
        if epoch_sum_loss == 0:
            logger.warning('{} | Zero loss', log_total_loss)
        else:
            epoch_sum_loss.backward()
            # loss_datasets.cuda().backward()
            g_norm = nn.utils.clip_grad_norm_(model.parameters(), train_max_grad_norm)
            log_total_norm = "{} | Total norm {:.3f}".format(log_total_loss, g_norm)
            if np.isnan(g_norm.cpu()) or g_norm.cpu() > 3 * train_max_grad_norm:
                logger.warning('{} | Max norm', log_total_norm)
            else:
                # 满足条件2保存本轮模型
                if epoch_sum_loss < train_sum_loss_min:
                    train_sum_loss_min = epoch_sum_loss
                    if not save_model_flag:
                        save_model_flag = True

                # 满足条件3保存本轮模型
                epoch_avg_loss = epoch_sum_loss / epoch_num_loss
                if epoch_avg_loss < train_avg_loss_min:
                    train_avg_loss_min = epoch_avg_loss
                    if not save_model_flag:
                        save_model_flag = True

                logger.info(log_total_norm)
                optimizer.step()
                writer.add_scalar("Loss/train", epoch_sum_loss, train_epoch_num)

        if save_model_flag:
            file_name_loss = math.floor(epoch_sum_loss * 1e6)
            file_name = "filter_schedule_{}_epoch_{}_{}_loss_{}.p".format(
                schedule_name, train_epoch_num, train_epochs, file_name_loss
            )
            model.save_filter(file_name)

        if train_epoch_num % 10 == 0:
            test_loss = test_only(args, test_dataset, model, loss_fn)
            writer.add_scalar("Loss/test", test_loss, train_epoch_num)

