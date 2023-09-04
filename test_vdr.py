import math
import os

import numpy as np
import torch


def test_filter(dataloader, model, loss_fn1, optimizer1, t1, epoch, time1, min_loss1, writer1, device):
    size = len(dataloader.dataset)
    model.train_filter()
    optimizer1.zero_grad()
    train_min_loss = min_loss1

    for batch, batch_data in enumerate(dataloader):
        batch_loss = torch.tensor(.0)
        for sensor_data_numpy, ground_truth_numpy in batch_data:
            sensor_data_tensor, ground_truth_tensor = torch.tensor(sensor_data_numpy), torch.tensor(ground_truth_numpy)
            sensor_data, ground_truth = sensor_data_tensor.to(device), ground_truth_tensor.to(device)
            input_float = sensor_data.float()
            output_float = ground_truth.float()

            # input_float_len = len(input_float)
            # random_sample0 = int(np.random.randint(0, input_float_len - 60))
            # random_sample1 = random_sample0 + 60
            # random_sample_input_float = input_float[random_sample0:random_sample1, :]
            # random_sample_output_float = output_float[random_sample0:random_sample1, :]

            random_sample_input_float = input_float
            random_sample_output_float = output_float

            # Compute prediction error
            pred = model(random_sample_input_float)
            loss = loss_fn1(pred, random_sample_output_float)
            batch_loss += loss

        # Backpropagation
        batch_loss = batch_loss / len(batch_data)
        batch_loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        writer1.add_scalar("Loss/train", batch_loss, t1)
        batch_loss, current = batch_loss.item(), (batch + 1) * len(batch_data)

        is_save_model = False
        if train_min_loss == -1:
            train_min_loss = batch_loss
            is_save_model = True
        else:
            if batch_loss < train_min_loss:
                train_min_loss = batch_loss
                is_save_model = True

        if is_save_model:
            file_name_loss = math.floor(batch_loss * 1e6)
            file_name = "model_deepodo_wang_train_schedule_{}_epoch_{}_{}_batch_{}_{}_Loss_{}.p".format(time1,
                                                                                                        t1, epoch, current,
                                                                                                        size,
                                                                                                        file_name_loss)
            file_path1 = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', file_name)
            torch.save(model.state_dict(), file_path1)
            file_path2 = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', 'model_deepodo_wang.p')
            torch.save(model.state_dict(), file_path2)
        print(f"loss: {batch_loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_min_loss
