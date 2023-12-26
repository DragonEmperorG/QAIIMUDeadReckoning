from utils.logs.log_utils import get_logger


def test_filter(args, datasets, model, loss_fn):

    logger_test = get_logger()

    model.eval()

    epoch_sum_loss = 0

    for i, dataset_track in enumerate(datasets):
        #
        dataset = datasets[i]
        log_track = "| Track {}".format(dataset.get_track_name())

        test_sequence = dataset_track.prepare_test_sequence()
        #
        predicted_sequence = model(test_sequence)

        dataset_track.save_test_data(predicted_sequence)

        ground_truth_relative_translation, predicted_relative_translation = dataset_track.prepare_sample_relative_translation(test_sequence, predicted_sequence, args.device)

        if ground_truth_relative_translation is None:
            logger_test.warning('{} | Not have relative translation', log_track)
        else:
            #
            loss_sequence = loss_fn(ground_truth_relative_translation, predicted_relative_translation)

            log_loss = '{} | Loss {:6.3f}'.format(log_track, loss_sequence)
            logger_test.info(log_loss)

            epoch_sum_loss = epoch_sum_loss + loss_sequence

    return epoch_sum_loss


def test_only(args, datasets, model, loss_fn):

    logger_test = get_logger()

    model.eval()

    epoch_sum_loss = 0

    for i, dataset_track in enumerate(datasets):
        #
        dataset = datasets[i]
        log_track = "| Track {}".format(dataset.get_track_name())

        test_sequence = dataset_track.prepare_test_sequence()
        #
        predicted_sequence = model(test_sequence)

        ground_truth_relative_translation, predicted_relative_translation = dataset_track.prepare_sample_relative_translation(test_sequence, predicted_sequence, args.device)

        if ground_truth_relative_translation is None:
            logger_test.warning('{} | Not have relative translation', log_track)
        else:
            #
            loss_sequence = loss_fn(ground_truth_relative_translation, predicted_relative_translation)

            log_loss = '{} | Loss {:6.3f}'.format(log_track, loss_sequence)
            logger_test.info(log_loss)

            epoch_sum_loss = epoch_sum_loss + loss_sequence

    return epoch_sum_loss
