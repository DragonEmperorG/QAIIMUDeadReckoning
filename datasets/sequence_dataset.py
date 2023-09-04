from utils.constant_utils import SAMPLE_INDEX, TIMESTAMP, PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD, \
    PHONE_MEASUREMENT_GYROSCOPE, PHONE_MEASUREMENT_ACCELEROMETER, PHONE_MEASUREMENT_NORMALIZED, \
    GROUND_TRUTH_NAV_ROTATION_MATRIX, GROUND_TRUTH_NAV_VELOCITY, GROUND_TRUTH_NAV_POSITION


class SequenceDataset:

    def __init__(self, sequence_dic):
        self.sample_index = sequence_dic[SAMPLE_INDEX]
        self.timestamp = sequence_dic[TIMESTAMP]
        self.phone_measurement_normalized = sequence_dic[PHONE_MEASUREMENT_NORMALIZED]
        self.phone_measurement_gyroscope = sequence_dic[PHONE_MEASUREMENT_GYROSCOPE]
        self.phone_measurement_accelerometer = sequence_dic[PHONE_MEASUREMENT_ACCELEROMETER]
        self.pseudo_measurement_car_velocity_forward = sequence_dic[PSEUDO_MEASUREMENT_CAR_VELOCITY_FORWARD]
        self.ground_truth_nav_rotation_matrix = sequence_dic[GROUND_TRUTH_NAV_ROTATION_MATRIX]
        self.ground_truth_nav_velocity = sequence_dic[GROUND_TRUTH_NAV_VELOCITY]
        self.ground_truth_nav_position = sequence_dic[GROUND_TRUTH_NAV_POSITION]


