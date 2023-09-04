import scipy


def resample_data(input_time, data, output_time):
    func = scipy.interpolate.interp1d(input_time, data, axis=0, fill_value='extrapolate')
    output_data = func(output_time)
    return output_data
