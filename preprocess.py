# TODO: import the yahoo finance data
# TODO: process the data
# TODO: save the data in this directory (so that we don't need to run preprocess every time we try to use the model)
# can use np's save function, pickle's save function, ....

# for example, save a dictionary {'train': ...,  'test': ...}
# and value of 'train' could be a [num_data_points, state_size] matrix, where each row represents one data point,
# containing info like opening price, closing price, technical indicators, ...
# similar for 'test'

# or to give us more flexibility later, 'train' could also be a nested dict, 'train' --> year --> month --> day --> data
