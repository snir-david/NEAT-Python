import numpy as np
import os.path
import os
from pathlib import Path


def rgb_to_gray_scale(data, size_in_pixels=1024):
    gray = np.zeros(shape=(data.shape[0], size_in_pixels))
    for idx, arr in enumerate(data):
        new_pic = []
        for i in range(0, size_in_pixels, 1):
            # TODO maybe use another formula? not just mean?
            new_pic.append((arr[i] + arr[i + 1 * size_in_pixels] + arr[i + 2 * size_in_pixels]) / 3.)
        gray[idx] = np.array(new_pic)
    return gray


class Data:

    def __init__(self):
        self.train_y = []
        self.train_x = []
        self.test_y = []
        self.test_x = []
        self.validation_y = []
        self.validation_x = []

    # create 3 datasets of train, test and validation
    def set_datasets(self, train_path, test_path, validate_path, override_old_data=False, compress_input=True):
        self.set_one_dataset(train_path, 'train', override_old_data, compress_input)
        self.set_one_dataset(test_path, 'test', override_old_data, compress_input)
        self.set_one_dataset(validate_path, 'validate', override_old_data, compress_input)

    # create a dataset from given path of comma sep file
    def set_one_dataset(self, path, data_type, override_old_data=False, compress_input=True):
        if data_type != 'train' and data_type != 'test' and data_type != 'validate':
            print('error: invalid data type. valid options: train, test, validate')
            return
        numpy_file = Path(f'datasets/{data_type}_data.npy')

        # create numpy file if doesn't exist or override flag is True.
        if override_old_data is True or not numpy_file.is_file():
            if not numpy_file.is_file() and override_old_data is False:
                print(f'error: {data_type} data numpy file was not found, creating new...')
            dataset = np.genfromtxt(path, delimiter=',')
            np.save(f'datasets/{data_type}_data', dataset)

        # load numpy file for better run time.
        dataset = np.load(f'datasets/{data_type}_data.npy')

        # get the first column with the labels
        # remove the first column
        if data_type == 'train':
            self.train_y = dataset[:, 0] - 1
            self.train_x = np.delete(dataset, 0, 1)
        elif data_type == 'test':
            self.test_y = dataset[:, 0] - 1
            self.test_x = np.delete(dataset, 0, 1)
        elif data_type == 'validate':
            self.validation_y = dataset[:, 0] - 1
            self.validation_x = np.delete(dataset, 0, 1)

        # RGB to grey scale
        if compress_input:
            if data_type == 'train':
                self.train_x = rgb_to_gray_scale(self.train_x)
            elif data_type == 'test':
                self.test_x = rgb_to_gray_scale(self.test_x)
            elif data_type == 'validate':
                self.validation_x = rgb_to_gray_scale(self.validation_x)

        if data_type == 'train':
            self.train_x = np.array(self.train_x)
        elif data_type == 'test':
            self.test_x = np.array(self.test_x)
        elif data_type == 'validate':
            self.validation_x = np.array(self.validation_x)

        # return self.train_x, self.train_y
