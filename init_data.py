import numpy as np
import glob
import pandas as pd
from augmentations import full_augment
import sklearn

freq_suffices = ["_8Hz", "_14Hz", "_28Hz"]


def cut(recording, cut_start, cut_length):
    recording = recording[cut_start:cut_start+cut_length]
    return recording


def scale(recording):
    transformer = sklearn.preprocessing.RobustScaler().fit(recording)
    scaled_recording = transformer.transform(recording)
    return scaled_recording


def one_hot_encode(number, dimensions):
    vector = np.zeros(dimensions)
    vector[number] = 1
    return vector


def load_data(data_dir, cut_start, cut_length):
    labels = np.empty((0, 3))
    recordings = np.empty((0, cut_length, 8))

    all_train_files = glob.glob(data_dir + "/*.csv")
    print(all_train_files)
    for index, freq_suffix in enumerate(freq_suffices):
        current_freq_files = [s for s in all_train_files if freq_suffix in s]
        for current_freq_file in current_freq_files:
            recording = pd.read_csv(current_freq_file, header=None)
            recording = cut(recording, cut_start, cut_length)
            recording = scale(recording)
            recordings = np.append(recordings, [recording], axis=0)

            label = one_hot_encode(index, len(freq_suffices))
            labels = np.append(labels, [label], axis=0)
    return recordings, labels


def init_data(conf):
    x_train, y_train = load_data(conf["train_data_dir"],
                                 cut_start=conf["train_cut_start"],
                                 cut_length=conf["train_cut_length"])

    x_train_aug, y_train_aug = full_augment(x_train, y_train, conf["aug_multiplier"])

    x_test, y_test = load_data(conf["test_data_dir"],
                               cut_start=conf["test_cut_start"],
                               cut_length=conf["test_cut_length"])

    train_test_data = {"x_train": x_train_aug, "y_train": y_train_aug, "x_test": x_test, "y_test": y_test}
    return train_test_data
