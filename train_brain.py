import numpy as np
import pandas as pd
import glob
import sklearn.preprocessing
import keras
import gc
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from dynamic_net import dynamic_net
from augmentations import full_augment
from sklearn.model_selection import StratifiedKFold

freq_suffices = ["_8Hz", "_14Hz", "_28Hz"]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
np.random.seed(0)


def pre_cut(recording, cut_start, cut_length):
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


def load_data(data_dir, pre_cut_start, pre_cut_length):
    labels = np.empty((0, 3))
    recordings = np.empty((0, pre_cut_length, 8))

    all_train_files = glob.glob(data_dir + "/*.csv")
    print(all_train_files)
    for index, freq_suffix in enumerate(freq_suffices):
        current_freq_files = [s for s in all_train_files if freq_suffix in s]
        for current_freq_file in current_freq_files:
            recording = pd.read_csv(current_freq_file, header=None)
            recording = pre_cut(recording, pre_cut_start, pre_cut_length)
            recording = scale(recording)
            recordings = np.append(recordings, [recording], axis=0)

            label = one_hot_encode(index, len(freq_suffices))
            labels = np.append(labels, [label], axis=0)
    return recordings, labels


def train(x_train, y_train, model):
    model.fit(x_train, y_train, shuffle=True, epochs=100, batch_size=1, validation_split=0.2, verbose=1)


def test(x_test, y_test, model):
    score = model.evaluate(x_test, y_test, batch_size=48)
    return score


def init_data(train_data_dir, test_data_dir, multiplier):
    my_x_train, my_y_train = load_data(train_data_dir, pre_cut_start=0, pre_cut_length=6000)
    x_train_aug, y_train_aug = full_augment(my_x_train, my_y_train, multiplier)

    my_x_test, my_y_test = load_data(test_data_dir, pre_cut_start=1000, pre_cut_length=5000)

    return x_train_aug, y_train_aug, my_x_test, my_y_test


def write_log_params(filepath, params):
    file_stream = open(filepath,"a+")
    file_stream.write("\nGenes: ")
    file_stream.write(str(params))
    file_stream.close()


def write_log_metrics(filepath, loss, acc):
    file_stream = open(filepath,"a+")
    file_stream.write("\nClassification Loss ")
    file_stream.write(loss)
    file_stream.write("\nClassification Accuracy ")
    file_stream.write(acc)
    file_stream.close()


def train_test_individual(params, x_train, y_train, x_test, y_test):
    lr = 0.0001
    feature_size = int(np.maximum(round(params[0]), 1))
    conv_layer_count = int(np.maximum(round(params[1]), 1))
    kernel_size = int(np.maximum(round(params[2]), 1))
    dilation_rate = int(np.maximum(round(params[3]), 1))
    dropout = params[4]
    filepath = "run_log.txt"

    my_params = {"lr": lr, "feature_size": feature_size, "conv_layer_count": conv_layer_count, "kernel_size": kernel_size,
                 "dilation_rate": dilation_rate, "dropout": dropout}

    write_log_params(filepath, my_params)
    print("Genes:")
    print(my_params)
    my_dynamic_net = dynamic_net(my_params)
    train(x_train, y_train, my_dynamic_net)

    classification_loss = test(x_test, y_test, my_dynamic_net)[0]
    classification_acc = test(x_test, y_test, my_dynamic_net)[1]

    print("Classification Loss " + str(classification_loss))
    print("Classification Accuracy " + str(classification_acc))
    
    write_log_metrics(filepath, str(classification_loss), str(classification_acc))

    keras.backend.clear_session()
    gc.collect()
    del my_dynamic_net

    return classification_loss
