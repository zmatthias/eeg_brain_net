import numpy as np
import keras as ks
import pandas as pd
import glob
import sklearn.preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import SGD

freq_suffices = ["_8Hz", "_14Hz", "_28Hz"]
train_data_dir = "/content/drive/My Drive/Colab Notebooks/data_matthias/training_data"
test_data_dir = "/content/drive/My Drive/Colab Notebooks/data_matthias/test_data"

cut_point_start = 1000
cut_point_end = 6000


def brain_net(learning_rate=0.01):
    model = Sequential()

    model.add(Conv1D(32, 1, activation='relu', input_shape=(5000, 8)))

    model.add(Conv1D(32, 5, dilation_rate=10, activation='relu'))
    model.add(Conv1D(32, 5, dilation_rate=10, activation='relu'))
    model.add(Conv1D(32, 5, dilation_rate=10, activation='relu'))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax'))

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def create_sinus_track(periods_max, ampl_max, length=5000):
    random_freq = np.random.uniform(0, periods_max)
    random_phi = np.random.uniform(0, 2 * np.pi)
    x = np.linspace(-random_phi, random_freq * 2 * np.pi, length)
    x = np.sin(x)
    random_ampl = np.random.uniform(0, ampl_max)
    x *= random_ampl
    return x


def add_sinus(dataset, periods_max, ampl_max):
    for recording in range(dataset.shape[0]):
        for channel in range(dataset.shape[2]):
            sinus_track = create_sinus_track(periods_max, ampl_max)
            dataset[recording][:, channel] += sinus_track
    return dataset


def mul_sinus(dataset, periods_max, replace_factor, ampl_max):
    for recording in range(dataset.shape[0]):
        for channel in range(dataset.shape[2]):
            sinus_track = create_sinus_track(periods_max, ampl_max)
            dataset[recording][:, channel] = (dataset[recording][:, channel] * sinus_track) * replace_factor + (
                        1 - replace_factor) * dataset[recording][:, channel]
    return dataset


def one_hot_encode(number, dimensions):
    vector = np.zeros(dimensions)
    vector[number] = 1
    return vector


def noise_augment(recordings, ampl):
    noise_matrix = np.random.random((recordings.shape[0], recordings.shape[1], recordings.shape[2])) * ampl
    augmented_recordings = np.add(noise_matrix, recordings)
    return augmented_recordings


def preprocess(recording, cut_start=cut_point_start, cut_end=cut_point_end):
    cut_recording = recording[cut_start:cut_end]
    transformer = sklearn.preprocessing.RobustScaler().fit(recording)
    preprocessed = transformer.transform(cut_recording)
    return preprocessed


def get_data(data_dir):
    labels = np.empty((0, 3))
    preprocessed_recordings = np.empty((0, 5000, 8))

    all_train_files = glob.glob(data_dir + "/*.csv")
    print(all_train_files)
    for index, freq_suffix in enumerate(freq_suffices):
        current_freq_files = [s for s in all_train_files if freq_suffix in s]

        for current_freq_file in current_freq_files:
            recording = pd.read_csv(current_freq_file, header=None)

            preprocessed_recording = preprocess(recording)
            preprocessed_recordings = np.append(preprocessed_recordings, [preprocessed_recording], axis=0)

            label = one_hot_encode(index, len(freq_suffices))
            labels = np.append(labels, [label], axis=0)

    return preprocessed_recordings, labels


x_train, y_train = get_data(train_data_dir)
x_test, y_test = get_data(test_data_dir)

my_brain_net = brain_net()

aug_x_train = np.empty((0, 5000, 8))
aug_y_train = np.empty((0, 3))

for j in range(0, 20):
    aug_x_train = np.append(aug_x_train, mul_sinus(x_train, 1000, 0.3, 1), axis=0)
    aug_y_train = np.append(aug_y_train, y_train, axis=0)

aug_x_train = noise_augment(aug_x_train, 2)

my_brain_net.fit(aug_x_train, aug_y_train, shuffle=True, validation_split=0.2, epochs=100, batch_size=50)

score = my_brain_net.evaluate(x_test, y_test, batch_size=48)
print(score)