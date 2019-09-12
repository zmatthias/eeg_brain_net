import numpy as np


def create_sinus_track(periods_max, amp_max, length=5000):
    random_freq = np.random.uniform(0, periods_max)
    random_phi = np.random.uniform(0, 2 * np.pi)
    x = np.linspace(-random_phi, random_freq * 2 * np.pi, length)
    x = np.sin(x)
    random_amp = np.random.uniform(0, amp_max)
    x *= random_amp
    return x


def mul_sinus_augment(x_train, periods_max, replace, amp_max):
    for recording in range(x_train.shape[0]):
        for channel in range(x_train.shape[2]):
            sinus_track = create_sinus_track(periods_max, amp_max)
            x_train[recording][:, channel] = (x_train[recording][:, channel] * sinus_track) * replace + (
                    1 - replace) * x_train[recording][:, channel]
    return x_train


def noise_augment(x_train, amp):
    noise_matrix = np.random.random((x_train.shape[0], x_train.shape[1], x_train.shape[2])) * amp
    x_train = np.add(noise_matrix, x_train)
    return x_train


def random_cuts_augment(x_train, cut_min_start, cut_max_deviation, cut_length):  # more data_balint with different cuts
    x_train_aug = np.empty((0, cut_length, 8))

    for recording in range(x_train.shape[0]):
        cut_start_deviation = np.random.randint(0, cut_max_deviation)
        cut_start = cut_min_start + cut_start_deviation
        cut_end = cut_start + cut_length
        cut_recording = x_train[recording][cut_start:cut_end]
        x_train_aug = np.append(x_train_aug, [cut_recording], axis=0)
    return x_train_aug


def full_augment(x_train, y_train, data_mul):
    all_aug_x_train = np.empty((0, 5000, 8))
    all_aug_y_train = np.empty((0, 3))

    for j in range(0, data_mul):
        aug_x_train = random_cuts_augment(x_train, cut_min_start=0, cut_max_deviation=800, cut_length=5000)
        aug_x_train = mul_sinus_augment(aug_x_train, periods_max=1000, replace=0.5, amp_max=1)
        aug_x_train = noise_augment(aug_x_train, amp=2)
        all_aug_x_train = np.append(all_aug_x_train, aug_x_train, axis=0)
        all_aug_y_train = np.append(all_aug_y_train, y_train, axis=0)
    return all_aug_x_train, all_aug_y_train
