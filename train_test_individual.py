import numpy as np
import time
import keras
import gc
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from dynamic_net import dynamic_net
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
import psutil
import humanize
import os
import GPUtil as GPU

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow

np.random.seed(0)
checkpoint_path = "model.h5"


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    print(gc.collect())  # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.compat.v1.ConfigProto()
    try:
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
    except:
        print("GPU settings failed, no GPU?")

    set_session(tensorflow.compat.v1.Session(config=config))


def print_mem():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
          " | Proc size: " + humanize.naturalsize(process.memory_info().rss))
    try:
        gpu = GPU.getGPUs()[0]

        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
                                                                                                    gpu.memoryUsed,
                                                                                                    gpu.memoryUtil * 100,
                                                                                                    gpu.memoryTotal))
    except:
        print("no GPU!")


def train(x_train, y_train, model, conf):
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=conf["patience"], mode='min', min_delta=0.0001),
        # ModelCheckpoint(conf["checkpoint_path"], monitor='val_loss', save_best_only=True, mode='min')
    ]
    # other validation here than k-fold
    start = time.time()
    model.fit(x_train, y_train, shuffle=True, epochs=conf["train_epochs"], batch_size=conf["train_batch_size"],
              verbose=conf["train_verbose"], validation_split=0.2, callbacks=keras_callbacks)

    end = time.time()
    print("(Partial) Training took:" + str(end - start) + "s")

def test(x_test, y_test, model, conf):
    print("Memory before Test")
    print_mem()
    # model = keras.models.load_model(conf["checkpoint_path"])  # load the best checkpoint model instead of the last
    score = model.evaluate(x_test, y_test, batch_size=conf["test_batch_size"])
    score = np.asarray(score)
    del model
    print("Memory after Test")
    print_mem()

    return score


def write_log_genes(file_path, genes):
    file_stream = open(file_path, "a+")
    file_stream.write("\n \nGenes: ")
    file_stream.write(str(genes))
    file_stream.close()


def write_log_metrics(file_path, val_scores, test_scores):
    file_stream = open(file_path, "a+")
    file_stream.write("\nValidation Loss, Acc: ")
    file_stream.write(val_scores)
    file_stream.write("\nTest Loss, Acc: ")
    file_stream.write(test_scores)
    file_stream.close()


def interpret(genes):
    lr = genes[0]
    feature_size = int(np.maximum(round(genes[1]), 1))
    conv_layer_count = int(np.maximum(round(genes[2]), 1))
    fc_layer_count = int(np.maximum(round(genes[3]), 1))
    fc_neurons = int(np.maximum(round(genes[4]), 1))
    kernel_size = int(np.maximum(round(genes[5]), 1))
    dilation_rate = int(np.maximum(round(genes[6]), 1))
    dropout = genes[7]
    labeled_genes = OrderedDict([("lr", lr), ("feature_size", feature_size), ("conv_layer_count", conv_layer_count),
                                 ("fc_layer_count", fc_layer_count), ("fc_neurons", fc_neurons),
                                 ("kernel_size", kernel_size), ("dilation_rate", dilation_rate), ("dropout", dropout)])
    return labeled_genes


def train_test_individual(genes, conf, data):  # x_test means actual test, not val

    labeled_genes = interpret(genes)
    file_path = conf["log_file_path"]
    write_log_genes(file_path, labeled_genes)
    print("Genes:")
    print(labeled_genes)

    skf = StratifiedKFold(n_splits=conf["fold_count"], shuffle=True, random_state=42)
    val_scores_sum, test_scores_sum = np.zeros(2), np.zeros(2)

    for train_index, val_index in skf.split(data["x_train"],
                                            data["y_train"].argmax(1)):  # convert one-hot to integer values to split
        # print("TRAIN SET:", train_index, "VAL SET:", val_index)
        x_train_piece, x_val_piece = data["x_train"][train_index], data["x_train"][val_index]
        y_train_piece, y_val_piece = data["y_train"][train_index], data["y_train"][val_index]

        my_dynamic_net = dynamic_net(labeled_genes)
        train(x_train_piece, y_train_piece, my_dynamic_net, conf)

        val_score = test(x_val_piece, y_val_piece, my_dynamic_net, conf)
        print("k-fold val score:" + str(val_score))

        if val_score[0] > conf["first_val_loss_max"]:  # if val score is too bad, don't even bother

            val_scores_sum = np.array([99.0, 0.0])
            test_scores_sum = np.array([99.0, 0.0])
            break

        else:
            val_scores_sum += val_score

            test_score = test(data["x_test"], data["y_test"], my_dynamic_net, conf)
            print("k-fold test score:" + str(test_score))
            test_scores_sum += test_score
            del my_dynamic_net
            print("try Reset")
            reset_keras()
            gc.collect()

    val_score_avg = val_scores_sum / skf.get_n_splits()
    test_score_avg = test_scores_sum / skf.get_n_splits()

    print("Validation Loss, Acc: " + str(val_score_avg))
    print("Test Loss, Acc: " + str(test_score_avg))

    write_log_metrics(conf["log_file_path"], str(val_score_avg), str(test_score_avg))
    keras.backend.clear_session()
    val_loss_avg = val_score_avg[0]

    return val_loss_avg