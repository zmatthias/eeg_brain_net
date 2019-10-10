import numpy as np

import sklearn.preprocessing
import keras
import gc
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from dynamic_net import dynamic_net
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)
np.random.seed(0)




def train(x_train, y_train, model):
    model.fit(x_train, y_train, shuffle=True, epochs=5, batch_size=5, verbose=1)


def test(x_test, y_test, model):
    score = model.evaluate(x_test, y_test, batch_size=48)
    score = np.asarray(score)
    return score


def write_log_params(file_path, params):
    file_stream = open(file_path, "a+")
    file_stream.write("\n \nGenes: ")
    file_stream.write(str(params))
    file_stream.close()


def write_log_metrics(file_path, val_scores, test_scores):
    file_stream = open(file_path,"a+")
    file_stream.write("\nValidation Loss, Acc: ")
    file_stream.write(val_scores)
    file_stream.write("\nTest Loss, Acc: ")
    file_stream.write(test_scores)
    file_stream.close()


def reset_weights(model):
    my_session = tf.compat.v1.keras.backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=my_session)


def train_test_individual(genes, x_train, y_train, x_test, y_test):
    lr = 0.0001
    feature_size = int(np.maximum(round(genes[0]), 1))
    conv_layer_count = int(np.maximum(round(genes[1]), 1))
    kernel_size = int(np.maximum(round(genes[2]), 1))
    dilation_rate = int(np.maximum(round(genes[3]), 1))
    dropout = genes[4]
    file_path = "run_log.txt"

    my_genes = OrderedDict([("lr", lr), ("feature_size", feature_size), ("conv_layer_count", conv_layer_count),
                             ("kernel_size", kernel_size), ("dilation_rate", dilation_rate), ("dropout", dropout)])

    write_log_params(file_path, my_genes)
    print("Genes:")
    print(my_genes)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores_sum, test_scores_sum = np.zeros(2), np.zeros(2)
    for train_index, val_index in skf.split(x_train, y_train.argmax(1)):  # convert one-hot to integer values to split
        print("TRAIN SET:", train_index, "VAL SET:", val_index)
        x_train_piece, x_val_piece = x_train[train_index], x_train[val_index]
        y_train_piece, y_val_piece = y_train[train_index], y_train[val_index]

        my_dynamic_net = dynamic_net(my_genes)
        train(x_train_piece, y_train_piece, my_dynamic_net)

        val_scores_sum += test(x_val_piece, y_val_piece, my_dynamic_net)
        test_scores_sum += test(x_test, y_test, my_dynamic_net)
        del my_dynamic_net

    val_score_avg = val_scores_sum / skf.get_n_splits()
    test_score_avg = test_scores_sum / skf.get_n_splits()

    print("Validation Loss, Acc: " + str(val_score_avg))
    print("Test Loss, Acc: " + str(test_score_avg))
    
    write_log_metrics(file_path, str(val_score_avg), str(test_score_avg))
    keras.backend.clear_session()
    gc.collect()
    val_loss_avg = val_score_avg[0]

    return val_loss_avg
