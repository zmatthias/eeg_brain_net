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
checkpoint_path = "model.h5"


def train(x_train, y_train, model, conf):
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, mode='min', min_delta=0.0001),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    ]
    # other validation here than k-fold
    model.fit(x_train, y_train, shuffle=True, epochs=conf["train_epochs"], batch_size=conf["train_batch_size"],
              verbose=conf["train_verbose"], validation_split=0.2, callbacks=keras_callbacks)


def test(x_test, y_test, model, conf):
    score = model.evaluate(x_test, y_test, batch_size=conf["test_batch_size"])
    score = np.asarray(score)
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


def reset_weights(model):
    my_session = tf.compat.v1.keras.backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=my_session)


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

    for train_index, val_index in skf.split(data["x_train"], data["y_train"].argmax(1)):  # convert one-hot to integer values to split
        # print("TRAIN SET:", train_index, "VAL SET:", val_index)
        x_train_piece, x_val_piece = data["x_train"][train_index], data["x_train"][val_index]
        y_train_piece, y_val_piece = data["y_train"][train_index], data["y_train"][val_index]

        my_dynamic_net = dynamic_net(labeled_genes)
        train(x_train_piece, y_train_piece, my_dynamic_net, conf)

        val_scores_sum += test(x_val_piece, y_val_piece, my_dynamic_net, conf)
        test_scores_sum += test(data["x_test"], data["y_test"], my_dynamic_net, conf)
        del my_dynamic_net

    val_score_avg = val_scores_sum / skf.get_n_splits()
    test_score_avg = test_scores_sum / skf.get_n_splits()

    print("Validation Loss, Acc: " + str(val_score_avg))
    print("Test Loss, Acc: " + str(test_score_avg))
    
    write_log_metrics(conf["log_file_path"], str(val_score_avg), str(test_score_avg))
    keras.backend.clear_session()
    gc.collect()
    val_loss_avg = val_score_avg[0]

    return val_loss_avg
