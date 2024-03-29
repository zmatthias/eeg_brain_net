from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D,MaxPooling1D,Flatten
from keras.optimizers import Adam
from typing import Dict


def dynamic_net(params: Dict):
    model = Sequential()

    model.add(Conv1D(params['feature_size'], 1, dilation_rate=params['dilation_rate'], activation='relu', input_shape=(5000, 8)))
    for i in range(0, params['conv_layer_count']):
        model.add(Conv1D(params['feature_size'], params['kernel_size'], dilation_rate=params['dilation_rate'], activation='relu'))

    model.add(Flatten())

    for i in range(0, params['fc_layer_count']):
        model.add(Dense(params['fc_neurons'], activation='relu'))
        model.add(Dropout(params['dropout']))

    model.add(Dense(3, activation='softmax'))

    adam = Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
