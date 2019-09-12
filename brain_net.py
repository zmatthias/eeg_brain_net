import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D,MaxPooling1D,Flatten
from keras.optimizers import SGD


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
