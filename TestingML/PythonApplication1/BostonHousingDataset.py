from keras.datasets import boston_housing
from keras import models
from keras import layers

def build_model(shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print (train_data.shape)

print (test_data.shape)

print (train_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

shape = (train_data.shape[1],)

model = build_model(shape)