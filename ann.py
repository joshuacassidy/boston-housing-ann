from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

normalised_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)

normalised_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0)

print(x_train.shape)

from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit(normalised_x_train,
                    y_train,
                    epochs=100,
                    batch_size=1,
)
print(model.evaluate(normalised_x_test, y_test))
