from keras.models import Sequencial
from keras.layers import Dense

model = Sequencial([Dense(64, activation='relu', input_shape=(784)), Dense=(10, activation="softmax")])

model = Sequencial()
model.add(Dense(64, activation='relu'))

model.compile(optimer='adam', loss='categorial', metrics='')