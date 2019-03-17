from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid', 
#     activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()
