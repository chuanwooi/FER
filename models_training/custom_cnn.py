import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from skimage.transform import resize
from tensorflow.python.lib.io import file_io
from keras.utils import to_categorical

img_height, img_width = 64, 64
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100

train_dataset    = './fer2013/fer2013_train.csv'
eval_dataset     = './fer2013/fer2013_eval.csv'

def preprocess_input(x):
	x -= 128.8006	# np.mean(train_dataset)
	return x

def get_data(dataset):
  file_stream = file_io.FileIO(dataset, mode='r')
  data = pd.read_csv(file_stream)
  pixels = data['pixels'].tolist()
  images = np.empty((len(data), img_height, img_width, 1))
  i = 0

  for pixel_sequence in pixels:
    single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
    single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
    single_image = resize(single_image, (img_height, img_width), order = 3, mode = 'constant') # Dimension: 64x64x1 
    ret = np.empty((img_height, img_width, 1))  
    ret[:, :, 0] = single_image
    #ret[:, :, 1] = single_image
    #ret[:, :, 2] = single_image
    images[i, :, :, :] = ret
    i += 1

  images = preprocess_input(images)
  labels = to_categorical(data['emotion'])

  return images, labels

# Data preparation
train_data_x, train_data_y  = get_data(train_dataset)
val_data_x, val_data_y = get_data(eval_dataset)

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

model.summary()

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpointer = ModelCheckpoint('./fer2013/CustomModel.h5', monitor='val_loss', verbose=1, save_best_only=True)

model.fit(train_data_x, train_data_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_data_x, val_data_y),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, checkpointer])
