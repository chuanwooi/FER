import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras.models import load_model
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback, ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import array_to_img

img_width, img_height =	197, 197
num_classes         = 7     
epochs_top_layers   = 5
epochs_all_layers   = 100
batch_size          = 128

train_dataset	= './fer2013/fer2013_train.csv'
eval_dataset 	= './fer2013/fer2013_eval.csv'

def preprocess_input(x):
    x -= 128.8006 # np.mean(train_dataset)
    return x

# Function that reads the data from the csv file, increases the size of the images and returns the images and their labels
    # dataset: Data path
def get_data(dataset):
    file_stream = file_io.FileIO(dataset, mode='r')
    data = pd.read_csv(file_stream)
    pixels = data['pixels'].tolist()
    images = np.empty((len(data), img_height, img_width, 3))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
        single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 3, mode = 'constant') # Dimension: 139x139x3 (Bicubic)
        ret = np.empty((img_height, img_width, 3))  
        ret[:, :, 0] = single_image
        ret[:, :, 1] = single_image
        ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1
    
    images = preprocess_input(images)
    labels = to_categorical(data['emotion'])

    return images, labels

train_data_x, train_data_y  = get_data(train_dataset)
val_data_x, val_data_y  = get_data(eval_dataset)

base_model = VGGFace(
    model       = 'resnet50',
    include_top = False,
    weights     = 'vggface',
    input_shape = (img_height, img_width, 3))

# Places x as the output of the pre-trained model
x = base_model.output

# Flattens the input. Does not affect the batch size
x = Flatten()(x)

# Add a fully-connected layer and a logistic layer
# Dense implements the operation: output = activation(dot(input, kernel) + bias(only applicable if use_bias is True))
    # units:        Positive integer, dimensionality of the output space
    # activation:   Activation function to use
    # input shape:  nD tensor with shape: (batch_size, ..., input_dim)
    # output shape: nD tensor with shape: (batch_size, ..., units)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
model.summary()

model.compile(
    optimizer   = SGD(lr = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpointer = ModelCheckpoint('./fer2013/ResNet50.h5', monitor='val_loss', verbose=1, save_best_only=True)

model.fit(train_data_x, train_data_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_data_x, val_data_y),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, checkpointer])
