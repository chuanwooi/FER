import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras.models import load_model

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

test_data_dir = "./fer2013/fer2013_test.csv"

file_stream = file_io.FileIO(test_data_dir, mode="r")
test_data = pd.read_csv(file_stream)
labels = test_data["emotion"].tolist()

ResNetModel = load_model('./ResNet50.h5')
InceptionResNetModel = load_model('./InceptionV3.h5')
CustomModel = load_model('./CustomModel.h5')

def preprocess_input(x, train_model):
    if train_model == "Inception":
        x /= 127.5
        x -= 1.
        return x
    elif train_model == "ResNet" or train_model == "Custom":
        x -= 128.8006    # np.mean(train_dataset)
        x /= 64.6497    # np.std(train_dataset)
    return x

def get_data(dataset, train_model):
    channel = 3
    if train_model == "Inception":
        img_width, img_height =    139, 139
    elif train_model == "ResNet":
        img_width, img_height =    197, 197
    elif train_model == "Custom":
        img_width, img_height =    64, 64
        channel = 1
        
    #file_stream = file_io.FileIO(test_data_dir, mode="r")
    #data = pd.read_csv(file_stream)
    #pixels = data["pixels"].tolist()
    pixels = dataset["pixels"].tolist()
    images = np.empty((len(dataset), img_height, img_width, channel))
    i = 0
    
    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(" ")]    # Extraction of each image
        single_image = np.asarray(single_image).reshape(48, 48)                    # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 3, mode = "constant") # Bicubic
        ret = np.empty((img_height, img_width, channel))
        if channel == 1:
            ret[:, :, 0] = single_image
        else:
            ret[:, :, 0] = single_image
            ret[:, :, 1] = single_image
            ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1
    
    images = preprocess_input(images, train_model)

    return images
    
batch_size = 1
class_names = list(["Anger", "Disgust", "Fear", "Happinness", "Sadness", "Surprise", "Neutral"])

ResNet_images = get_data(test_data, 'ResNet')
Inception_images = get_data(test_data, 'Inception')
Custom_images = get_data(test_data, 'Custom')

yhats = [ResNetModel.predict(ResNet_images, batch_size = batch_size),
         InceptionResNetModel.predict(Inception_images, batch_size = batch_size),
         CustomModel.predict(Custom_images, batch_size = batch_size)]
yhats_array = np.array(yhats)
summed = np.sum(yhats_array, axis=0)
result = np.argmax(summed, axis=1)

accuracy = accuracy_score(
	labels, 
	result, 
	normalize = True)

report = classification_report(
	labels, 
	result,
	target_names = class_names)

print("Accuracy:")
print(accuracy)
print("\n")
print("Report:")
print(report)
print("\n")

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    print(tick_marks)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = confusion_matrix(y_true=labels, y_pred=result)
cm_plot_labels = ["Anger", "Disgust", "Fear", "Happinness", "Sadness", "Surprise", "Neutral"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
