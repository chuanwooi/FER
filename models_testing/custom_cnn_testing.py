import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

img_width, img_height =	64, 64

test_data_dir = "./fer2013/fer2013_test.csv"
predict_batch_size = 1

model = load_model("./trained_models/CustomModel.h5")

def preprocess_input(x):
	x -= 128.8006	
	x /= 64.6497	
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
    images[i, :, :, :] = ret
    i += 1

  images = preprocess_input(images)
  labels = data["emotion"].tolist()

  return images, labels

X_test, y_test = get_data(test_data_dir)

predictions = model.predict(
	X_test,
	batch_size	= predict_batch_size)

predicted_classes	= np.argmax(predictions, axis = 1)		
true_classes 		=  y_test 						
class_names 		= list(["Anger", "Disgust", "Fear", "Happinness", "Sadness", "Surprise", "Neutral"])

accuracy = accuracy_score(
	true_classes, 
	predicted_classes, 
	normalize = True)

report = classification_report(
	true_classes, 
	predicted_classes,
	target_names = class_names)

# Print the result of the evaluation
print("Accuracy:")
print(accuracy)

print("\n")
print("Report:")
print(report)
print("\n")
