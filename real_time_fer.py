import cv2
import numpy as np
from time import sleep
from keras.models import load_model

emotions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise', 'Neutral']

ResNet50Model = load_model('./ResNet50.h5')
InceptionV3Model = load_model('./InceptionV3.h5')
CustomModel = load_model('./CustomModel.h5')

models = [ResNet50Model, InceptionV3Model, CustomModel]

cascPath = '/Users/ChuanWooi/Documents/GitHub/FER/trained_models/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

def preprocess_input(image, train_model):
    if train_model == "Custom":
        img_width, img_height = 64, 64
        image = cv2.resize(image, (img_width, img_height))
        ret = np.empty((img_height, img_width, 1))
        ret[:, :, 0] = image
        x = np.expand_dims(ret, axis = 0)
        
        x -= 128.8006   # np.mean(train_dataset)
        x /= 64.6497    # np.std(train_dataset)
        
        return x
        
    elif train_model == "Inception":
        img_width, img_height = 139, 139
    elif train_model == "ResNet":
        img_width, img_height = 197, 197
        
    image = cv2.resize(image, (img_width, img_height))  # Resizing images for the trained model
    ret = np.empty((img_height, img_width, 3)) 
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis = 0)   # (1, XXX, XXX, 3)

    if train_model == "Inception":
        x /= 127.5
        x -= 1.
        #return x
    elif train_model == "ResNet":
        x -= 128.8006   # np.mean(train_dataset)
        x /= 64.6497    # np.std(train_dataset)

    return x

def ensemble_predict(models, emotion):
    yhats = [model.predict(emotion[index]) for index, model in enumerate(models)]
    yhats = np.array(yhats)
    # sum across ensemble members
    summed = np.sum(yhats, axis=0)
    
    return summed
  
  while True:
    if not video_capture.isOpened():    # If the previous call to VideoCapture constructor or VideoCapture::open succeeded, the method returns true
        print('Unable to load camera.')
        sleep(5)                        # Suspend the execution for 5 seconds
    else:
        sleep(0.5)
        ret, frame = video_capture.read()                        # Grabs, decodes and returns the next video frame (Capture frame-by-frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Conversion of the image to the grayscale
        
        faces = faceCascade.detectMultiScale(
            gray_frame,
            scaleFactor     = 1.1,
            minNeighbors    = 5,
            minSize         = (30, 30))

        prediction = None
        x, y = None, None

        for (x, y, w, h) in faces:
            ROI_gray = gray_frame[y:y+h, x:x+w] # Extraction of the region of interest (face) from the frame
            padding = int(h * 0.1)

            emotion = [preprocess_input(ROI_gray, 'ResNet'), 
                       preprocess_input(ROI_gray, 'Inception'), 
                       preprocess_input(ROI_gray, 'Custom')]
            prediction = ensemble_predict(models, emotion)
            top_1_prediction = emotions[np.argmax(prediction)]
            percentage = prediction[0][np.argmax(prediction)] / 3 * 100
            
            if top_1_prediction == "Happy" or top_1_prediction == "Surprise":
                category = "(Positive)"
                color = (0, 255, 0)
                text_color = (0, 0, 0)
            elif top_1_prediction == "Neutral":
                category = ""
                color = (255, 0, 0)
                text_color = (255, 255, 255)
            else:
                category = "(Negative)"
                color = (0, 0, 255)
                text_color = (255, 255, 255)

            font_scale = h/320
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + padding), color, -1)
            cv2.putText(frame, top_1_prediction + category, (x, y + int(padding/1.3)), cv2.FONT_HERSHEY_TRIPLEX, font_scale, text_color, 1, cv2.LINE_AA)

        # Display the resulting frame
        frame = cv2.resize(frame, (500, 500))
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
