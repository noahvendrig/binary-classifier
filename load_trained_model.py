
# Import required modules

import cv2

import tensorflow as tf
import os

os.chdir("D:/py/project/train_model") # Set the Current Working Directory

CATEGORIES = ['Dog', 'Not Dog'] # Categories for prediction, turning 0 and 1 into Dog and Not Dog

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True) 


def prepare(filepath): # Preparing model
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("128x3-CNN.model")

prediction = model.predict([prepare('dog.jpg')])
print(prediction)

# prediction = model.predict([prepare('dog.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('cat.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('bunny.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

imgs = ["dog.jpg", "cat.jpg", "bunny.jpg"]

def predict(imgs):
    for img in imgs:
        prediction = model.predict([prepare(img)])
        print(img, "=",(CATEGORIES[int(prediction[0][0])]))

predict(imgs)
