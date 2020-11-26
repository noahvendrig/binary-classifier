import cv2
print(cv2.__version__)


import tensorflow as tf
import os

os.chdir("D:/py/project/train_model")

CATEGORIES = ['Dog', 'Not Dog']

filelist= []

for filename in os.listdir("D:/py/project/train_model/input_imgs"): 
    filelist.append(filename)

print(filelist)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("128x3-CNN.model")

#prediction = model.predict([prepare('dog.jpg')])
#print(prediction)

prediction = model.predict([prepare('input_imgs/dog.jpg')]) # Initialising 'prediction' (doesn't actually matter what image is used here)
print(prediction)

# prediction = model.predict([prepare('input_imgs/dog.jpg')])  # One by one approach, will only analyse images that have been selected
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('cat.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('bunny.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

os.chdir("D:/py/project/train_model/input_imgs")



def predict(filelist):
    for img in filelist:
        prediction = model.predict([prepare(img)])
        print(img, "=",(CATEGORIES[int(prediction[0][0])]))
        


        
predict(filelist)
