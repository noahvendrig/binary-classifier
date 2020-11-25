'''
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


prediction = model.predict([prepare('input_imgs/dog.jpg')]) # Initialising 'prediction' (doesn't actually matter what image is used here)

#print(prediction)

# prediction = model.predict([prepare('input_imgs/dog.jpg')])  # One by one approach, will only analyse images that have been selected
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('input_imgs/cat.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('input_imgs/bunny.jpg')])
# print(CATEGORIES[int(prediction[0][0])])


# Create Directory
path = "D:/py/project/train_model/predictions"
try:
    os.mkdir(path)  # Create the directory for prediction images
except FileExistsError:
    "Directory already exists"
else:
    print("Successfully created "+ path)

# os.chdir("D:/py/project/train_model/input_imgs")

filelist= []

font = 'FONT_HERSHEY_SIMPLEX' # Font for text

for filename in os.listdir("D:/py/project/train_model/input_imgs"): 
    filelist.append(filename) # Add each file in the directory to the list

def predict(filelist):
    for image in filelist:
        prediction = model.predict([prepare(image)])
        print(image, "=",(CATEGORIES[int(prediction[0][0])])) # Print prediction in the terminal

        '''
        str_prediction = (CATEGORIES[int(prediction[0][0])])

        img = cv2.imread(file)
        cv2.putText(img, str_prediction, (100,50), font, 3, (255, 0, 255), 3) # Draw prediction onto the image, mainly for presentation purposes

        cv2.imwrite("D:/py/project/train_model/predictions", img) # Write images to path

        #show img
        cv2.imshow(file, img) # Displays image in a window (easy for testing)
        cv2.waitKey(0)
        '''

predict(filelist) # Call prediction function
'''







import cv2
print(cv2.__version__)


import tensorflow as tf
import os

os.chdir("D:/py/project/train_model")

CATEGORIES = ['Dog', 'Not Dog']



physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def prepare(filepath):

model = tf.keras.models.load_model("128x3-CNN.model")

prediction = model.predict([prepare('dog.jpg')])
print(prediction)

# prediction = model.predict([prepare('dog.jpg')])
prediction = model.predict([prepare('input_imgs/dog.jpg')]) # Initialising 'prediction' (doesn't actually matter what image is used here)
#print(prediction)

# prediction = model.predict([prepare('input_imgs/dog.jpg')])  # One by one approach, will only analyse images that have been selected
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('cat.jpg')])
# prediction = model.predict([prepare('input_imgs/cat.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('bunny.jpg')])
# prediction = model.predict([prepare('input_imgs/bunny.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

imgs = ["dog.jpg", "cat.jpg", "bunny.jpg"]
os.chdir("D:/py/project/train_model/input_imgs")

filelist= []

for filename in os.listdir("D:/py/project/train_model/input_imgs"): 
    filelist.append(filename)

def predict(imgs):
    for img in imgs:
def predict(filelist):
    for img in filelist:
        prediction = model.predict([prepare(img)])
        print(img, "=",(CATEGORIES[int(prediction[0][0])]))

predict(imgs)
predict(filelist)