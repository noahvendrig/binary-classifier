import cv2
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
output_path = "D:/py/project/train_model/output"

try:
    os.mkdir(output_path)
except FileExistsError:
    print("Directory already exists")

os.chdir("D:/py/project/train_model/input_imgs")

font = cv2.FONT_HERSHEY_SIMPLEX

def predict(filelist):
    for img in filelist:
        prediction = model.predict([prepare(img)])
        print(img, "=",(CATEGORIES[int(prediction[0][0])]))
        
        str_prediction = str(CATEGORIES[int(prediction[0][0])])
        
        img_cv = cv2.imread(img)
        img_cv = cv2.resize(img_cv, (1920,1080))

        cv2.putText(img_cv, str_prediction, (200,200), font, 5, (255,0,0), 10)
        #cv2.putText(img, "START", start_pt_centre, font, 3, (255, 0, 255), 3)

        cv2.imwrite(output_path+'/'+img+'.jpg', img_cv)

predict(filelist)
