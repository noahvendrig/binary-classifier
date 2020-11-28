import time

# Train the neural network to identify dog or cat
# !! Always make sure dataset is balanced : Same amount of dog and cat images !!
# Grayscale convolutional neural network
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import os

os.chdir("D:/py/project/train_model")

DATADIR = "D:/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

PREPARE_DATASET = False

import time
start = time.time()

if PREPARE_DATASET:

 
  # for category in CATEGORIES:
  #   path = os.path.join(DATADIR, category) #path to cats or dogs dir
  #   for img in os.listdir(path):
  #     img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
  #     #plt.imshow(img_array, cmap="gray")
  #     plt.show()
  #     break
  #   break
 

  import pathlib
  pathlib.Path(DATADIR).parent.absolute()

  #print(img_array.shape)

  # IMG_SIZE = 50
  # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  # plt.imshow(new_array, cmap='gray')
  # plt.show()

  #Create training data
  #Index value of dog is 0, cat is 1
  

"""
  def create_training_data():
    training_data = []
    for category in CATEGORIES:
      path = os.path.join(DATADIR, category) #path to cats or dogs dir
      class_num = CATEGORIES.index(category)
      for img in os.listdir(path):
        try:
          img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          training_data.append([new_array, class_num])
        except Exception as e:
          print(e, path, img)
          pass
    return training_data
        
  training_data = create_training_data()

  print("Size of training data", len(training_data))

  import random
  random.shuffle(training_data)

  for sample in training_data[:10]:
    print(sample[1])

  X = []
  y = []

  for features, label in training_data:
    X.append(features)
    y.append(label)

  X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  #y = np.asarray(y).astype('float32').reshape((-1,1))
  
  
  import pickle
  pickle_out = open("X.pickle", "wb")
  pickle.dump(X, pickle_out)
  pickle_out.close()

  pickle_out = open("y.pickle", "wb")
  pickle.dump(y, pickle_out)
  pickle_out.close()


#  pickle_in = open("X.pickle", "rb")
#  X = pickle.load(pickle_in)


############################################################################

import pickle

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


print("Loaded in", (time.time()-start))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import  TensorBoard

main_dir = "D:/py/project/train_model/"


#NAME = "Cats-vs-dog-cnn-64x3-{}".format(int(time.time()))


#run in train_model dir (dir housing /logs/) and open cmd. paste in: python -m tensorboard.main --logdir=logs/
# rmdir logs /s TO DELETE LOGS FOLDER

X = X/255.0

'''
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]
'''

# From analysing TensorBoard data we can conclude that 3, 128, 2 gives us the best results
dense_layers = [2]
layer_sizes = [128]
conv_layers = [3]

for dense_layer in dense_layers:
  for layer_size in layer_sizes:
    for conv_layer in conv_layers:
      NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
      tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

      model = Sequential()

      model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2,2)))

      for l in range(conv_layer-1):
        model.add(Conv2D(layer_size, (3,3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

      model.add(Flatten())
      for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

      model.add(Dense(1))
      model.add(Activation("sigmoid"))

      model.compile(loss="binary_crossentropy",
                    optimizer='adam',
                    metrics=['accuracy'])

      model.summary()

      y = np.array(y).reshape(-1)

      model.fit(X, y, batch_size=32, epochs=7, validation_split=0.3, callbacks=[tensorboard])

      model.save('128x3-CNN.model')

"""