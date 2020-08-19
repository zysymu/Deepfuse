import os
import tensorflow as tf
from tensorflow import keras
import efficientnet.tfkeras
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import cv2
import matplotlib.pyplot as plt

# functions
def load_dataset(datasetPath, imgDims):
    data = []
    labels = []

    for d in os.listdir(datasetPath):
        p = os.path.join(datasetPath, d)

        for f in os.listdir(p):
            pp = os.path.join(p, f)
            hdul = fits.open(pp)
            img = hdul[1].data
            hdul.close()

            img = cv2.resize(img, imgDims)
            img /= 255.0
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img = np.pad(img, [(0,0), (0,0), (0, 2)], 'constant')

            data.append(img)

            if d == "galaxies":
                label = 1
                labels.append(label)
            elif d == "not":
                label = 0
                labels.append(label)
          
    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")
    
    return data, labels

# getting data
TRAIN_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/train"
print("creating dataset...")
IMG_DIMS = (200, 200) 
data, labels = load_dataset(TRAIN_DIR_PATH, IMG_DIMS)

# preparing for training
le = LabelBinarizer()
labels = le.fit_transform(labels)
print(labels)
counts = labels.sum(axis=0)

# class weights, not sure?
classTotals = labels.sum(axis=0)
classWeight = {}
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# split training and validation data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# data augmentation
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# setup model
model = keras.models.load_model("efn0_vis.hdf5", custom_objects={'RAdam': tfa.optimizers.RectifiedAdam})
opt = tfa.optimizers.RectifiedAdam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train and evaluate
BS = 1
EPOCHS = 10

H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
    epochs=EPOCHS,
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps = len(testX) // BS,
    class_weight=classWeight,
    verbose=1)

# define the list of label names
labelNames = ["not", "galaxy"]

# check out some meta data
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# save model
model.save("deepfuse.h5", save_format="h5")

# construct a plot 
plt.style.use("ggplot")

N = np.arange(0, EPOCHS)

plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plt.show()
