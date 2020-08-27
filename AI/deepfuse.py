import os
from tensorflow import keras
import efficientnet.tfkeras
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from skimage.transform import resize
import matplotlib.pyplot as plt

from efficientnet.tfkeras import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import optimizers


plt.style.use('ggplot')

# functions
def load_dataset(datasetPath, imgDims):
    data = []
    labels = []
    g = 0
    n = 0

    for d in os.listdir(datasetPath):
        p = os.path.join(datasetPath, d)

        for f in os.listdir(p):
            pp = os.path.join(p, f)
            hdul = fits.open(pp)
            img = hdul[1].data
            hdul.close()

            img = resize(img, imgDims)
            
            """check wether to use zero padding (1) or repeat image to the 3 dimensions"""

            # (1)
            #img = img.reshape((img.shape[0], img.shape[1], 1))
            #img = np.pad(img, [(0,0), (0,0), (0, 2)], 'constant')

            # (2)
            img = np.repeat(img[..., np.newaxis], 3, -1)
            
            if not (np.isnan(img).any() or np.isinf(img).any()): # if the image has weird values value, throw it out
                data.append(img)

                if d == "galaxies":
                    label = 1
                    g += 1
                    labels.append(label)
                elif d == "not":
                    label = 0
                    n += 1
                    labels.append(label)
          
    data = np.asarray(data)
    data = data.astype("float32")

    labels = np.asarray(labels) 

    print("total galaxies: ", g)
    print("total not: ", n)
    
    return data, labels


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)

    auroc = float("{:.2f}".format(auroc))
    n = f"{name}; auroc = {auroc}"

    plt.plot(fp, tp, label=n, **kwargs)
    plt.xlabel('False Positives [%]')
    plt.ylabel('True Positives [%]')
    ax = plt.gca()
    ax.set_aspect('equal')


# getting data
TRAIN_DIR_PATH = "train"
print("creating dataset...")
IMG_DIMS = (200, 200) 
data, labels = load_dataset(TRAIN_DIR_PATH, IMG_DIMS)


# preparing for training
#le = LabelBinarizer()
#labels = le.fit_transform(labels)
#counts = labels.sum(axis=0)

# class weights, not sure?
#classTotals = labels.sum(axis=0)
#classWeight = {}
#for i in range(0, len(classTotals)):
#    classWeight[i] = classTotals.max() / classTotals[i]

# split training and validation data (train = 60%; val = 20%; test = 20%)
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=42)
#trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, random_state=42)
#trainX, valX, trainY, valY = train_test_split(data, labels, test_size=0.20, random_state=42)


print("size of training data: ", len(trainX))
print("size of validation data: ", len(valX))
print("size of test data: ", len(testX))

"""
# data augmentation
aug = ImageDataGenerator(
    rotation_range=90,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
"""

# setup model
model = keras.models.load_model("efn0_vis.hdf5", custom_objects={'RAdam': tfa.optimizers.RectifiedAdam})
opt = tfa.optimizers.RectifiedAdam(0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


# freeze the model
# 0; 4; 14; 42; 70; 113; 156; 214
for layer in model.layers[:113]:
    layer.trainable = False
for layer in model.layers[113:]:
    layer.trainable = True

# train and evaluate
BS = 64
EPOCHS = 10

history = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, steps_per_epoch=len(trainX)//BS, validation_data=(valX, valY), validation_steps = len(valX)//BS, verbose=1)
model.save("deepfuse-roc.hdf5", save_format="hdf5")


# define the list of label names
#labelNames = ["not", "galaxy"]

# check out some meta data
#predictions = model.predict(testX, batch_size=BS)
#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

train_predictions = model.predict(trainX, batch_size=BS).max(axis=1)
test_predictions = model.predict(testX, batch_size=BS).max(axis=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig("accuracy.png")

plt.figure()
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.savefig("loss.png")

plt.figure()
plot_roc("Training", trainY, train_predictions, color="b")
plot_roc("Testing", testY, test_predictions, color="r")#, linestyle='--')
plt.legend(loc='lower right')
plt.savefig("roc.png")
plt.show()

