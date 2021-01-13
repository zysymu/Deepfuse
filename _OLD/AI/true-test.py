import os
from tensorflow import keras
from tensorflow.keras import layers
import efficientnet.tfkeras
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from astropy.io import fits

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import seaborn as sns

from skimage.transform import resize
import matplotlib.pyplot as plt


plt.style.use('ggplot')

# FUNCTIONS:
def load_stamps(path, imgDims):
    data = []

    for d in os.listdir(path):
        p = os.path.join(path, d)

        for f in os.listdir(p):
            if f[-4:] == "fits":
                pp = os.path.join(p, f)
                hdul = fits.open(pp)
                img = hdul[1].data
                hdul.close()
    
                # apply pre-processing
                img = resize(img, imgDims)
                img = img.reshape((img.shape[0], img.shape[1], 1))
                img = np.pad(img, [(0,0), (0,0), (0, 2)], 'constant')
                data.append(img)

    data = np.asarray(data)
    data = data.astype("float32")
    return data

print("processing data...")
STAMPS_PATH = "stamps"
IMG_DIMS = (50, 50) 
data = load_stamps(STAMPS_PATH, IMG_DIMS)

print("loading model and predicting...")
model_path = "metrics/model-roc.hdf5"
model = keras.models.load_model(model_path)
BS = 64

test_predictions = model.predict(data, batch_size=BS).max(axis=1)

for t in test_predictions:
    if t >= 0.5:
        print(t)
print("end")

