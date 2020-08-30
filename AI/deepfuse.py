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

            if d == "galaxies":
                img = hdul[0].data
                hdul.close()
                if not (np.isnan(img).any() or np.isinf(img).any()): # if the image has weird values value, throw it out
     
                    """check wether to use zero padding (1) or repeat image to the 3 dimensions"""
                    # (1)
                    img = resize(img, imgDims)
                    img = img.reshape((img.shape[0], img.shape[1], 1))
                    img = np.pad(img, [(0,0), (0,0), (0, 2)], 'constant')

                    #img = resize(img, imgDims)
                    #img = np.repeat(img[..., np.newaxis], 3, -1)
                    data.append(img)

                    label = 1
                    g += 1
                    labels.append(label)

            elif d == "not":
                img = hdul[1].data
                hdul.close()
                if not (np.isnan(img).any() or np.isinf(img).any()): # if the image has weird values value, throw it out
     
                    """check wether to use zero padding (1) or repeat image to the 3 dimensions"""
                    # (1)
                    img = resize(img, imgDims)
                    img = img.reshape((img.shape[0], img.shape[1], 1))
                    img = np.pad(img, [(0,0), (0,0), (0, 2)], 'constant')

                    #img = resize(img, imgDims)
                    #img = np.repeat(img[..., np.newaxis], 3, -1)
                    data.append(img)

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
    plt.xlabel('False Positives')
    plt.ylabel('True Positives')
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_pr(name, labels, predictions, **kwargs):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    
    plt.plot(recall, precision, label=name, **kwargs)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions)# > p)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

############

# PREPARING DATA:
TRAIN_DIR_PATH = "train"
print("creating dataset...")
IMG_DIMS = (50, 50) 
data, labels = load_dataset(TRAIN_DIR_PATH, IMG_DIMS)

# Split training and validation data (train = 60%; val = 20%; test = 20%)
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=42)

print("size of training data: ", len(trainX))
print("size of validation data: ", len(valX))
print("size of test data: ", len(testX))

weights = class_weight.compute_class_weight('balanced', np.unique(trainY), trainY)
weights_dict = dict(enumerate(weights))
print(weights_dict)
"""
trainY = keras.utils.to_categorical(trainY)
valY = keras.utils.to_categorical(valY)
testY = keras.utils.to_categorical(testY)
"""
# SETTING UP AND TRAINING MODEl:
base = keras.applications.EfficientNetB4(include_top=False, input_shape=(50,50,3))
model = keras.models.Sequential()
model.add(base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid')) #sigmoid

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

#model = keras.models.load_model("efn0_vis.hdf5", custom_objects={'RAdam': tfa.optimizers.RectifiedAdam})
#opt = tfa.optimizers.RectifiedAdam(0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
"""
# Freezing the model
# 0; 4; 14; 42; 70; 113; 156; 214
for layer in model.layers[:113]:
    layer.trainable = False
for layer in model.layers[113:]:
    layer.trainable = True
"""
BS = 64
EPOCHS = 6

#history = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, steps_per_epoch=len(trainX)//BS, validation_data=(valX, valY), validation_steps = len(valX)//BS, verbose=1)
history = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, steps_per_epoch=len(trainX)//BS, validation_data=(valX, valY), validation_steps = len(valX)//BS, class_weight=weights_dict, verbose=1)

model_dir = "metrics"
os.mkdir(model_dir)
model.save(os.path.join(model_dir, "model-roc.hdf5"), save_format="hdf5")


# METRICS:
# define the list of label names
label_names = ["not", "galaxy"]

train_predictions = model.predict(trainX, batch_size=BS).max(axis=1)
test_predictions = model.predict(testX, batch_size=BS).max(axis=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Accuracy
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.savefig(os.path.join(model_dir,"accuracy.png"))

# Loss
plt.figure()
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.savefig(os.path.join(model_dir,"loss.png"))

# ROC
plt.figure()
plot_roc("Training", trainY, train_predictions, color="b")
plot_roc("Testing", testY, test_predictions, color="r")
plt.legend(loc='lower right')
plt.title("ROC")
plt.savefig(os.path.join(model_dir,"roc.png"))

# PR-curve
plt.figure()
plot_pr("Training", trainY, train_predictions, color="b")
plot_pr("Testing", testY, test_predictions, color="r")
plt.legend()
plt.title("Precision-Recall")
plt.savefig(os.path.join(model_dir,"pr.png"))

# Confusion matrix
plt.figure()
plot_cm(testY, test_predictions)
plt.savefig(os.path.join(model_dir,"confusion.png"))
print("not = 0; galaxies = 1")

plt.show()
