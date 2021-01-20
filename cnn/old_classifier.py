import os
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
from astropy.io import fits
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

class Classifier():
    def __init__(self, img_dim, batch_size): #if model == lens, img_dims = (200,200)
        self.img_dim = img_dim
        self.input_shape = (img_dim, img_dim, 3)

        self.batch_size = batch_size

    def fit(self, data, labels, model_type, epochs, optimizer):
        self.data = data
        self.labels = labels

        # Split training and validation data (train = 60%; val = 20%; test = 20%)
        trainX, self.testX, trainY, self.testY = train_test_split(self.data, self.labels, test_size=0.20)#, random_state=42)
        self.trainX, valX, self.trainY, valY = train_test_split(trainX, trainY, test_size=0.25)#, random_state=42)

        print("size of training data: ", len(self.trainX))
        print("size of validation data: ", len(valX))
        print("size of test data: ", len(self.testX))

        weights = class_weight.compute_class_weight('balanced', np.unique(self.trainY), self.trainY)
        weights_dict = dict(enumerate(weights))

        print("setting up model...")

        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.5)
        ])

        # Create model
        # LENS
        if model_type.split("-")[0] == "lens":
            base = keras.models.load_model("/content/drive/My Drive/Deepfuse/efn0_vis.hdf5", custom_objects={'RAdam': tfa.optimizers.RectifiedAdam})

            block = model_type.split("-")[1]
            l = "block" + block + "a_expand_conv"
            
        # EFFNETS
        elif model_type.split("-")[0][:3] == "eff":
            if model_type.split("-")[0] == "eff0":
                base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff1":
                base = keras.applications.EfficientNetB1(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff2":
                base = keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff3":
                base = keras.applications.EfficientNetB3(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff4":
                base = keras.applications.EfficientNetB4(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff5":
                base = keras.applications.EfficientNetB5(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff6":
                base = keras.applications.EfficientNetB6(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "eff7":
                base = keras.applications.EfficientNetB7(include_top=False, weights="imagenet", input_shape=self.input_shape)

            block = model_type.split("-")[1]
            l = "block" + block + "a_expand_conv"

        # VGGS
        elif model_type.split("-")[0][:3] == "VGG":
            if model_type.split("-")[0] == "VGG16": # 2, 3, 4
                base = keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=self.input_shape)

            elif model_type.split("-")[0] == "VGG19": # 2, 3, 4
                base = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=self.input_shape)

            block = model_type.split("-")[1]
            l = "block" + block + "_conv1"

        # RESNET
        elif model_type.split("-")[0] == "resnet": # 2, 3, 4
            base = keras.applications.ResNet101(include_top=False, weights="imagenet", input_shape=self.input_shape)

            block = model_type.split("-")[1]
            l = "conv" + block + "_block1_1_conv"

        # Build model
        base.trainable = True
        set_trainable = False

        for layer in base.layers:
            if layer.name == l:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        self.model = keras.models.Sequential([
            data_augmentation,
            base,
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.build((None, self.img_dim, self.img_dim, 3))
        print(self.model.summary())

        # Compile model
        #earlystopper = keras.callbacks.EarlyStopping(monitor="loss", patience=5, verbose=1)
        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        print(np.shape(self.trainX), np.shape(self.trainY))

        #history = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, steps_per_epoch=len(trainX)//BS, validation_data=(valX, valY), validation_steps = len(valX)//BS, class_weight=weights_dict, callbacks=[reduce_lr, earlystopper], verbose=1)
        self.history = self.model.fit(self.trainX, self.trainY, batch_size=self.batch_size, epochs=epochs, steps_per_epoch=len(self.trainX)//self.batch_size, validation_data=(valX, valY), validation_steps = len(valX)//self.batch_size, class_weight=weights_dict, verbose=1)

    def save(self, path):
        self.model.save(path, save_format="hdf5")

    def evaluate(self, eval_dir_path, candidates_data, candidates_fnames, show=False):
        os.mkdir(eval_dir_path)

        train_predictions = self.model.predict(self.trainX, batch_size=self.batch_size).max(axis=1)
        test_predictions = self.model.predict(self.testX, batch_size=self.batch_size).max(axis=1)

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # Accuracy
        plt.figure()
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title("Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(eval_dir_path,"accuracy.png"))

        # Loss
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(eval_dir_path,"loss.png"))

        # ROC
        plt.figure()
        self._plot_roc("Training", self.trainY, train_predictions, color="b")
        self._plot_roc("Testing", self.testY, test_predictions, color="r")
        plt.legend(loc='lower right')
        plt.title("ROC")
        plt.savefig(os.path.join(eval_dir_path,"roc.png"))

        # PR-curve
        plt.figure()
        self._plot_pr("Training", self.trainY, train_predictions, color="b")
        self._plot_pr("Testing", self.testY, test_predictions, color="r")
        plt.legend()
        plt.title("Precision-Recall")
        plt.savefig(os.path.join(eval_dir_path,"pr.png"))

        # Confusion matrix
        plt.figure()
        tn, fp, fn, tp = self._cm(self.testY, test_predictions)
        plt.savefig(os.path.join(eval_dir_path,"confusion.png"))

        # Candidates
        df = self.classify(candidates_data, candidates_fnames)
        df.to_csv(os.path.join(eval_dir_path,"candidates.csv"))

        if show:
            plt.show()
        else:
            plt.close("all")

        return df, tn, fp, fn, tp
       
    def load(self, model_path):
        self.model = keras.models.load_model(model_path)

    def classify(self, data, fnames):
        preds = self.model.predict(data, batch_size=self.batch_size).max(axis=1)
        df = pd.DataFrame({"file":fnames, "prediction":preds, "label":np.where(preds>0.5, 1, 0)})
        
        return df

    def _plot_roc(self, name, labels, predictions, color):
        fp, tp, _ = roc_curve(labels, predictions)
        auroc = auc(fp, tp)

        auroc = float("{:.2f}".format(auroc))
        n = f"{name}; auroc = {auroc}"

        plt.plot(fp, tp, label=n, color=color)
        plt.xlabel('False Positives')
        plt.ylabel('True Positives')
        ax = plt.gca()
        ax.set_aspect('equal')

    def _plot_pr(self, name, labels, predictions, color):
        precision, recall, _ = precision_recall_curve(labels, predictions)
        auroc = auc(recall, precision)

        auroc = float("{:.2f}".format(auroc))
        n = f"{name}; auroc = {auroc}"

        plt.plot(recall, precision, label=n, color=color)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        ax = plt.gca()
        ax.set_aspect('equal')

    def _cm(self, labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        #tn, fp, fn, tp = confusion_matrix(labels, predictions > p).ravel()
        tn, fp, fn, tp = cm.ravel()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        return tn, fp, fn, tp
