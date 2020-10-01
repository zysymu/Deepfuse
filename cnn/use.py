from deepfuse import Classifier
from dataset import Dataset
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

tf.get_logger().setLevel('INFO')

TRAIN_DIR_PATH = "train-cuts"
TEST_DIR_PATH = "CANDIDATE-STAMPS"
img_dim = 100
scale = None  #vmin =2065#-2360; #vmax =  2360#2390????
n = "eff-net-ALLSET"

# training
dataset = Dataset(img_dim=img_dim, vscale=scale)
#train_data, labels = dataset.training(data_dir_path=TRAIN_DIR_PATH, dir_names=("not", "NEW-GALAXIES"))

model = Classifier(img_dim=img_dim, batch_size=64)
"""
model.fit(data=train_data, labels=labels, model_type="EfficientNetB0", epochs=20, optimizer=tfa.optimizers.RectifiedAdam(lr=1e-4, decay=1e-6))
model.save(n + ".hdf5")
model.evaluate(n, show=True)
"""

# testing
test_data, f_names = dataset.classifying(data_dir_path=TEST_DIR_PATH)
model.load(n + ".hdf5")
model.classify(data=test_data, f_names=f_names)

#1 - 1
#6 - 2