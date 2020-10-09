from deepfuse import Classifier
from dataset import Dataset
import tensorflow_addons as tfa
import os

def test_params(n, model_type, d, s):
    TRAIN_DIR_PATH = "/home/hdd2T/marcos_dados/train-cuts"
    TEST_DIR_PATH = "/home/hdd2T/marcos_dados/CANDIDATE-STAMPS"
    img_dim = d
    scale = (2, 24)
    #n = "eff-net-ALLSET"

    # training
    dataset = Dataset(img_dim=img_dim, vscale=scale)
    train_data, labels = dataset.training(data_dir_path=TRAIN_DIR_PATH, dir_names=("not", "NEW-GALAXIES"))

    model = Classifier(img_dim=img_dim, batch_size=64)

    model.fit(data=train_data, labels=labels, model_type=model_type, epochs=20, optimizer=tfa.optimizers.RectifiedAdam(lr=1e-4, decay=1e-6))
    #model.save(n + ".hdf5")
    model.evaluate(n, show=False)
    
    # testing
    test_data, f_names = dataset.classifying(data_dir_path=TEST_DIR_PATH)
    #model.load(n + ".hdf5")
    model.classify(data=test_data, f_names=f_names, output=os.path.join(s,"score.txt"))
    
    #1 - 1
    #6 - 2


model_types_lens = ["lens-2", "lens-3", "lens-4", "lens-5", "lens-6", "lens-7"]
model_types_eff = ["effX-2", "effX-3", "effX-4", "effX-5", "effX-6", "effX-7"]
local = "/home/hdd2T/marcos_dados/"

# lens:
a = os.path.join(local, "lens-64batch")
os.mkdir(a)
for m in model_types_lens:
    n = os.path.join(a, m)
    test_params(n, m, 200, s=n)

# effnet:
for i in range(8): # 0-7
    a = os.path.join(local, "effnetB{i}-64batch".format(i=i))
    os.mkdir(a)

    for m in model_types_eff: # cycle through blocks to freeze
        m = m.replace("X", str(i)) # adjust block in model name
        n = os.path.join(a, m)
        test_params(n, m, 150, s=n)

# eff0-4 had the best results: 15 - 100px and 17 - 150 px
# classified as galaxies: 150: 83-219; 100: 87-218
