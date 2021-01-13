from dataloader import DataLoader
from classifier import Classifier
import tensorflow_addons as tfa
import os
import pandas as pd

# Loading data
# params
TRAIN_DIR_PATH = "/content/drive/My Drive/Deepfuse/train-cuts"
TEST_DIR_PATH = "/content/drive/My Drive/Deepfuse/candidates-iguess"

global img_dim, scale, EPOCHS, BS
img_dim = 150
scale = (2, 20)
EPOCHS = 30
BS = 128

# training
dataloader = DataLoader(img_dim=img_dim, vscale=scale)
train_data, labels = dataloader.training(data_dir_path=TRAIN_DIR_PATH, dir_names=("not", "galaxies"))

# real objects
test_data=0; test_files=0
#test_data, test_files = dataloader.classifying(data_dir_path=os.path.join(TRAIN_DIR_PATH, "not")) # check only real objects

# candidates
candidates_data, candidates_fnames = dataloader.classifying(data_dir_path=TEST_DIR_PATH)

def test_params(n, model_type, s, train_data, labels, test_data, test_files, candidates_data, candidates_fnames):
    # training
    model = Classifier(img_dim=img_dim, batch_size=BS)

    model.fit(data=train_data, labels=labels, model_type=model_type, epochs=EPOCHS, optimizer=tfa.optimizers.RectifiedAdam(lr=1e-4, decay=1e-6))
    #model.fit(data=train_data, labels=labels, model_type=model_type, epochs=EPOCHS, optimizer=tf.keras.optimizers.RMSprop()) # default RMSprop / doesnt work
    #model.save(n + ".hdf5")
    df, tn, fp, fn, tp = model.evaluate(n, candidates_data=candidates_data, candidates_fnames=candidates_fnames, show=False)

    # testing with training data
    #test_preds = model.classify(data=test_data, fnames=test_files)
    #test_preds.to_csv(os.path.join(s,"nontest_preds.csv"))

    # candidates
    gals = df["label"].sum()
    non = len(candidates_data) - gals

    df = pd.DataFrame({"model":[model_type], "true_negatives":[tn], "false_positives":[fp], "false_negatives":[fn], "true_positives":[tp], "galaxies": [gals], "non":[non], "epochs":[EPOCHS], "img_size":[img_dim], "batch":[BS]})
    return df

#model_types_lens = ["lens-3", "lens-4", "lens-5", "lens-6"]
model_types_eff = ["effX-3", "effX-4", "effX-5", "effX-6"]
model_types_vgg16 = ["VGG16-2", "VGG16-3", "VGG16-4"]
model_types_vgg19 = ["VGG19-2", "VGG19-3", "VGG19-4"]
model_types_resnet = ["resnet-2", "resnet-3", "resnet-4"]

for _ in range(10):
    if not (_ == 1 or _ == 2 or _ == 3):
        local = f"/content/drive/My Drive/Deepfuse/scale2-20_{_}/{EPOCHS}epochs-{img_dim}px-{BS}batch-RAdam"
        #os.mkdir(local)
        dfs = []

        # lens:
        """
        a = os.path.join(local, "lens")
        os.mkdir(a)

        for m in model_types_lens:
            n = os.path.join(a, m)
            df = test_params(n, m, 200, s=n)
            dfs.append(df)
        """
        """        
        # effnet:
        #for i in range(0,3): # 0-2    
        for i in range(3,5): # 3-4        
            a = os.path.join(local, "effnetB{i}".format(i=i))
            os.mkdir(a)

            for m in model_types_eff: # cycle through blocks to freeze
                m = m.replace("X", str(i)) # adjust block in model name
                n = os.path.join(a, m)
                df = test_params(n, m, n, train_data, labels, test_data, test_files, candidates_data, candidates_fnames)
                dfs.append(df)
        """
        
        # vgg16
        a = os.path.join(local, "vgg16")
        os.mkdir(a)

        for m in model_types_vgg16:
            n = os.path.join(a, m)
            df = test_params(n, m, n, train_data, labels, test_data, test_files, candidates_data, candidates_fnames)
            dfs.append(df)

        # vgg19
        a = os.path.join(local, "vgg19")
        os.mkdir(a)

        for m in model_types_vgg19:
            n = os.path.join(a, m)
            df = test_params(n, m, n, train_data, labels, test_data, test_files, candidates_data, candidates_fnames)
            dfs.append(df)

        """
        # resnet
        a = os.path.join(local, "resnet")
        os.mkdir(a)

        for m in model_types_resnet:
            n = os.path.join(a, m)
            df = test_params(n, m, n, train_data, labels, test_data, test_files, candidates_data, candidates_fnames)
            dfs.append(df)
        """

        tables = pd.concat(dfs)
        tables.to_csv(os.path.join(local,"data1.csv"))
