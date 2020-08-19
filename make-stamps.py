from stamps import AnalyzeImage
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

#cutout_sizes = 1000
SIZETHRESH = 10
SBTHRESH = 25
ELTHRESH = 0.7
I50THRESH = 6
I50AVTHRESH = 9
R50THRESH = 5
ANGTHRESH = None

r = False

cutout_dir = "/home/marcostidball/ic-astro/PROJECT/AI/fo-real3"

for directory in os.listdir(cutout_dir): # directory where the cutouts are stored
    full_path = os.path.join(cutout_dir, directory)
    files = os.listdir(full_path)  # list all files inside a directory
    print("looking at... ", directory)

    stamps_dir = os.path.join(full_path, "stamps")
    os.mkdir(stamps_dir)

    df_list = []

    for f in tqdm(files):  # goes through the cutout fits files in this directory
        # saves each source detected in a certain file to a directory
        try:
            cutout = AnalyzeImage(os.path.join(full_path, f))
            cutout.thresholds(SIZETHRESH, SBTHRESH, ELTHRESH, I50THRESH, I50AVTHRESH, R50THRESH, ANGTHRESH)
            if r:
                cutout.apply_ring_filter()
            else:
                cutout.subtract_sky()
            #cutout.show_stamps(f)
            df = cutout.save_stamps(dir_name=os.path.join(stamps_dir, f.rsplit(".", 1)[0]))
            df_list.append(df)

        except AttributeError as e:
            continue

    catalog = pd.concat(df_list)
    catalog.to_csv(os.path.join(full_path, "catalog.csv"), index=False)
