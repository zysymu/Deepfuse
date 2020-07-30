from stamps import AnalyzeImage
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import shutil


ANGTHRESH = None

cutout_sizes = ["500", "1000", "1500"]
sizethreshs = [None, 10, 20, 30, 40, 50, 60, 70]
sbthreshs = [25, 25.5, 26, 26.5, 27]
elthreshs = [0.6, 0.7, 0.8, 0.9]
ring = [True, False]


for c in cutout_sizes:
    for directory in os.listdir("/home/marcostidball/ic-astro/PROJECT/CANDIDATES-FITS/" + c): # gets each new directory

        for sizethresh in sizethreshs:
            for SBTHRESH in sbthreshs:
                for ELTHRESH in elthreshs:
                    for r in ring:

                        full_path = os.path.join("/home/marcostidball/ic-astro/PROJECT/CANDIDATES-FITS/" + c, directory)
                        files = os.listdir(full_path) # list all files inside a directory
                        print("looking at... ", directory)

                        stamps_dir = os.path.join(full_path, "stamps") 
                        os.mkdir(stamps_dir)

                        df_list = []

                        for f in tqdm(files): # goes through the cutout fits files in this directory
                            # saves each source detected in a certain file to a directory
                            try:
                                cutout = AnalyzeImage(os.path.join(full_path,f))
                                cutout.thresholds(sizethresh, SBTHRESH, ELTHRESH, ANGTHRESH)
                                if r:
                                    cutout.apply_ring_filter()
                                else:
                                    cutout.sky()
                                #cutout.show_stamps(f)
                                df = cutout.save_stamps(dir_name=os.path.join(stamps_dir, f.rsplit(".", 1)[0]))
                                df_list.append(df)
                                

                            except AttributeError as e:
                                continue

                        catalog = pd.concat(df_list)
                        name = f"{c}-s{sizethresh}-sb{SBTHRESH}-el{ELTHRESH}-r{r}.csv"
                        catalog.to_csv(os.path.join("/home/marcostidball/ic-astro/PROJECT/CANDIDATES-FITS", name), index=False)
                        shutil.rmtree(stamps_dir)
