from stamps import EllipseBBox
#from stamps import get_candidate
from mosaic import make_and_segment_mosaic
from astropy.io import fits
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

"""
# create the cutouts:

fits_files = ["/data/william/ngc3115/c4d_170216_050619_osi_g_v2.fits.fz", "/data/william/ngc3115/c4d_170216_061516_osi_r_v2.fits.fz"]
mask_files = ["c4d_170216_050619_osd_g_v2.fits.fz", "c4d_170216_061516_osd_r_v2.fits.fz"]


#os.mkdir("/data/marcos-cutouts/cutouts")

for i, filename in tqdm(enumerate(fits_files)):
    # change the filename split according to the file adress
    dir_name = filename.split("/")[4].split(".")[0]
    print("looking at... ", dir_name)

    full_path = os.path.join("/data/marcos-cutouts", dir_name)
    mask = mask_files[i]
    make_and_segment_mosaic(filename, mask, 1000, 0.1, full_path)
    # inside each directory we'll have all the cutouts from each .fits.fz file


"""
# analyze the cutouts turning them into stamps for each source

# values for DECAM r-band
ps = 0.27
#mzero = 31.395 #MAGZERO
sizethresh = 50
SBthresh = (24.3, 28.8)


for directory in os.listdir("/home/marcostidball/ic-astro/PROJECT/marcos-cutouts"): # gets each new directory
    full_path = os.path.join("marcos-cutouts", directory)
    files = os.listdir(full_path) # list all files inside a directory
    print("looking at... ", directory)

    stamps_dir = os.path.join(full_path, "stamps") 
    os.mkdir(stamps_dir)

    df_list = []

    for f in tqdm(files): # goes through the cutout fits files in this directory
        # saves each source detected in a certain file to a directory
        df = EllipseBBox(os.path.join(full_path,f), ps, sizethresh, SBthresh).save_stamps(dir_name=os.path.join(stamps_dir, f.rsplit(".", 1)[0]))
        df_list.append(df)

    catalog = pd.concat(df_list)
    catalog.to_csv(os.path.join(full_path, "catalog.csv"), index=False)

