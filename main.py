from stamps import EllipseBBox
#from stamps import get_candidate
from mosaic import make_and_segment_mosaic
from astropy.io import fits
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

fits_files = ["c4d_170217_075805_osi_g_v2.fits.fz"]

dir_list = []

os.mkdir("cutouts")

for filename in tqdm(fits_files):
    dir_name = filename.split(".")[0]
    dir_list.append(dir_name)
    print("looking at... ", dir_name)

    full_path = os.path.join("cutouts", dir_name)
    make_and_segment_mosaic(filename, 1000, 0.1, full_path)
    # inside each directory we'll have all the cutouts from each .fits.fz file


# values for DECAM r-band
ps = 0.27
mzero = 31.395
sizethresh = 50



for directory in dir_list: # gets each new directory
    full_path = os.path.join("cutouts", directory)
    files = os.listdir(full_path) # list all files inside a directory
    print("looking at... ", directory)

    stamps_dir = os.path.join(full_path, "stamps") 
    os.mkdir(stamps_dir)

    df_list = []

    for f in tqdm(files): # goes through the cutout fits files in this directory
        hdul = fits.open(os.path.join(full_path,f), memmap=True)
        img = hdul[1].data
        img = img.byteswap().newbyteorder() 
        hdul.close()
        
        # saves each source detected in a certain file to a directory
        df = EllipseBBox(img, ps, mzero, sizethresh).save_stamps(dir_name=os.path.join(stamps_dir, f.rsplit(".", 1)[0]))
        df_list.append(df)

    catalog = pd.concat(df_list)
    catalog.to_csv(os.path.join(full_path, "catalog.csv"), index=False)
