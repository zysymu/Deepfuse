from stamps import EllipseBBox
from stamps import get_candidate
from mosaic import make_and_segment_mosaic
from astropy.io import fits
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u



# CREATE CUTOUTS FROM MOSAIC
"""
fits_files = ["/data/william/ngc3115/c4d_170216_050619_osi_g_v2.fits.fz", "/data/william/ngc3115/c4d_170216_061516_osi_r_v2.fits.fz"]
mask_files = ["c4d_170216_050619_osd_g_v2.fits.fz", "c4d_170216_061516_osd_r_v2.fits.fz"]


#os.mkdir("/data/marcos-cutouts/cutouts")

for i, filename in tqdm(enumerate(fits_files)):
    # change the filename split according to the file adress
    dir_name = filename.split("/")[4].split(".")[0]
    print("looking at... ", dir_name)

    full_path = os.path.join("/data/marcos-cutouts", dir_name)
    mask = mask_files[i]
    make_and_segment_mosaic(filename, mask, 1000, 0.2, full_path)
    # inside each directory we'll have all the cutouts from each .fits.fz file


"""

# EXTRACT SOURCES FROM CUTOUTS

# values for DECAM r-band
ps = 0.27
sizethresh = None #50
SBTHRESH = 26
ELTHRESH = 0.7
HRTHRESH = 60


for directory in os.listdir("/home/marcostidball/ic-astro/PROJECT/CUTOUTS-TESTS/testing-masks/masktest"): # gets each new directory
    full_path = os.path.join("/home/marcostidball/ic-astro/PROJECT/CUTOUTS-TESTS/testing-masks/masktest", directory)
    files = os.listdir(full_path) # list all files inside a directory
    print("looking at... ", directory)

    stamps_dir = os.path.join(full_path, "stamps") 
    #os.mkdir(stamps_dir)

    df_list = []

    for f in tqdm(files): # goes through the cutout fits files in this directory
        # saves each source detected in a certain file to a directory
        try:
            #df = EllipseBBox(os.path.join(full_path,f), ps, sizethresh, SBTHRESH, ELTHRESH, HRTHRESH).save_stamps(dir_name=os.path.join(stamps_dir, f.rsplit(".", 1)[0]))
            #df_list.append(df)
            EllipseBBox(os.path.join(full_path,f), ps, sizethresh, SBTHRESH, ELTHRESH, HRTHRESH).show_stamps(f)
        except AttributeError as e:
            continue

    #catalog = pd.concat(df_list)
    #catalog.to_csv(os.path.join(full_path, "catalog.csv"), index=False)


# GET CANDIDATE
"""
input_file= ['/data/william/ngc3115/c4d_170217_075805_osi_g_v2.fits.fz', '/data/william/ngc3115/c4d_170218_051026_osi_r_v2.fits.fz']
df = pd.read_csv("coordenadas_candidatas.csv")

size = 1500

for f in input_file:
    dir_name = f.split("/")[4].split(".")[0]
    dir_path = os.path.join("/data/marcos-cutouts/new-tile-candidates", dir_name)
    os.mkdir(dir_path)

    for index, row in df.iterrows():
        ra = row["ra"]
        dec = row["dec"]

        output_dir = dir_path
        output_file = os.path.join(dir_path, str(row["id_1"])) + ".fits"

        try:
            get_candidate(f, output_file, ra, dec, size) # gets position in pixels
        except UnboundLocalError as e:
            continue
"""



