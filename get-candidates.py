from stamps import get_candidate
import os
import numpy as np
import pandas as pd


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