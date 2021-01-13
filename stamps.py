from scan import Scanner
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
import numpy as np

STAMPSIZE = 150
"""
# simulations
sim_dir = "/home/marcostidball/ic-astro/PROJECT/AI/subsamples/sub_point_061516_r"
final_dir = "/home/marcostidball/ic-astro/PROJECT/AI/train-cuts/NEW-GALAXIES"

for d in os.listdir(sim_dir):
    num = d.split("_")[1]

    full_path = os.path.join(sim_dir, d)
    print(f"looking at... {d}")    

    for f in os.listdir(full_path):
        f_path = os.path.join(full_path, f)
        
        if f == f"mock_{num}_r.fits":
            try:
                scan = Scanner(filename=f_path, stamp_size=STAMPSIZE, mode="simulation-30")
                scan.thresholds()
                #scan.show_stamps(title=f)
                scan.save_stamps(dir_name=os.path.join(final_dir, f), catalog=False)
    
            except AssertionError as e:
                print(e)
                continue
"""

# candidates
candidate_dir = "/home/marcostidball/Desktop/Deepfuse/candidates-iguess"
final_dir = "/home/marcostidball/ic-astro/PROJECT/AI/CANDIDATE-STAMPS"

table = "/home/marcostidball/ic-astro/PROJECT/NGC3115_candidates.fits"

hdul = fits.open(table)
data = hdul[1].data

#os.mkdir(final_dir)

for f in os.listdir(candidate_dir):
    f_path = os.path.join(candidate_dir, f)

    try:   
        for i, n in enumerate(data["paper_id"]):
            if str(n) == f_path.split("/")[-1].split(".")[0]:
                ra = data["ra"][i]
                dec = data["dec"][i]
                print(ra, dec)                

                hdu_wcs = fits.open(f_path)
                w = wcs.WCS(hdu_wcs[1].header)
                x, y = w.wcs_world2pix(ra, dec, 1)
                print(x,y)
    
        f = fits.open(f_path, memmap=True)
        a = f[1].data
        avg = np.mean(np.arcsinh(a))
        plt.imshow(np.arcsinh(a), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")

        plt.colorbar()
        plt.title(f_path.split("/")[-1])
        plt.scatter(x, y)
        plt.show()
        #scan.save_stamps(dir_name=os.path.join(final_dir, f), catalog=False)

    except AssertionError as e:
        print(e)
        continue
