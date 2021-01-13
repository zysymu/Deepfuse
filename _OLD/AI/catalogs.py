import os
from shutil import copy, move
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.coordinates import Angle
from astropy import wcs
import matplotlib.pyplot as plt


def true_catalog(catalog_path, mock_dir_path, train_dir_path, mu_0_thresh, mean_mu_e_thresh):
    df = pd.read_csv(catalog_path)

    mu_0_threshl, mu_0_threshu = mu_0_thresh
    mean_mu_e_threshl, mean_mu_e_threshu = mean_mu_e_thresh

    df.drop(df[df["mu_0"] < mu_0_threshl].index, inplace=True) # cuts everything below mu_0_threshl
    df.drop(df[df["mu_0"] > mu_0_threshu].index, inplace=True) # cuts everything above mu_0_threshu

    df.drop(df[df["mean_mu_e"] < mean_mu_e_threshl].index, inplace=True)
    df.drop(df[df["mean_mu_e"] > mean_mu_e_threshu].index, inplace=True)
    print(df)

    galaxy_dir = os.path.join(train_dir_path, "galaxies")
    os.mkdir(galaxy_dir)

    for index, row in df.iterrows():
        name = int(row["ID"])
        d = f"gal_{name}"
        inside = os.path.join(mock_dir_path, d) # directory
        filename = f"mock_{name}_g.fits"
        
        f = fits.open(os.path.join(inside, filename), memmap=True)

        data = f[1].data
        w = wcs.WCS(f[1].header)

        row["size"] = row["re_R"] * 7
        stamp = Cutout2D(data, position=(data.shape[0]/2, data.shape[0]/2), size=(row["size"],row["size"]), wcs=w, copy=True)

        """
        avg = np.mean(np.arcsinh(data))
        plt.imshow(np.arcsinh(data), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")
        plt.colorbar()

        stamp.plot_on_original(color='red')
        plt.show()
        """        

        hdu = fits.PrimaryHDU(stamp.data)
        hdul = fits.HDUList([hdu])
        stamp_filename = os.path.join(galaxy_dir, filename)
        hdul.writeto(stamp_filename)

    
def false_catalog(catalog_path, cutout_dir_path, train_dir_path, sample):
    df = pd.read_csv(catalog_path)

    #
    # here we apply another function to delete replicas (for now doing it manually is okay)
    #

    df = df.sample(n=sample).reset_index(drop=True)
    print(df)

    stamp_dir = os.path.join(train_dir_path, "not")
    os.mkdir(stamp_dir)

    for index, row in df.iterrows():
        d = os.path.join(cutout_dir_path, row["mosaic"])
        d2 = os.path.join(d, os.path.join("stamps",row["association"]))

        nam = str(row["name"]) + ".fits"
        f = os.path.join(d2, nam) # complete file name

        if nam in os.listdir(stamp_dir):
            i = row["ra"]
            nam_sub = f"{i}.fits"
            old = os.path.join(stamp_dir, nam)
            new = os.path.join(stamp_dir, nam_sub)
            os.rename(old, new)
            copy(f, stamp_dir)
        else:
            copy(f, stamp_dir)

        
MOCK_CATALOG = "/home/marcostidball/ic-astro/PROJECT/AI/mocks_marcos/point_050619_g.csv"
MOCK_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/mocks_marcos/point_050619_g"
TRAIN_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/train"
sbthresh = 25

STAMPS_CATALOG = "/home/marcostidball/ic-astro/PROJECT/AI/catalog_final.csv"
CUTOUT_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/fo-real3"
sample = 400

mu_0_thresh = (22.5, 25.5)
mean_mu_e_thresh = (24.5, 27.5)
true_catalog(MOCK_CATALOG, MOCK_DIR_PATH, TRAIN_DIR_PATH, mu_0_thresh, mean_mu_e_thresh)
false_catalog(STAMPS_CATALOG, CUTOUT_DIR_PATH, TRAIN_DIR_PATH, sample) # BIG PROBLEM, HAVE TO RERUN MAKE-STAMPS.PY IN ORDER TO FIX IT
