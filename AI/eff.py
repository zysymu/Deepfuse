from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def true_catalog(catalog_path, mock_dir_path, train_dir_path, mu_0_thresh, mean_mu_e_thresh):
    df = pd.read_csv(catalog_path)

    mu_0_threshl, mu_0_threshu = mu_0_thresh
    mean_mu_e_threshl, mean_mu_e_threshu = mean_mu_e_thresh

    df.drop(df[df["mu_0"] < mu_0_threshl].index, inplace=True) # cuts everything below mu_0_threshl
    df.drop(df[df["mu_0"] > mu_0_threshu].index, inplace=True) # cuts everything above mu_0_threshu

    df.drop(df[df["mean_mu_e"] < mean_mu_e_threshl].index, inplace=True)
    df.drop(df[df["mean_mu_e"] > mean_mu_e_threshu].index, inplace=True)

    galaxy_dir = os.path.join(train_dir_path, "galaxies")
    #os.mkdir(galaxy_dir)


    for index, row in df.iterrows():
        name = int(row["ID"])
        d = f"gal_{name}"
        inside = os.path.join(mock_dir_path, d) # directory
        filename = f"mock_{name}_g.fits"
        f = fits.open(os.path.join(inside, filename), memmap=True)

        data = f[1].data
        w = wcs.WCS(f[1].header)
        #re,re_R

        row["xcen"], row["ycen"] = w.wcs_world2pix(row["RA"], row["DEC"], 0) 
        row["size"] = row["re_R"] * 7

        stamp = Cutout2D(data, position=(data.shape[0]/2, data.shape[0]/2), size=(row["size"],row["size"]), wcs=w, copy=True)

        avg = np.mean(np.arcsinh(data))
        plt.imshow(np.arcsinh(data), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")
        plt.colorbar()

        stamp.plot_on_original(color='red')
        plt.show()


MOCK_CATALOG = "/home/marcostidball/ic-astro/PROJECT/AI/mocks_marcos/point_050619_g.csv"
MOCK_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/mocks_marcos/point_050619_g"
TRAIN_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/train"

mu_0_thresh = (22.5, 25.5)
mean_mu_e_thresh = (24.5, 27.5)
true_catalog(MOCK_CATALOG, MOCK_DIR_PATH, TRAIN_DIR_PATH, mu_0_thresh, mean_mu_e_thresh)
