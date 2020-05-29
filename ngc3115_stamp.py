from stamps import EllipseBBox
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch, simple_norm)
import os


def extract_stamps(ps, mzero, sizethresh, SBthresh=None):
    """
    Take .fits files in the folder and extract its stamps
    """
    pwd = os.getcwd()
    path_imgs = os.path.join(pwd, "ngc3115")
    filenames = os.listdir(path_imgs)

    for filename in filenames:
        if filename[-12:] == ".resamp.fits":
            fits_image_filename = os.path.join(path_imgs, filename)
            hdul = fits.open(fits_image_filename)

            img_norm = hdul[0].data
            data = img_norm.byteswap().newbyteorder()
            EllipseBBox(data, ps, mzero, sizethresh).save_stamps(dir_name=filename)


# values for DECAM (r-band)
ps = 0.27
mzero = 31.395
sizethresh = 30

extract_stamps(ps, mzero, sizethresh)




"""
here = os.getcwd()
folder = os.path.join(here, "candidate_032")
for img in os.listdir(folder):
    fits_image_filename = os.path.join(folder,img)
        
    hdul = fits.open(fits_image_filename)
    img_norm = hdul[0].data

    data = img_norm.byteswap().newbyteorder()

    imshow_norm(np.arcsinh(data), origin='lower', interval=MinMaxInterval(), stretch=SqrtStretch(), cmap="binary_r")
    plt.colorbar()

    plt.show()
"""