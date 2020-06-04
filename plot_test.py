from stamps import EllipseBBox
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch, simple_norm)
import os


def extract_stamps(ps, mzero, sizethresh, SBthresh=None):
    
    #Take .fits files in the folder and extract its stamps
    
    pwd = os.getcwd()
    path_imgs = os.path.join(pwd, "fits-files")
    filenames = os.listdir(path_imgs)

    for filename in filenames:
        folder = os.path.join(path_imgs, filename)
        img = filename + "_r_img.fits"
        fits_image_filename = os.path.join(folder,img) # path to .fits image
        print(filename)

        hdul = fits.open(fits_image_filename)
        img_norm = hdul[0].data
        data = img_norm.byteswap().newbyteorder()
        #EllipseBBox(data, ps, mzero, sizethresh).save_stamps(dir_name=filename)
        EllipseBBox(data, ps, mzero, sizethresh).show_stamps(title=filename)
        # IT WOOOOORKS!!!!!!!!! :DDD


# values for DECAM (r-band)
ps = 0.27
mzero = 31.395
sizethresh = 15
#SBthresh = 22

extract_stamps(ps, mzero, sizethresh)#, SBthresh)



here = os.getcwd()
folder = os.path.join(here, "candidate_032")
for img in os.listdir(folder):
    fits_image_filename = os.path.join(folder,img)
        
    hdul = fits.open(fits_image_filename)
    img_norm = hdul[0].data

    data = img_norm.byteswap().newbyteorder()

    #plt.imshow(np.arcsinh(data), origin='lower', vmin=8., vmax=9., cmap="binary_r")
    imshow_norm(np.arcsinh(data), origin='lower', interval=MinMaxInterval(), stretch=SqrtStretch(), cmap="binary_r")
    plt.colorbar()

    plt.show()
