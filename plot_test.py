import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch, simple_norm)
import os

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
