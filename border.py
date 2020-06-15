import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import Angle
import astropy.units as u
import os

# testing out the borders
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.coordinates import SkyCoord

###########################################

folder = "/home/marcostidball/ic-astro/PROJECT/ngc3115"
img = "c4d_170217_075805_osi_g_v2.fits.fz"
fits_image_filename = os.path.join(folder,img) # path to .fits image
f = fits.open(fits_image_filename, memmap=True)
#print(f[1:9]) # type: HDUList, has only CompImageHDU files

#print(f[0].header) # PrimaryHDU object (not important for now?)

print(len(f[1:3])) # two images; total len = 10, where f[1:10] = errthing
print(f[1:10])


wcs_out, shape_out = find_optimal_celestial_wcs(f[1:3])
print(wcs_out.to_header())
print(shape_out) # output shape: (8925, 19690) makes sense, two images side by side

array, footprint = reproject_and_coadd(f[1:3], wcs_out, shape_out=shape_out, reproject_function=reproject_interp, match_background=True) # freezes my pc
print(array) 

#plt.imshow(np.arcsinh(array), origin="lower", vmin=8.33, vmax=8.38)
#plt.show()

hdu_p = fits.PrimaryHDU(header=f[0].header)
hdu_i = fits.ImageHDU(array)
hdulist = fits.HDUList([hdu_p,hdu_i])

output_file = os.path.join(os.getcwd(), "mosaic.fits")
hdulist.writeto(output_file, overwrite=True)

f.close()
