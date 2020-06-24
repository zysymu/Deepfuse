from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.nddata.utils import extract_array
from astropy import wcs
from astropy.coordinates import Angle
import astropy.units as u
import os
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs
from astropy.coordinates import SkyCoord

###########################################


def make_and_segment_mosaic(filename, cutout_size, overlap_percentage, dir_name):
    """
    Creates a mosaic of a .fits image and segment it in smaller cutouts with some overlap, saving the resulting images to a newly created directory.
    
    -------
    Input:
    filename = str / path to .fits image
    cutout_size = int / size of the cutout
    overlap_percentage = float / percentage of overlap (0. = completely new cutout, no overlap; 1. = same cutout, total overlap)
    dir_name = str / name of the directory where the stamps are going to be stored
    
    """

    f = fits.open(filename, memmap=True)
    orig_header = f[0].header # PrimaryHDU object

    print("finding wcs...")
    wcs_out, shape_out = find_optimal_celestial_wcs(f[1:3]) # has only CompImageHDU files

    print("creating mosaic...")
    array, footprint = reproject_and_coadd(f[1:3], wcs_out, shape_out=shape_out, reproject_function=reproject_interp)

    #plt.imshow(np.arcsinh(array), origin="lower", vmin=8.33, vmax=8.38)
    #plt.show()
    
    # segment
    os.mkdir(dir_name)
    cutout = (cutout_size, cutout_size) 
    overlap = cutout_size * (1-overlap_percentage) # if cutout_size = 500 px, setting overlap = 0.2 would leave 100 pixels overlapping 
    num_images_per_row = ceil(shape_out[1]/overlap)
    num_images_per_column = ceil(shape_out[0]/overlap)

    print("mosaic shape: ", shape_out)
    print("num_images_per_row: ", num_images_per_row)
    print("num_images_per_column: ", num_images_per_column)
    
    row_pos = ceil(cutout_size/2)
    
    for i in range(num_images_per_row): # changes the row
        col_pos = ceil(cutout_size/2)

        for j in range(num_images_per_column): # changes the column
            cen = (row_pos, col_pos)
            print("processing cutout at position ", str(cen))

            segment = Cutout2D(array, position=cen, size=cutout, wcs=wcs_out, copy=True)
            header_new = segment.wcs.to_header()
            
            hdu_p = fits.PrimaryHDU(header=orig_header)
            hdu_i = fits.ImageHDU(segment.data, header=header_new)
            hdulist = fits.HDUList([hdu_p,hdu_i])

            output_file = os.path.join(dir_name, str(cen)+".fits")
            hdulist.writeto(output_file, overwrite=True)

            #plt.imshow(np.arcsinh(segment), origin="lower", vmin=8.33, vmax=8.38)
            #plt.show()

            col_pos += overlap
            
        row_pos += overlap

    f.close()

###########################################