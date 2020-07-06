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


def make_and_segment_mosaic(filename, maskfile, cutout_size, overlap_percentage, dir_name):
    """
    Creates a mosaic of a .fits image and segment it in smaller cutouts with some overlap, saving the resulting images to a newly created directory.
    
    -------
    Input:
    filename = str / path to .fits image file
    maskfile = str / path to corresponding .fits mask file
    cutout_size = int / size of the cutout
    overlap_percentage = float / percentage of overlap (0. = completely new cutout, no overlap; 1. = same cutout, total overlap)
    dir_name = str / name of the directory where the stamps are going to be stored
    
    """

    f = fits.open(filename, memmap=True)
    m = fits.open(maskfile, memmap=True)
    print(m)
    orig_header = f[0].header # PrimaryHDU object

    print("finding wcs...")
    wcs_out, shape_out = find_optimal_celestial_wcs(f[1:10]) # has only CompImageHDU files

    
    print("applying mask...")
    for i in range(1, len(f)):
        m[i].data = (m[i].data < 0.5).astype(int)
        f[i].data = f[i].data * m[i].data

    m.close()
    del m

    print("creating mosaic...")
    array, footprint = reproject_and_coadd(f[1:10], wcs_out, shape_out=shape_out, reproject_function=reproject_interp)
    print(array)
    print(type(array))
    
    #mask, footprint = reproject_and_coadd(m[1:4], wcs_out, shape_out=shape_out, reproject_function=reproject_interp)
    #mask = (mask < 0.5).astype(int)
    #array = array * mask
    #del mask
    #del footprint

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
            
            # checks if the segment has values other than 0, ignores the CCD parts with no good pixels
            if not np.all((segment.data == 0)): # returns True if all zeros, use not in front of it to look at the segments with other values only
                header_new = segment.wcs.to_header()
            
                hdu_p = fits.PrimaryHDU(header=orig_header)
                hdu_i = fits.ImageHDU(segment.data, header=header_new)
                hdulist = fits.HDUList([hdu_p,hdu_i])

                output_file = os.path.join(dir_name, str(cen)+".fits")
                hdulist.writeto(output_file, overwrite=True)

                # check out the data (it works!!!)
                #avg = np.mean(np.arcsinh(segment.data))
                #plt.imshow(np.arcsinh(segment.data), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")
                #plt.show()

            col_pos += overlap
            
        row_pos += overlap

    f.close()

###########################################
