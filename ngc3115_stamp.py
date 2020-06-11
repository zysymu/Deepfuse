from stamps import EllipseBBox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import Angle
import astropy.units as u
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch, simple_norm)
import os
from math import ceil
from astropy.nddata.utils import extract_array
from astropy.nddata.utils import overlap_slices
import time


def get_candidate(input_file, output_file, ra, dec, size):
    """
    Extract a stamp from "input_file" according to its WCS positions. From: https://github.com/rodff/LSB_galaxies/blob/master/cutout_decam_image.ipynb

    Parameters:
    -------
    input_file = string
        name of .fits file (original image)

    output_file = string
        name of output .fits file (extracted candidate)

    ra = ra position of the candidate in the image (astropy.coordinates.Angle format)

    dec = dec position of the candidate in the image (astropy.coordinates.Angle format)

    size = int
        lenght of the stamp 
    """

    f = fits.open(input_file, memmap=True)
    orig_header = f[0].header # MAIN info stuff

    for i in range(1, len(f)): # go over the hdul 
        data_ext = f[i].data # image data        
        w_ext = wcs.WCS(f[i].header) # gets WCS stuff of the image
        
        # perform the core WCS transformation from pixel to world coordinates:
        ra_i, dec_i = w_ext.wcs_pix2world(0,0,0) # gets WCS for start of the image
        ra_f, dec_f = w_ext.wcs_pix2world(data_ext.shape[0], data_ext.shape[1], 0) # gets WCS for end of the image
        # wcs_pix2world inputs: an array for each axis, followed by an origin

        if (ra_f < ra < ra_i) and (dec_i < dec < dec_f): # makes sure the values of ra and dec are inside the image (assertion)
            scidata = f[i].data # image data (again?)
            w = wcs.WCS(f[i].header) # WCS stuff (again?)

    position = w.wcs_world2pix(ra, dec, 0) # gets position in pixels

    cutout = Cutout2D(scidata, position, size, wcs=w) # employs cutout retaining wcs info
    header_new = cutout.wcs.to_header() # gives header info to cutout

    # hdu config:
    hdu_p = fits.PrimaryHDU(header=orig_header)
    hdu_i = fits.ImageHDU(cutout.data, header=header_new)
    hdulist = fits.HDUList([hdu_p,hdu_i])
    
    hdulist.writeto(output_file, overwrite=True)


#input_file= 'ngc3115/c4d_170217_075805_osi_g_v2.fits.fz'
#output_file = 'candidate_002_g.fits'

# position of the candidate in the image?, in pixels: (8004,6987)
#ra = Angle('10:06:37.0581 hours').degree
#dec = Angle('-8:29:07.129 degrees').degree

# cutout size
#size = 100

#get_candidate(input_file, output_file)


###########################################


def make_pieces(img_norm, cutout_size, border_size, ps, mzero, sizethresh, dir_name):
    """

    Parameters
    -------
    img_norm = 2D numpy array

    cutout_size = int - size of the cutout

    border_size = float - size of the border, from 0. to 1. (times cutout_size)

    ps = float - pixel scale

    mzero = float - magnitude zero point

    sizethresh = int - size threshold for detected sources

    dir_name = str - name of the directory where the stamps are going to be stored
    """

    """
    borders
    """
    
    #os.mkdir(dir_name)
    
    # stamps
    cutout = (cutout_size, cutout_size) 
    num_images_per_line = ceil(img_norm.shape[0]/cutout_size)

    # borders
    border_vertical = (cutout_size, cutout_size*border_size)
    border_horizontal = (cutout_size*border_size, cutout_size)

    row_pos = ceil(cutout_size/2)
    border_row_pos = cutout_size

    frames = []
    
    for i in range(num_images_per_line): # changes the row
        col_pos = ceil(cutout_size/2)
        border_col_pos = cutout_size

        for j in range(num_images_per_line): # changes the column
            cen = (row_pos, col_pos)
            border_cen_vertical = (row_pos, border_col_pos)
            border_cen_horizontal = (border_row_pos, col_pos)
            
            # cutout
            small = extract_array(img_norm, cutout, cen)
            EllipseBBox(small, ps, mzero, sizethresh).show_stamps(title="center = " + str(cen))
            #df = EllipseBBox(small, ps, mzero, sizethresh).save_stamps(os.path.join(dir_name,str(cen)))
            #frames.append(df)

            # right border
            if j != num_images_per_line: # excludes the edge scenario
                small = extract_array(img_norm, border_vertical, border_cen_vertical)
                _, slices_small = overlap_slices(img_norm.shape, small.shape, border_cen_vertical)
                EllipseBBox(small[slices_small], ps, mzero, sizethresh).show_stamps(title="border center = " + str(border_cen_vertical))
                #df = EllipseBBox(small[slices_small], ps, mzero, sizethresh).save_stamps(os.path.join(dir_name,str(border_cen_vertical)))
                #frames.append(df)

            # upper border
            if i != num_images_per_line:
                small = extract_array(img_norm, border_horizontal, border_cen_horizontal)
                _, slices_small = overlap_slices(img_norm.shape, small.shape, border_cen_horizontal)
                EllipseBBox(small[slices_small], ps, mzero, sizethresh).show_stamps(title="border center = " + str(border_cen_horizontal))
                #df = EllipseBBox(small[slices_small], ps, mzero, sizethresh).save_stamps(os.path.join(dir_name,str(border_cen_horizontal)))
                #frames.append(df)

            col_pos += cutout_size
            border_col_pos += cutout_size

        row_pos += cutout_size
        border_row_pos += cutout_size
        break

    #result = pd.concat(frames)
    #result.reset_index(drop=True, inplace=True)
    #result.to_csv(os.path.join(dir_name, "catalog.csv"), index=False)

    """
    overlap
    """
    


# 300x300 image
#folder = "/home/marcostidball/ic-astro/PROJECT/fits-files/candidate_011"
#img = "candidate_011_r_img.fits"

# big image 
folder = "/home/marcostidball/ic-astro/PROJECT/ngc3115"
img = "c4d_170217_075805_osi_g_v2.fits.fz"

# values for DECAM r-band
ps = 0.27
mzero = 31.395
sizethresh = 50

fits_image_filename = os.path.join(folder,img) # path to .fits image
    
hdul = fits.open(fits_image_filename, memmap=True)
#img_norm = hdul[0].data 
img_norm = hdul[5].data 
hdul.close()

make_pieces(img_norm, 500, 0.3, ps, mzero, sizethresh, "stamps")