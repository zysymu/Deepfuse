import os
import numpy as np
import pandas as pd
import astropy
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from reproject.mosaicking import find_optimal_celestial_wcs


#input_files= ['/data/william/ngc3115/c4d_170217_075805_osi_g_v2.fits.fz', '/data/william/ngc3115/c4d_170216_050619_osi_g_v2.fits.fz']
input_files= ['/data/william/ngc3115/c4d_170216_050619_osi_g_v2.fits.fz']
df = pd.read_csv("coordenadas_candidatas.csv")

size = 1000

for input_file in input_files:
    
    dir_name = input_file.split("/")[4].split(".")[0]
    dir_path = os.path.join("/data/marcos-cutouts/candidates-1000l", dir_name)
    os.mkdir(dir_path)  

    f = fits.open(input_file, memmap=True)
    orig_header = f[0].header # MAIN info stuff
    
    print("finding wcs...")
    wcs_out, shape_out = find_optimal_celestial_wcs(f[1:10]) # has only CompImageHDU files

    print("creating mosaic...")
    array, footprint = reproject_and_coadd(f[1:10], wcs_out, shape_out=shape_out, reproject_function=reproject_interp)        

    for index, row in df.iterrows():
        ra = row["ra"]
        dec = row["dec"]

        output_dir = dir_path
        output_file = os.path.join(dir_path, str(row["id_1"])) + ".fits"

        try:
            position=wcs_out.wcs_world2pix(ra,dec,0) # gets position in pixels
    
            cutout = Cutout2D(array, position, size, wcs=wcs_out, copy=True) # employs cutout retaining wcs info
            header_new = cutout.wcs.to_header() # gives header info to cutout

            # hdu config:
            hdu_p = fits.PrimaryHDU(header=orig_header)
            hdu_i = fits.ImageHDU(cutout.data, header=header_new)
            hdulist = fits.HDUList([hdu_p,hdu_i])

            hdulist.writeto(output_file, overwrite=True)

        except astropy.nddata.utils.NoOverlapError as e:
            continue
