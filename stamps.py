import numpy as np
import matplotlib.pyplot as plt
from deepscan.deepscan import DeepScan
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.io import fits
import os
from astropy import wcs
from astropy.coordinates import Angle
import astropy.units as u


##############################################

class EllipseBBox():
    """
    Outputs a stamp for each source detected in a fits image
    -------
    Input:
    
    filename = str / path to .fits image
    ps = float / pixel scale [arcsec per pixel] 
    mzero = float / magnitude zero point
    sizethresh = float / threshold for minimum stamp side size [pixels]
    SBthresh = tuple / minimum and maximum threshold for surface brightness of the detected sources
    """

    def __init__(self, filename, ps, mzero, sizethresh, SBthresh):
        self.filename = filename
        self.f = fits.open(filename, memmap=True)
        img = self.f[1].data
        self.data = img.byteswap().newbyteorder()
        self.ps = ps
        self.mzero = mzero
        self.sizethresh = sizethresh
        self.SBthresh = SBthresh


    def _SB(self, flux, area):
        """
        Computes the Surface Brightness
        """
        area_arcsec = area*self.ps*self.ps
        return -2.5*np.log10(flux) + 2.5*np.log10(area_arcsec) + self.mzero


    def _get_ellipse_bb(self, x, y, major, minor, angle_deg):
        """
        Get the bounding box coordinates. From https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0
        """
        t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
        [max_x, min_x] = [x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) - minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t, t + np.pi)]
    
        return min_x, max_x


    def get_stamps(self):
        """
        Get the sources in the image as stamps
        """
        # running DeepScan
        result = DeepScan(self.data, verbose=False)
        df = result["df"]
        df.dropna(inplace=True)

        # excluding data bellow a certain surface brightness threshold
        df["SB"] = self._SB(df["flux"].values, df["area"].values)

        SB_min, SB_max = self.SBthresh
        df.drop(df[df["SB"] <= SB_min].index, inplace=True)
        df.drop(df[df["SB"] >= SB_max].index, inplace=True)
        df.reset_index(inplace=True)

        # apply _get_ellipse_bb to find values bounding box extreme points
        df["min_x"], df["max_x"] = self._get_ellipse_bb(df["xcen"].values, df["ycen"].values, df["a_rms"].values, df["b_rms"].values, df["theta"].values)

        # crop "data" according to the positions in df, outputs individual cutouts
        df["size"] = (df["max_x"].values - df["min_x"].values) * 10
        #print(df)
        df.drop(df[df["size"] < self.sizethresh].index, inplace=True)
        df.reset_index(inplace=True)
        
        # WCS for the detected sources
        w = wcs.WCS(self.f[1].header)
        df["ra"], df["dec"] = w.wcs_pix2world(df["xcen"], df["ycen"], 0) 

        # creating the cutout and preserving the header
        stamps = []
        headers = []
        
        for index, row in df.iterrows():
            stamp = Cutout2D(self.data, position=(row["xcen"], row["ycen"]), size=(row["size"],row["size"]), wcs=w, copy=True)
            stamps.append(stamp)
            header_new = stamp.wcs.to_header()
            headers.append(header_new)

            df.at[index, "name"] = str(index) # name of the output .fits image
            df.at[index, "association"] = str(self.filename.rsplit(".", 1)[0].split("/")[2]) # name of the big cutout image
        
        assert len(stamps) == len(headers)

        return stamps, headers, df 


    def show_stamps(self, title=""):
        """
        Detects stamps and show where they are on the original image
        """
        stamps, headers, df = self.get_stamps()
        
        avg = np.mean(np.arcsinh(self.data))
        plt.imshow(np.arcsinh(self.data), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")
        plt.title(title)
        plt.colorbar()

        for el in stamps:
            el.plot_on_original(color='red')
        print(df)
        plt.show()


    def save_stamps(self, dir_name="stamps"):
        """
        Save stamps to a newly created directory
        """        
        pwd = os.getcwd()
        dir_stamps = os.path.join(pwd, dir_name)
        os.mkdir(dir_stamps)

        orig_header = self.f[0].header

        stamps, headers, df = self.get_stamps()

        for index in range(len(stamps)):
            hdu_p = fits.PrimaryHDU(header=orig_header)
            hdu_i = fits.ImageHDU(stamps[index].data, header=headers[index])
            hdul = fits.HDUList([hdu_p,hdu_i])

            stamp_filename = os.path.join(dir_stamps, str(index) + ".fits")
            hdul.writeto(stamp_filename)
        
        df.to_csv(os.path.join(dir_stamps, "catalog.csv"), index=False)
        return df


##############################################


def get_candidate(input_file, output_file, ra, dec, size):
    """
    Extract a stamp from "input_file" according to its WCS positions. From: https://github.com/rodff/LSB_galaxies/blob/master/cutout_decam_image.ipynb
    
    -------
    Input:
    input_file = string / name of .fits file (original image)
    output_file = string / name of output .fits file (extracted candidate)
    ra = astropy.coordinates.Angle / right ascension of the candidate
    dec = astropy.coordinates.Angle / declination of the candidate
    size = int / lenght of the stamp 
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