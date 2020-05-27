import numpy as np
import matplotlib.pyplot as plt
from deepscan.deepscan import DeepScan
from deepscan import remote_data
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch)
import os

##############################################

class EllipseBBox():
    """
    Outputs a stamp for each source detected in an image
    """

    def __init__(self, data, ps, mzero, SBthresh=None):
        """
        Applies DeepScan algorithm to "data" and retrieves the values for each source detected
        --------------
        Input:
        data = 2D float array / image 

        ps = float / pixel scale [arcsec per pixel] 

        mzero = float / magnitude zero point

        SBthresh = float / threshold for the minimum surface brightness of the sources to be detected
        """ 

        # running DeepScan
        result = DeepScan(data)
        df = result["df"]

        # excluding data bellow a certain surface brightness threshold
        df["SB"] = self._SB(df["flux"].values, df["area"].values, ps, mzero)
        if SBthresh == None:
            SBthresh = df["SB"].mean() # use the mean of the SB as the threshold
        df = df[df["SB"] <= SBthresh]
        df.reset_index(inplace=True)

        # variables to be used for finding the size of each source's stamp
        x = df["xcen"].values
        y = df["ycen"].values

        # apply _get_ellipse_bb, outputs new df with the padding already considered
        min_x, max_x = self._get_ellipse_bb(x, y, df["a_rms"].values, df["b_rms"].values, df["theta"].values)
        box_df = pd.DataFrame({"min_x": min_x, "max_x": max_x, "xcen": x, "ycen": y}).dropna()

        # apply _crop, outputs images (first print, if its working output them to a directory)
        stamp = self._crop(data, box_df)
        #plt.imshow(np.arcsinh(data), cmap='binary', vmin=-0.5, vmax=3, origin="lower")
        
        # astropy plot test
        imshow_norm(np.arcsinh(data), origin='lower', interval=MinMaxInterval(), stretch=SqrtStretch(), cmap="binary_r")

        for el in stamp:
            el.plot_on_original(color='red')
        plt.show()


    def _SB(self, flux, area, ps, mzero):
        """
        Computes the Surface Brightness
        """
        area_arcsec = area*ps*ps
        return -2.5*np.log10(flux) + 2.5*np.log10(area_arcsec) + mzero


    def _get_ellipse_bb(self, x, y, major, minor, angle_deg):
        """
        Get the bounding box coordinates. From https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0
        """
        t = np.arctan(-minor / 2 * np.tan(np.radians(angle_deg)) / (major / 2))
        [max_x, min_x] = [x + major / 2 * np.cos(t) * np.cos(np.radians(angle_deg)) - minor / 2 * np.sin(t) * np.sin(np.radians(angle_deg)) for t in (t, t + np.pi)]
    
        return min_x, max_x


    def _crop(self, data, box_df): 
        """
        Crop "data" according to the positions in box_df, outputs individual cutouts
        """
        cutout = []
        for index, row in box_df.iterrows():
            del index
            size = np.linalg.norm(row["max_x"] - row["min_x"]) * 4
            stamp = Cutout2D(data, position=(row["xcen"], row["ycen"]), size=(size,size), copy=True) 
            cutout.append(stamp)

        return cutout


    def _export_imgs(self, stamp):
        pass

##############################################

def extract_stamps(ps, mzero):
    """
    Take .fits files in the folder and extract its stamps
    """

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
        EllipseBBox(data, ps, mzero)


# values for DECAM (r-band)
ps = 0.27
mzero = 31.395

extract_stamps(ps, mzero)
