import numpy as np
import matplotlib.pyplot as plt
from deepscan.deepscan import DeepScan
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.visualization import (imshow_norm, MinMaxInterval, SqrtStretch, simple_norm)
import os

##############################################

class EllipseBBox():
    """
    Outputs a stamp for each source detected in an image
    """

    def __init__(self, data, ps, mzero, sizethresh, SBthresh=None):
        """
        Applies DeepScan algorithm to "data" and retrieves the values for each source detected
        --------------
        Input:
        data = 2D float array / image 

        ps = float / pixel scale [arcsec per pixel] 

        mzero = float / magnitude zero point

        sizethresh = float / threshold for minimum stamp side size [pixels]

        SBthresh = float / threshold for the maximum surface brightness of the sources to be detected (if = None we use the mean+1)
        """ 
        self.data = data
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
        result = DeepScan(self.data)
        df = result["df"]
        df.dropna(inplace=True)

        # excluding data bellow a certain surface brightness threshold
        df["SB"] = self._SB(df["flux"].values, df["area"].values)
        if self.SBthresh == None:
            SBthresh = df["SB"].mean() - 2
        else:
            SBthresh = self.SBthresh

        #df = df[df["SB"] >= SBthresh]
        df.drop(df[df["SB"] < SBthresh].index, inplace=True)

        # apply _get_ellipse_bb to find values bounding box extreme points
        df["min_x"], df["max_x"] = self._get_ellipse_bb(df["xcen"].values, df["ycen"].values, df["a_rms"].values, df["b_rms"].values, df["theta"].values)

        # crop "data" according to the positions in df, outputs individual cutouts
        df["size"] = (df["max_x"].values - df["min_x"].values) * 10
        #print(df)
        df.drop(df[df["size"] < self.sizethresh].index, inplace=True)
        df.reset_index(inplace=True)
        
        cutout = []        
        for index, row in df.iterrows():
            stamp = Cutout2D(self.data, position=(row["xcen"], row["ycen"]), size=(row["size"],row["size"]), copy=True) 
            cutout.append(stamp)
        
        return cutout, df


    def show_stamps(self, title=""):
        """
        Detects stamps and show where they are on the original image
        """
        stamps, df = self.get_stamps()
        
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

        stamps, df = self.get_stamps()
        for index, cutout in enumerate(stamps):
            hdu = fits.PrimaryHDU(cutout.data)
            hdul = fits.HDUList([hdu])
            cutout_filename = os.path.join(dir_stamps, str(index) + ".fits")
            #hdu.writeto(cutout_filename, overwrite=True)
            hdul.writeto(cutout_filename)
        
        df.to_csv(os.path.join(dir_stamps, "catalog.csv"), index=False)
        return df


##############################################