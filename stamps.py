import numpy as np
import matplotlib.pyplot as plt
from deepscan import skymap, dbscan, deblend, makecat
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.io import fits
import os
from astropy import wcs
from astropy.coordinates import Angle
import astropy.units as u
from astropy.convolution import Ring2DKernel
from astropy.convolution import convolve


##############################################

class AnalyzeImage(object):
    """
    Outputs a stamp for each source detected in a fits image
    -------
    Input:
    
    filename = str / path to .fits image
    mzero = float / magnitude zero point
    sizethresh = float / threshold for minimum stamp side size [pixels]
    sbthresh = float / minimum threshold for surface brightness of the detected sources
    elthresh = float / maximum ellipticity
    angthresh = float / minimum angular size in arcseconds
    """

    def __init__(self, filename):
        self.filename = filename
        self.f = fits.open(filename, memmap=True)
        img = self.f[1].data
        self.data = img.byteswap().newbyteorder()
        self.a = img.byteswap().newbyteorder()

    def thresholds(self, sizethresh, sbthresh, elthresh, i50thresh, i50avthresh, r50thresh, angthresh):
        self.sizethresh = sizethresh
        self.sbthresh = sbthresh
        self.elthresh = elthresh
        self.i50thresh = i50thresh
        self.i50avthresh = i50avthresh
        self.r50thresh = r50thresh
        self.angthresh = angthresh

    def subtract_sky(self):
        sky, self.rms = skymap.skymap(self.data, verbose=False)
        self.data -= sky

    def apply_ring_filter(self, inner_radius=None, width=5):
        sky, self.rms = skymap.skymap(self.data, verbose=False)
        self.data -= sky

        if inner_radius == None:
            inner_radius = self.f[0].header["FWHM"]/2

        print("applying ring filter...")
        kernel = Ring2DKernel(inner_radius, width)
        self.data = convolve(self.data, kernel)

    def show_stamps(self, title=""):
        """
        Detects stamps and show where they are on the original image
        """
        stamps, headers, df = self._get_stamps()
        
        avg = np.mean(np.arcsinh(self.a))
        plt.imshow(np.arcsinh(self.a), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")
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

        stamps, headers, df = self._get_stamps()

        for index in range(len(stamps)):
            hdu_p = fits.PrimaryHDU(header=orig_header)
            hdu_i = fits.ImageHDU(stamps[index].data, header=headers[index])
            hdul = fits.HDUList([hdu_p,hdu_i])

            stamp_filename = os.path.join(dir_stamps, str(index) + ".fits")
            hdul.writeto(stamp_filename)
        
        df.to_csv(os.path.join(dir_stamps, "catalog.csv"), index=False)
        return df


    def _SB(self, flux, area, mzero, ps):
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

    def _get_stamps(self):
        """
        Get the sources in the image as stamps
        """
        # running DeepScan
        C = dbscan.DBSCAN(self.data, self.rms, verbose=False)
        segmap, segments = deblend.deblend(self.data, bmap=C.segmap, rms=self.rms, verbose=False)
        df = makecat.MakeCat(self.data, segmap=segmap, segments=segments, verbose=False)

        #result = DeepScan(self.data, verbose=False)
        #df = result["df"]
        df.dropna(inplace=True)

        # remove data below ellipticity treshold
        df.drop(df[(1 - df["q"]) > self.elthresh].index, inplace=True)

        # determine pixel scale
        cam_deg = self.f[0].header["CDELT2"]
        cam_arc = (cam_deg*u.deg).to(u.arcsec)
        ps = cam_arc.value 
        pixelscale = u.pixel_scale(cam_deg*u.arcsec/u.pixel)

        # remove data below angular size threshold
        a = (df["R50"].values * u.pixel).to(u.arcsec, pixelscale)
        df['angular_size'] = a.value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        df.drop(df[df["angular_size"] < self.angthresh].index, inplace=True)

        # excluding data bellow a certain surface brightness threshold
        mzero = self.f[0].header["MAGZERO"]
        df["SB"] = self._SB(df["flux"].values, df["area"].values, mzero, ps)
        df.drop(df[df["SB"] <= self.sbthresh].index, inplace=True)

        # remove data below other thresholds
        df.drop(df[df["I50"] < self.i50thresh].index, inplace=True)
        df.drop(df[df["I50av"] < self.i50avthresh].index, inplace=True)
        df.drop(df[df["R50"] < self.r50thresh].index, inplace=True)

        # apply _get_ellipse_bb to find values bounding box extreme points
        df["min_x"], df["max_x"] = self._get_ellipse_bb(df["xcen"].values, df["ycen"].values, df["a_rms"].values, df["b_rms"].values, df["theta"].values)

        # crop "data" according to the positions in df, outputs individual cutouts
        df["size"] = (df["max_x"].values - df["min_x"].values) * 5
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
            df.at[index, "association"] = str(self.filename.rsplit(".", 1)[0].rsplit("/")[-1]) # name of the big cutout image
        
        assert len(stamps) == len(headers)

        # arrange final catalog
        df.drop(['index', 'segID', 'parentID'], axis=1, inplace=True)
        names = ["dec", "ra"]
        for n in names:
            col = n
            first_col = df.pop(col)
            df.insert(0, col, first_col)

        return stamps, headers, df 
