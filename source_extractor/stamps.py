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

class AnalyzeImage(object):
    """
    Outputs a stamp for each source detected in a .fits image.

    Parameters
    ----------   
    filename: str
        Path to .fits image file.

    stamp_size: int
        Side dimension of the stamp containing the source in the center. For example: 150 outputs a 150x150 stamp.

    img_hdu: int, default 1
        HDU where the image data is stored.
    """
    def __init__(self, filename, stamp_size, img_hdu=1):
        self.filename = filename
        self.f = fits.open(filename, memmap=True)
        img = self.f[img_hdu].data
        self.data = img.byteswap().newbyteorder()
        self.a = img.byteswap().newbyteorder()

        self.stamp_size = stamp_size        

    def thresholds(self, sbthresh=25, elthresh=0.7, i50thresh=6, i50avthresh=9, r50thresh=5):
    """
    Define the thresholds used for source identification. The default values are the ones that we find work best for detecting LSBGs in DECam images.

    Parameters
    ----------   
    sbthresh: float, default 25
        Minimum surface brightness. Will only detect sources with surface brightness >= sbthresh.

    elthresh: float, default 0.7
        Maximum ellipticity. Will only detect sources with ellipticity <= elthresh.

    i50thresh: float, default 6
        Minimum I50. Will only detect sources with I50 >= i50thresh.

    i50avthresh: float, default 9
        Minimum I50AV. Will only detect sources with I50AV >= i50avthresh.

    r50thresh: float, default 5
        Minimum R50. Will only detect sources with R50 >= r50thresh.
    """
        self.sbthresh = sbthresh
        self.elthresh = elthresh
        self.i50thresh = i50thresh
        self.i50avthresh = i50avthresh
        self.r50thresh = r50thresh

    def show_stamps(self, title="", scale=None):
        """
        Show detected sources on the original image.

        Parameters
        ----------   
        title: str
            Title displayed on the cutout images that are shown.

        scale: tuple, deafault None
            Scale for the image in a format of (vmin,vmax). If set to None, uses a default scale based on the average value of the pixels in the image.
        """
        stamps, headers, df = self._get_stamps()
        
        if type(scale) == tuple:
            vmin = scale[0]; vmax = scale[1]
            plt.imshow(np.arcsinh(self.a), origin='lower', vmin=vmin, vmax=vmax, cmap="binary_r")

        else:
            avg = np.mean(np.arcsinh(self.a))
            plt.imshow(np.arcsinh(self.a), origin='lower', vmin=avg*0.999, vmax=avg*1.005, cmap="binary_r")

        plt.title(title)
        plt.colorbar()

        for el in stamps:
            el.plot_on_original(color='red')
        plt.show()


    def save_stamps(self, dir_name):
        """
        Save stamps to a newly created directory and returns their catalog in a DataFrame object.

        Parameters
        ----------   
        dir_name: str
            Path to a new directory where the stamps and their catalog are going to be stored.

        Returns
        -------
        pandas.DataFrame
            DataFrame catalog containing information about the sources.            
        """
        orig_header = self.f[0].header
        stamps, headers, df = self._get_stamps()

        os.mkdir(dir_name)

        for index in range(len(stamps)):
            hdu_p = fits.PrimaryHDU(header=orig_header)
            hdu_i = fits.ImageHDU(stamps[index].data, header=headers[index])
            hdul = fits.HDUList([hdu_p,hdu_i])

            stamp_filename = os.path.join(dir_name, str(index) + ".fits")
            hdul.writeto(stamp_filename)
        
        df.to_csv(os.path.join(dir_name, "catalog.csv"), index=False)
        return df


    def _SB(self, flux, area, mzero, ps):
        """
        Computes surface brightness.
        """
        area_arcsec = area*ps*ps
        return -2.5*np.log10(flux) + 2.5*np.log10(area_arcsec) + mzero

    def _get_stamps(self):
        """
        Get the sources in the image as stamps.
        """
        # run DeepScan
        sky, rms = skymap.skymap(self.data, verbose=False)
        self.data -= sky

        C = dbscan.DBSCAN(self.data, rms, verbose=False)
        segmap, segments = deblend.deblend(self.data, bmap=C.segmap, rms=rms, verbose=False)
        df = makecat.MakeCat(self.data, segmap=segmap, segments=segments, verbose=False)
        df.dropna(inplace=True)

        # remove data above ellipticity treshold
        df.drop(df[(1 - df["q"]) > self.elthresh].index, inplace=True)

        # determine pixel scale
        cam_deg = self.f[0].header["CDELT2"]
        cam_arc = (cam_deg*u.deg).to(u.arcsec)
        ps = cam_arc.value 
        pixelscale = u.pixel_scale(cam_deg*u.arcsec/u.pixel)

        # remove data below surface brightness threshold
        mzero = self.f[0].header["MAGZERO"]
        df["SB"] = self._SB(df["flux"].values, df["area"].values, mzero, ps)
        df.drop(df[df["SB"] < self.sbthresh].index, inplace=True)

        # remove data below other thresholds
        df.drop(df[df["I50"] < self.i50thresh].index, inplace=True)
        df.drop(df[df["I50av"] < self.i50avthresh].index, inplace=True)
        df.drop(df[df["R50"] < self.r50thresh].index, inplace=True)

        df.reset_index(inplace=True)
        
        # compute detected source's WCS
        w = wcs.WCS(self.f[1].header)
        df["ra"], df["dec"] = w.wcs_pix2world(df["xcen"], df["ycen"], 0) 

        # crop
        stamps = []
        headers = []
        
        for index, row in df.iterrows():
            stamp = Cutout2D(self.data, position=(row["xcen"], row["ycen"]), size=(self.stamp_size, self.stamp_size), wcs=w, copy=True)
            stamps.append(stamp)
            header_new = stamp.wcs.to_header()
            headers.append(header_new)

            df.at[index, "name"] = str(index) # name of the output .fits image
            df.at[index, "association"] = str(self.filename.rsplit(".", 1)[0].rsplit("/")[-1]) # name of the big cutout image
            df.at[index, "mosaic"] = str(self.filename.rsplit(".", 1)[0].rsplit("/")[-2]) # name of the mosaic

        assert len(stamps) == len(headers)

        # arrange final catalog
        df.drop(['index', 'segID', 'parentID'], axis=1, inplace=True)
        names = ["dec", "ra"]
        for n in names:
            col = n
            first_col = df.pop(col)
            df.insert(0, col, first_col)

        return stamps, headers, df 
