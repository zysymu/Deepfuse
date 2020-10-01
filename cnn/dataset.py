from astropy.io import fits
import numpy as np
from skimage.transform import resize
import os

class Dataset():
    """
    Generate a dataset to be used by the classifier.
    """
    def __init__(self, img_dim, vscale):
        """
        Define the dataset parameters.

        Parameters
        ----------
        img_dim: float
            Side lenght for images after being resized. For example: img_dim = 100 resizes the image to 100x100.

        vscale: float tuple, default tbd
            Tuple with format (vmin, vmax) for adjusting the image scale. If set to None, no normalization occurs.
        """    
        self.img_dims = (img_dim, img_dim)
        self.vscale = vscale

        if type(vscale) == tuple:
            self.vmin = vscale[0]; self.vmax=vscale[1]
        elif vscale == None: 
            self.vscale = False

    def training(self, data_dir_path, dir_names=("not", "galaxies")):
        """
        Load images and labels as arrays that can then be processed for training by the classifier.

        Parameters
        ----------
        data_dir_path: str
            Path to directory that contains 2 subdirectories, one containing negative images and the other containing positive images.

        dir_names: str tuple, default ("not", "galaxies")
            Name of the directories that contain the two classes. 

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing the dataset as (data, labels).
        """
        negative, positive = dir_names[0], dir_names[1]

        data = [] 
        labels = []
        g = 0
        n = 0

        for d in os.listdir(data_dir_path):
            p = os.path.join(data_dir_path, d)

            for f in os.listdir(p):
                pp = os.path.join(p, f)
                hdul = fits.open(pp, memmap=False)
                img = hdul[1].data
                hdul.close()

                if not (np.isnan(img).any() or np.isinf(img).any()):
                    img = resize(img, self.img_dims)
                    img = img.reshape((img.shape[0], img.shape[1], 1))
                    img = np.pad(img, [(0, 0), (0, 0), (0, 2)], 'constant')

                    data.append(img)

                    if d == positive:
                        label = 1
                        g += 1
                        labels.append(label)

                    elif d == negative:
                        label = 0
                        n += 1
                        labels.append(label)

        # change scale
        if self.vscale:
            scale = 1
            data = np.arcsinh(np.clip(data, vmin, vmax)/vmax * scale)

        data = np.asarray(data)    
        labels = np.asarray(labels)

        print("total galaxies: ", g)
        print("total not: ", n)

        return data, labels

    def classifying(self, data_dir_path):
        """
        Load images as an array that can then be classified by the classifier.

        Parameters
        ----------
        data_dir_path: str
            Path to directory that contains images.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing the data to be analysed and the name of the files.
        """
        data = [] 
        f_names = []

        for f in os.listdir(data_dir_path):
            pp = os.path.join(data_dir_path, f)
            hdul = fits.open(pp, memmap=False)
            img = hdul[1].data
            hdul.close()

            if not (np.isnan(img).any() or np.isinf(img).any()):
                img = resize(img, self.img_dims)
                img = img.reshape((img.shape[0], img.shape[1], 1))
                img = np.pad(img, [(0, 0), (0, 0), (0, 2)], 'constant')

                data.append(img)
                f_names.append(f)

        # change scale
        if self.vscale:
            scale = 1
            data = np.arcsinh(np.clip(data, vmin, vmax)/vmax * scale)

        data = np.asarray(data)    

        return data, f_names
