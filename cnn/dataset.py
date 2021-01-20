import torch
from torch.utils.data import Dataset
import pandas as pd
from astropy.io import fits
import numpy as np
from torchvision import transforms


def rescale(data, newmin, newmax): # https://stackoverflow.com/questions/33610825/normalization-in-image-processing
    shape = len(data.shape)
    if shape == 3:
        for i in range(shape):
            data[i,:,:] = (data[i,:,:] - torch.min(data[i,:,:])) * (newmax - newmin)/(torch.max(data[i,:,:]) - torch.min(data[i,:,:])) + newmin

    else:
        data = (data - torch.min(data)) * (newmax - newmin)/(torch.max(data) - torch.min(data)) + newmin

    return data


class Rescale(object):
    def __init__(self, newmin, newmax):
        self.newmin = newmin
        self.newmax = newmax

    def __call__(self, tensor):
        image = rescale(tensor, self.newmin, self.newmax)
        
        return tensor


class LegacySurveyDataset(Dataset):
    def __init__(self, csv_file, transform=None, ps=0.14):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.ps = ps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ra = self.df.loc[idx, 'ra']
        dec = self.df.loc[idx, 'dec']
        label = self.df.loc[idx, 'label']
        
        legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr8&pixscale={self.ps}'

        hdul = fits.open(legacy_survey)
        img = hdul[0].data # channels first
        img = img.astype(np.float32)
        img = torch.tensor(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)


def transform(training):
    if training:
        return transforms.Compose([Rescale(0,1), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(5), transforms.RandomCrop(200), transforms.Resize(128)])
    else: # evaluation
        return transforms.Compose([Rescale(0,1), transforms.Resize(128)])