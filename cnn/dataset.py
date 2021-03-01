import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from astropy.io import fits
import numpy as np
import os
from helper_dataset import transform_legacy, transform_deepscan

class LegacySurveyDataset(Dataset):
    def __init__(self, csv_file, dir_path, transform, return_filename=False):
        self.df = pd.read_csv(csv_file)
        self.dir_path = dir_path
        self.transform = transform
        self.return_filename = return_filename

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.loc[idx, 'name']

        img_path = os.path.join(self.dir_path, name)
        
        try:
            hdul = fits.open(img_path)
            img = hdul[0].data # channels first
            hdul.close()
        except OSError as e:
            print(e, name)
        
        img = img.astype(np.float32)        
        img = torch.tensor(img)

        if self.transform == 'train':
            t = transform_legacy(True)
            img = t(img)
        
        elif self.transform == 'eval':
            t = transform_legacy(False)
            img = t(img)

        if self.return_filename:
            return img, name
        
        else:
            label = self.df.loc[idx, 'label']
            return img, torch.tensor(label)


class DeepscanDataset(Dataset):
    def __init__(self, csv_file, dir_path, transform, return_filename=False):
        self.df = pd.read_csv(csv_file)
        self.dir_path = dir_path
        self.transform = transform
        self.return_filename = return_filename

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.loc[idx, 'name']

        img_path = os.path.join(self.dir_path, name)
        
        hdul = fits.open(img_path)
        img = hdul[1].data # no channels       

        hdul.close()
        
        img = img.astype(np.float32)        
        img = torch.tensor(img)

        img = torch.unsqueeze(img, 0) # create new dimension, channels first
        img = img.expand(3, img.shape[1], img.shape[2])
        
        for i in range(1,3):
            img[i,:,:] = img[0,:,:] 

        if self.transform == 'train':
            print('DeepscanDataset only accepts transforms in `eval` mode')

        elif self.transform == 'eval':
            t = transform_deepscan(False)
            img = t(img)

        if self.return_filename:
            return img, name
        
        else:
            label = self.df.loc[idx, 'label']
            return img, torch.tensor(label)
