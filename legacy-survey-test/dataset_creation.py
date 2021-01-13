import numpy as np 
import pandas as pd
from urllib.request import urlretrieve
import matplotlib
import requests
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

d = '/home/marcostidball/Deepfuse/legacy-survey-test'

candidates = pd.read_csv(os.path.join(d,'coordenadas_candidatas.csv'))

# params:
ps = 0.2 # zoom level

def extract(df, ps, fits):
    for i, row in df.iterrows():
        ra = row['ra']
        dec = row['dec']
        
        if fits:
            legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr8&pixscale={ps}'
            urlretrieve(legacy_survey, f'{i}.fits')
            
        else: # jpg
            legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer=ls-dr8&pixscale={ps}'
            urlretrieve(legacy_survey, f'{i}.jpg')
        

if __name__ ==  '__main__':
    extract(candidates, ps, fits=True)
    