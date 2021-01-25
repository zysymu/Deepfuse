import pandas as pd
from urllib.request import urlretrieve
import os

d = '/home/marcostidball/Deepfuse/legacy-survey-test'

candidates = pd.read_csv(os.path.join(d,'train.csv'))

# params:
ps = 0.2 # zoom level

def fetch(df, ps, fits):
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
    fetch(candidates, ps, fits=True)
    
