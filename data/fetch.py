import pandas as pd
from urllib.request import urlretrieve
import os

"""
import time
import urllib
import math
import sys

def truncate(f, n=4): # saves from url problems
    if f < 0:
        return math.ceil(f * 10 ** n) / 10 ** n
    else:
        return math.floor(f * 10 ** n) / 10 ** n

        ra = truncate(self.df.loc[idx, 'ra'])
        dec = truncate(self.df.loc[idx, 'dec'])
        label = self.df.loc[idx, 'label']

        legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr9&pixscale={self.ps}'

        # dealing with http errors    
        print(legacy_survey)

        retries = 1
        success = False
        while not success:
            try:
                hdul = fits.open(legacy_survey)
                success = True
            #except urllib.error.HTTPError as e:
            except (Exception, urllib.error.ContentTooShortError) as e:
                wait = retries * 30;
                print(f'Error! Waiting {wait} secs and re-trying...')
                sys.stdout.flush()
                time.sleep(wait)
                retries += 1
"""


def fetch(csv, output_dir, id_df_path, ps=0.13):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv)

    if os.path.exists(id_df_path): 
        id_df = pd.read_csv(id_df_path)

    else:
        id_df = pd.DataFrame({'name':[], 'label':[]})

    for i, row in df.iterrows():
        candidate = f'{i}.fits'

        ra = row['ra']
        dec = row['dec']            
        label = row['label']

        # check if candidate is in the folder
        if os.path.exists(os.path.join(output_dir, candidate)): 

            # if candidate already exists, skip iteration
            try:
                if candidate in id_df.loc[i, 'name']:
                    continue

            # if candidate is in the folder but not in id_df
            # it means it was not downloaded properly, so
            # we redownload it
            except Exception as e:
                os.remove(os.path.join(output_dir, candidate))
                print(f'redownloading ra={ra}, dec={dec}')
                legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr8&pixscale={ps}'
                urlretrieve(legacy_survey, os.path.join(output_dir, candidate))

                id_df.loc[i, 'name'] = candidate
                id_df.loc[i, 'label'] = label

                id_df['label'] = id_df['label'].astype(int)
                id_df.to_csv(id_df_path, index=False)
                print('csv saved')
                
        # download new candidate
        else:           
            print(f'downloading ra={ra}, dec={dec}')
            legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr8&pixscale={ps}'
            urlretrieve(legacy_survey, os.path.join(output_dir, candidate))

            id_df.loc[i, 'name'] = candidate
            id_df.loc[i, 'label'] = label

            id_df['label'] = id_df['label'].astype(int)
            id_df.to_csv(id_df_path, index=False)
            print('csv saved')
        

if __name__ ==  '__main__':
    train = '/content/drive/My Drive/Deepfuse/data/train.csv'
    train_dir = '/content/drive/My Drive/Deepfuse/data/train'

    fetch(train, train_dir, '/content/drive/My Drive/Deepfuse/data/id-train.csv')
