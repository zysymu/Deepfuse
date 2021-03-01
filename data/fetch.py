import pandas as pd
import urllib.request
import urllib.error
import os

def fetch(csv, output_dir, id_df_path, ps=0.27): # same ps as DECam
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
                try:
                  urllib.request.urlretrieve(legacy_survey, os.path.join(output_dir, candidate))
                except urllib.error.HTTPError as e:
                  print(e)
                  continue

                id_df.loc[i, 'name'] = candidate
                id_df.loc[i, 'label'] = label

                id_df['label'] = id_df['label'].astype(int)
                id_df.to_csv(id_df_path, index=False)
                print('csv saved')
                
        # download new candidate
        else:           
            print(f'downloading ra={ra}, dec={dec}')
            legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr8&pixscale={ps}'
            print(legacy_survey)
            try:
              urllib.request.urlretrieve(legacy_survey, os.path.join(output_dir, candidate))
            except urllib.error.HTTPError as e:
              print(e)
              continue

            id_df.loc[i, 'name'] = candidate
            id_df.loc[i, 'label'] = label

            id_df['label'] = id_df['label'].astype(int)
            id_df.to_csv(id_df_path, index=False)
            print('csv saved')
   
# an example of how to use it:     
"""
if __name__ ==  '__main__':
    train = '/content/drive/My Drive/Deepfuse/data/train.csv'
    train_dir = '/content/drive/My Drive/Deepfuse/data/train-ps'

    fetch(train, train_dir, '/content/drive/My Drive/Deepfuse/data/id-train-ps.csv', ps=0.27)

    val = '/content/drive/My Drive/Deepfuse/data/val.csv'
    val_dir = '/content/drive/My Drive/Deepfuse/data/val-ps'

    fetch(val, val_dir, '/content/drive/My Drive/Deepfuse/data/id-val-ps.csv', ps=0.27)

    test = '/content/drive/My Drive/Deepfuse/data/test.csv'
    test_dir = '/content/drive/My Drive/Deepfuse/data/test-ps'

    fetch(test, test_dir, '/content/drive/My Drive/Deepfuse/data/id-test-ps.csv', ps=0.27)
"""
