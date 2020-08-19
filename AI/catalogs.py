import os
from shutil import copy, move
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

def delete_replicas(dir_path):
    """
    Open each directory inside dir_path. In each of these directories, get the catalog 
    related to the sources detected using DeepScan and delete the sources that have
    a maximum separation of sep arcseconds (i.e. delete replicas).
    """
    catalogs = []

    for f in os.listdir(dir_path):
        full_path = os.path.join(dir_path, f)
        c = pd.read_csv(os.path.join(full_path, "catalog.csv"))
        c["mosaic"] = f
        catalogs.append(c)

    big_one = pd.concat(catalogs)
    big_one.sort_values('size', ascending=False).drop_duplicates(subset=['ra', 'dec'], inplace=True)
    big_one.reset_index(inplace=True)
    big_one.to_csv(os.path.join(dir_path, "catalog.csv"), index=False)

    return big_one

CATALOGS_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/fo-real3"
#catalog = delete_replicas(CATALOGS_PATH)

################################################

def true_catalog(catalog_path, mock_dir_path, train_dir_path, mu_0_thresh, mean_mu_e_thresh):
    df = pd.read_csv(catalog_path)

    mu_0_threshl, mu_0_threshu = mu_0_thresh
    mean_mu_e_threshl, mean_mu_e_threshu = mean_mu_e_thresh

    df.drop(df[df["mu_0"] < mu_0_threshl].index, inplace=True) # cuts everything below mu_0_threshl
    df.drop(df[df["mu_0"] > mu_0_threshu].index, inplace=True) # cuts everything above mu_0_threshu

    df.drop(df[df["mean_mu_e"] < mean_mu_e_threshl].index, inplace=True)
    df.drop(df[df["mean_mu_e"] > mean_mu_e_threshu].index, inplace=True)

    galaxy_dir = os.path.join(train_dir_path, "galaxies")
    os.mkdir(galaxy_dir)

    for d in os.listdir(mock_dir_path):
        inside = os.path.join(mock_dir_path, d)
        num = d.split("_")[-1]
        f = f"mock_{num}_g.fits"
        copy(os.path.join(inside, f), galaxy_dir)


    
def false_catalog(catalog_path, cutout_dir_path, train_dir_path, sample):
    df = pd.read_csv(catalog_path)

    #
    # here we apply another function to delete replicas (for now doing it manually is okay)
    #

    df = df.sample(n=sample).reset_index(drop=True)
    print(df)

    stamp_dir = os.path.join(train_dir_path, "not")
    os.mkdir(stamp_dir)

    for index, row in df.iterrows():
        d = os.path.join(cutout_dir_path, row["mosaic"])
        d2 = os.path.join(d, os.path.join("stamps",row["association"]))
        nam = str(row["name"]) + ".fits"
        f = os.path.join(d2, nam)

        if nam in os.listdir(stamp_dir):
            i = row["index"]
            nam_sub = f"{i}.fits"
            old = os.path.join(stamp_dir, nam)
            new = os.path.join(stamp_dir, nam_sub)
            os.rename(old, new)
            copy(f, stamp_dir)
        else:
            copy(f, stamp_dir)

        
MOCK_CATALOG = "/home/marcostidball/ic-astro/PROJECT/AI/mocks_marcos/point_050619_g.csv"
MOCK_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/mocks_marcos/point_050619_g"
TRAIN_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/train"
sbthresh = 25

STAMPS_CATALOG = "/home/marcostidball/ic-astro/PROJECT/AI/catalog.csv"
CUTOUT_DIR_PATH = "/home/marcostidball/ic-astro/PROJECT/AI/fo-real3"
sample = 500

mu_0_thresh = (22.5, 25.5)
mean_mu_e_thresh = (24.5, 27.5)
true_catalog(MOCK_CATALOG, MOCK_DIR_PATH, TRAIN_DIR_PATH, mu_0_thresh, mean_mu_e_thresh)
