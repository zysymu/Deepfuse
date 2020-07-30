from astropy.io import fits
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

NGC_CATALOG = "/home/marcostidball/ic-astro/PROJECT/NCG3115_candidates.fits"
CATALOGS_PATH = "/home/marcostidball/ic-astro/PROJECT/CANDIDATES-FITS" # contains test catalogs and folders containing stamps of different sizes

hdu_list = fits.open(NGC_CATALOG, memmap=True)
evt_data = Table(hdu_list[1].data)
hdu_list.close()

ra1 = evt_data["ra"]*u.degree
dec1 = evt_data["dec"]*u.degree
goal = SkyCoord(ra=ra1, dec=dec1)

max_sep = 2. * u.arcsec

good_catalogs = {"name": None, "size": None}
error_catalogs = []

for f in tqdm(os.listdir(CATALOGS_PATH)):
    if os.path.isfile(os.path.join(CATALOGS_PATH, f)):
        c = pd.read_csv(os.path.join(CATALOGS_PATH, f))
        
        try: 
            ra2 = (c["ra"].to_numpy()) *u.degree
            dec2 = (c["dec"].to_numpy()) *u.degree
            catalog = SkyCoord(ra=ra2, dec=dec2)

            idx, d2d, d3d = goal.match_to_catalog_sky(catalog)
            sep_constraint = d2d <= max_sep
            goal_matches = goal[sep_constraint]
            catalog_matches = catalog[idx[sep_constraint]]
            
            if len(catalog_matches) >= 20:          
                good_catalogs["name"] = f
                good_catalogs["size"] = len(catalog)

        except IndexError as e:
            error_catalogs.append(f)
            continue

print("good ones: ", good_catalogs)
print("weird ones: ", error_catalogs)




