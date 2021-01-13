from tqdm import tqdm
import os
from mosaic import make_and_segment_mosaic

fits_files = ["/data/william/ngc3115/c4d_170216_050619_osi_g_v2.fits.fz", "/data/william/ngc3115/c4d_170217_075805_osi_g_v2.fits.fz"]
mask_files = ["masks/old/c4d_170216_050619_osd_g_v2.fits.fz", "masks/c4d_170217_075805_osd_g_v2.fits.fz"]

for i, filename in tqdm(enumerate(fits_files)):
    # change the filename split according to the file address
    dir_name = filename.split("/")[4].split(".")[0]
    print("looking at... ", dir_name)

    full_path = os.path.join("/data/marcos-cutouts/fo-real", dir_name)
    mask = mask_files[i]
    make_and_segment_mosaic(filename, mask, 1000, 0.2, full_path)
    # inside each directory we'll have all the cutouts from each .fits.fz file
