from tqdm import tqdm
import os

fits_files = ["/data/william/ngc3115/c4d_170216_050619_osi_g_v2.fits.fz", "/data/william/ngc3115/c4d_170216_061516_osi_r_v2.fits.fz"]
mask_files = ["c4d_170216_050619_osd_g_v2.fits.fz", "c4d_170216_061516_osd_r_v2.fits.fz"]

for i, filename in tqdm(enumerate(fits_files)):
    # change the filename split according to the file address
    dir_name = filename.split("/")[4].split(".")[0]
    print("looking at... ", dir_name)

    full_path = os.path.join("/data/marcos-cutouts", dir_name)
    mask = mask_files[i]
    make_and_segment_mosaic(filename, mask, 1000, 0.2, full_path)
    # inside each directory we'll have all the cutouts from each .fits.fz file
