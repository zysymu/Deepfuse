from scan import Scanner
from tqdm import tqdm
import os
import pandas as pd

STAMPSIZE = 150

# simulations
"""
sim_dir = "/home/marcostidball/ic-astro/PROJECT/AI/subsamples/sub_point_061516_r"
final_dir = "/home/marcostidball/ic-astro/PROJECT/AI/train-cuts/NEW-GALAXIES"

for d in os.listdir(sim_dir):
    num = d.split("_")[1]

    full_path = os.path.join(sim_dir, d)
    print(f"looking at... {d}")    

    for f in os.listdir(full_path):
        f_path = os.path.join(full_path, f)
        
        if f == f"mock_{num}_r.fits":
            try:
                scan = Scanner(filename=f_path, stamp_size=STAMPSIZE, mode="simulation-30")
                scan.thresholds()
                #scan.show_stamps(title=f)
                scan.save_stamps(dir_name=os.path.join(final_dir, f), catalog=False)
    
            except AssertionError as e:
                print(e)
                continue
"""

# candidates
candidate_dir = "/home/marcostidball/ic-astro/PROJECT/aaa/a"
final_dir = "/home/marcostidball/ic-astro/PROJECT/AI/CANDIDATE-STAMPS"

os.mkdir(final_dir)

for f in os.listdir(candidate_dir):
    f_path = os.path.join(candidate_dir, f)

    try:
        scan = Scanner(filename=f_path, stamp_size=STAMPSIZE, mode="simulation-30")
        scan.thresholds()
        #scan.show_stamps(title=f)
        scan.save_stamps(dir_name=os.path.join(final_dir, f), catalog=False)

    except AssertionError as e:
        print(e)
        continue
