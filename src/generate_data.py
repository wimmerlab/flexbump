#!/bin/python3

import numpy as np
import pandas as pd
import sys

sys.path.append('./lib')
from lib_analysis import get_sqr_errors
import subprocess
import logging
import pathlib
import glob

logging.getLogger().setLevel('DEBUG')

logging.info('Simulating, this could take a while...')
rc = subprocess.call("./run_simulation.sh")

folder = sys.argv[1]
output_folder = pathlib.Path('./results/').joinpath(folder)
logging.info(f'Copying data to {output_folder}...')
if not output_folder.exists():
    output_folder.mkdir(parents=True)
try:
    files = glob.glob("./results/simu_10000_*")
    for f in files:
        cmd_copy = f"cp -i {f} {output_folder.as_posix()}"
        logging.info(f"{cmd_copy}")
        process = subprocess.Popen(cmd_copy.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    logging.info("Done")
except FileNotFoundError:
    pass

logging.info("Generating new stats file...")
stats = pd.read_csv("./results/integration_stats_frames-8_t-2.0_template.csv", index_col=0)
# Reset stats (if any)
stats0 = stats[0:1].copy()
stats0 = pd.DataFrame().reindex_like(stats0)
stats_file = output_folder.joinpath("integration_stats_frames-8_t-2.0.csv")
stats0.to_csv(stats_file)
# Generate stats
dat = get_sqr_errors(data_dir=output_folder, stats_file=stats_file, compute_errors=True)
stats = pd.read_csv(stats_file, index_col=0)

logging.info("Generating auxiliary files...")
i0s = [0.02, 0.05, 0.08]

save_path = output_folder.joinpath('modified')
if not save_path.exists():
    save_path.mkdir()
for i0 in i0s:
    filename = pathlib.Path(stats[stats.i0 == i0].filename.to_list()[0])
    data = np.load(str(filename.with_suffix('.npy')), allow_pickle=True)
    data.pop('rates')
    df = data['data']
    new_path = save_path.joinpath(filename.stem.replace('t-2.0', 't-3.0'))
    np.save(str(new_path.with_suffix('')), data, allow_pickle=True)
    df.to_csv(new_path.with_suffix('.csv'))
