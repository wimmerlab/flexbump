#!/bin/bash

export LC_NUMERIC=POSIX

sigma=0.15
for ieff in `seq -0.035 0.01 0.055`
do
    i0=$(python -c "print($ieff + 2*$sigma**2)")
    echo "Running ring_simulation.py with sigma =" $sigma "and i0 =" $i0 ", (i0eff =" $ieff")."
    OMP_NUM_THREADS=8 python3 -W ignore ring_simulation.py -f conf/conf_paper.txt -ntrials 10000 -chunk 50 -cpus 8 -max 0.003 -go_int 0.5 -i_urge 0.46 -i_rest 0.33 -mod 2.0 -i0_init $ieff -i0 $i0 -no_bump -sigmaOU $sigma -i1 0.005 -nframes 8 -tmax 3.0 -cue_duration 0.250 -stim_file Stim_df_10000_8.npy
done