#!/usr/bin/env bash
# Command-line example of running a Ramulator config generation pipeline
python -m ramutils.cli.expconf -s R1365N -x FR5 \
    -e scratch/system3_configs/ODIN_configs/R1365N/R1365N_16NOV2017L0M0STIM.csv \
    --anodes LAD12 LAH11 --cathodes LAD13 LAH12 \
    --target-amplitudes 0.5 0.5 \
    --dest /scratch/ramutils2/demo --force-rerun
