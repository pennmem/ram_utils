#!/usr/bin/env bash
# Command-line example of running a Ramulator config generation pipeline
python -m ramutils.cli.expconf \
    -s R1364C -x FR5 \
    -e scratch/system3_configs/ODIN_configs/R1364C/R1364C_06NOV2017L0M0STIM.csv \
    --anodes AMY7 TOJ7 --cathodes AMY8 TOJ8 \
    --target-amplitudes 0.5 0.5 \
    --root ~/mnt/rhino --dest ~/tmp/ramutils2 --force-rerun
