#!/bin/bash

workspace=/scratch/system3_configs/THR3
subject=R1154D
n_channels=128
anode=LTCG29
anode_num=29
cathode=LTCG30
cathode_num=30
pulse_frequency=200
pulse_duration=500
target_amplitude=1000


cd ../tests/thr3_biomarker
python thr3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
cd -