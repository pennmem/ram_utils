#!/bin/bash

workspace=/scratch/leond/TH3_biomarkers
subject=R1294C
n_channels=128
anode=RHH11
anode_num=19
cathode=RHH12
cathode_num=20
pulse_frequency=200
pulse_duration=500
target_amplitude=1000

cd ../tests/th3_biomarker
python th3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
cd -
