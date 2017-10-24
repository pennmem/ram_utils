#!/bin/bash

workspace=/scratch/leond/TH3_biomarkers
subject=R1201P_1
n_channels=123
anode=AT3
anode_num=71
cathode=AT4
cathode_num=72
pulse_frequency=50
pulse_duration=500
target_amplitude=1000

python th3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
