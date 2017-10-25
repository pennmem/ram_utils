#!/bin/bash

workspace=/scratch/busygin/TH3_biomarkers
subject=R1145D
n_channels=128
anode=LTG3
anode_num=3
cathode=LTG4
cathode_num=4
pulse_frequency=200
pulse_duration=500
target_amplitude=1000

python th3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
