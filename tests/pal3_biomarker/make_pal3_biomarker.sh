#!/bin/bash

workspace=/scratch/busygin/PAL3_biomarkers
subject=R1175N
n_channels=128
anode=LPT5
anode_num=19
cathode=LPT6
cathode_num=20
pulse_frequency=100
pulse_duration=500
target_amplitude=1250

python pal3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
