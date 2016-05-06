#!/bin/bash

workspace=/scratch/busygin/PAL3_biomarkers
subject=R1162N
n_channels=128
anode=AD2
anode_num=56
cathode=AD3
cathode_num=57
pulse_frequency=200
pulse_count=100
target_amplitude=1000

python pal3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-count=$pulse_count --target-amplitude=$target_amplitude
