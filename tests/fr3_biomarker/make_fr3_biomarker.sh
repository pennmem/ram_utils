#!/bin/bash

workspace=/scratch/busygin/FR3_biomarkers
subject=R1145J_1
n_channels=128
anode=RD2
anode_num=34
cathode=RD3
cathode_num=35
pulse_frequency=200
pulse_count=100
target_amplitude=1000

python fr3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-count=$pulse_count --target-amplitude=$target_amplitude
