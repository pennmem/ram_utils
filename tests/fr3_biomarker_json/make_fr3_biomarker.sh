#!/bin/bash

workspace=/scratch/busygin/FR3_biomarkers
subject=R1191J
n_channels=256
anode=RX1
anode_num=1
cathode=RX2
cathode_num=2
pulse_frequency=200
pulse_duration=500
target_amplitude=1000

python fr3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
