#!/bin/bash

workspace=/scratch/pwanda/FR3_biomarkers
subject=R1154D
n_channels=128
anode=LTCG29
anode_num=29
cathode=LTCG30
cathode_num=30
pulse_frequency=200
pulse_count=100
target_amplitude=1000

cd ../tests/fr3_biomarker
python fr3_biomarker.py --workspace-dir=$workspace --subject=$subject --n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-count=$pulse_count --target-amplitude=$target_amplitude
cd -
