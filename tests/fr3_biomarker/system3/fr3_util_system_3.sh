#!/bin/bash

workspace=D:/scratch/FR3_utils_system_3
subject=R1247P
#n_channels=128
#anode=LTCG29
#anode_num=29
#cathode=LTCG30
#cathode_num=30
#pulse_frequency=200
#pulse_duration=500
#target_amplitude=1000

python fr3_biomarker.py --workspace-dir=$workspace --subject=$subject
#--n-channels=$n_channels --anode=$anode --anode-num=$anode_num --cathode=$cathode --cathode-num=$cathode_num --pulse-frequency=$pulse_frequency --pulse-duration=$pulse_duration --target-amplitude=$target_amplitude
