#!/bin/bash
cd ../tests/fr4_biomarker 

workspace=../../clinical_affairs/fr4_biomarkers
subject=ISAAC
anode=RD2
anode_num=34
cathode=RD3
cathode_num=35
pulse_frequency=200
target_amplitude=1000

python fr4_biomarker.py --subject=$subject --anode=$anode --anode_num=$anode_num --cathode=$cathode --cathode_num=$cathode_num --pulse_frequency=$pulse_frequency --target_amplitude=$target_amplitude --workspace_dir=$workspace
