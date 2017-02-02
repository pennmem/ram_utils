#!/bin/bash

subject=R1250N
experiment=PAL33
workspace=/scratch/pwanda/system3/$exeperiment
electrode_config_file=/scratch/electrode_configs/R1250N_PAL3.bin
anode=RD7
anode_num=95
cathode=RE1
cathode_num=97
pulse_frequency=100
target_amplitude=1000

#cd ../../tests/fr3_biomarker_json/system3
cd ../../

python tests/pal3_biomarker_json/system3/pal3_util_system_3.py\
--workspace-dir=$workspace\
--experiment=$experiment\
--electrode-config-file=$electrode_config_file\
--subject=$subject\
--anode=$anode\
--anode-num=$anode_num\
--cathode=$cathode\
--cathode-num=$cathode_num\
--pulse-frequency=$pulse_frequency\
--target-amplitude=$target_amplitude

cd -
