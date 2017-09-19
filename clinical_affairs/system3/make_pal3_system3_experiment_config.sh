#!/bin/bash

subject=R1250N
experiment=PAL3
workspace=/scratch/system3_configs/$experiment
electrode_config_file=${workspace}/electrode_configs/R1250N_PAL3.bin
anode=PG10
anode_num=10
cathode=PG11
cathode_num=11
pulse_frequency=100
target_amplitude=1000

#cd ../../tests/fr3_biomarker/system3
cd ../../

python tests/pal3_biomarker_json/system3/pal3_util_system_3.py \
--workspace-dir=$workspace \
--experiment=$experiment \
--electrode-config-file=$electrode_config_file \
--subject=$subject \
--anode=$anode \
--anode-num=$anode_num \
--cathode=$cathode \
--cathode-num=$cathode_num \
--pulse-frequency=$pulse_frequency \
--target-amplitude=$target_amplitude

cd -
