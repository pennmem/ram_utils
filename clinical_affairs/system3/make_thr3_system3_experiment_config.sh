#!/bin/bash

subject=R1111M
experiment=THR3
workspace=/scratch/system3_configs/${experiment}

electrode_config_file=${workspace}/electrode_config/contactsR1111M.csv
anode=LPOG10
anode_num=10
cathode=LPOG11
cathode_num=11
pulse_frequency=200
target_amplitude=1.0

#cd ../../tests/fr3_biomarker/system3
cd ../../

python tests/thr3_biomarker/system3/thr3_util_system_3.py \
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
