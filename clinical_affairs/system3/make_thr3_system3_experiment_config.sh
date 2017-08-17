#! /usr/bin/bash

subject=R1328E
experiment=THR3
workspace=/scratch/system3_configs/${experiment}

electrode_config_file=/scratch/system3_configs/ODIN_configs/R1328E/R1328E_L0M0STIMaug152017.csv
anode=5Ld8
anode_num=48
cathode=5Ld9
cathode_num=49
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
