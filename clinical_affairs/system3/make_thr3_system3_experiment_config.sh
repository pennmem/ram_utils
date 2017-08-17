#! /usr/bin/bash

experiment=THR3
workspace=/scratch/system3_configs/${experiment}

#electrode_config_file=/scratch/system3_configs/ODIN_configs/R1328E/R1328E_L0M0STIMaug152017.csv
read -p 'SUBJECT' subject
read -p 'ELECTRODE CONFIG FILE' electrode_config_file
read -p 'ANODE' anode
read -p 'ANODE_NUM' anode_num
read -p 'CATHODE' cathode
read -p 'CATHODE_NUM' cathode_num
read -p 'PULSE FREQUENCY (Hz)' pulse_frequency
read -p 'STIM AMPLITUDE (mA)' target_amplitude

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
