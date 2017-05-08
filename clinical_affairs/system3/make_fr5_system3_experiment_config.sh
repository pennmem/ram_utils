#!/usr/bin/env bash
# NOTE:
# THIS IS THE CONFIG FILE GENERATOR FOR PS4_FR5 AND FOR FR5.

subject=R1302M
experiment=PS4_FR5
anode1=LTG29
cathode1=LTG30
anode2=LTG30
cathode2=LTG31
target_amplitude=1.9
min_amplitude=0.1
max_amplitude=1.9

today=$(date +%m_%d_%y)

config_dir="/scratch/pwanda/${experiment}_biomarkers/${subject}_${task}_\
    ${anode1}_${cathode1}_${anode2}_${cathode2}_${max_amplitude}mA_${today}/electrode_configs"

workspace_dir="/scratch/pwanda/${experiment}_biomarkers/${subject}_${task}_\
    ${anode1}_${cathode1}_${anode2}_${cathode2}_${max_amplitude}mA_${today}"

pulse_frequency=200
stim_pair_1=${anode1}-${cathode1}
stim_pair_2=${anode2}-${cathode2}

if [ -z $stim_pair_1 ] || [ -z $stim_pair_2 ];
then echo "Insufficient number of stim pairs for PS4 config"; exit 0
fi


python ../../system_3_utils/odin_config_tool_generator.py --subject=${subject}\
  --contacts-json-output-dir=${config_dir}\
  --stim-channels ${stim_pair_1} ${stim_pair_2}


python ../../tests/fr5_biomarker/system3/fr5_util_system_3.py\
 --subject=${subject}\
 --workspace-dir=${workspace_dir}\
 --experiment=${experiment}\
 --electrode-config-file=${config_dir}/contacts${subject}.csv\
 --anodes ${anode1} ${anode2}\
 --cathodes ${cathode1} ${cathode2}\
 --pulse-frequency=${pulse_frequency}\
 --target-amplitude=${target_amplitude}\
 --min-amplitude=${min_amplitude}\
 --max-amplitude=${max_amplitude} "${@:1}"
