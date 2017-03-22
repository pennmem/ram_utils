#!/usr/bin/env bash
# NOTE: BY DEFAULT, FR5 CONFIG USES ANODE1 IF BOTH ANODE1 AND ANODE2 ARE PRESENT,
# AND SIMILARLY FOR CATHODES.
# IF YOU DESIRE TO USE ANODE2 AND CATHODE2 FOR FR5 CONFIG,
# PLEASE SET ANODE1 AND CATHODE1 TO ''.
# THANK YOU.

subject=R1200T
experiment=PS4
anode1=LB6
cathode1=LB7
anode2=LB7
cathode2=
target_amplitude=1.0
config_dir='/scratch/pwanda/FR5_biomarkers/electrode_configs'

workspace_dir='/scratch/pwanda/FR5_biomarkers/'

pulse_frequency=200
min_amplitude=0.25
max_amplitude=2.0


stim_pair_1=${anode1}-${cathode1}
stim_pair_2=${anode2}-${cathode2}
if [ -z $anode1 ];
then stim_pair_1='';
elif [ -z $cathode1 ];
then stim_pair_1='';
fi
if [ -z $anode2 ];
then stim_pair_2='';
elif [ -z $cathode2 ];
then stim_pair_2='';
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
 --max-amplitude=${max_amplitude}