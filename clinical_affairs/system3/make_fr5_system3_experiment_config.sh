#!/usr/bin/env bash
# NOTE:
# THIS IS THE CONFIG FILE GENERATOR FOR PS4 AND FOR FR5.
# PS4 HAS 2 STIM PAIRS, WHILE FR5 HAS ONLY 1 STIM PAIR.
# BY DEFAULT, FR5 CONFIG USES anode1 IF BOTH anode1 AND anode2 ARE PRESENT,
# AND SIMILARLY FOR CATHODES.
# IF YOU DESIRE TO USE anode2 AND cathode2 FOR FR5 CONFIG,
# PLEASE SET anode1 OR cathode1 TO ''
# THANK YOU.

subject=R1200T
experiment=PS4
anode1=LB6
cathode1=LB7
anode2=
cathode2=LB8
target_amplitude=1.0
config_dir='/scratch/pwanda/FR5_biomarkers/electrode_configs'

workspace_dir='/scratch/pwanda/FR5_biomarkers/'

pulse_frequency=200
min_amplitude=0.25
max_amplitude=2.0


stim_pair_1=${anode1}-${cathode1}
stim_pair_2=${anode2}-${cathode2}
if [ -z $anode1 ] || [ -z $cathode1 ];
then stim_pair_1='';
fi
if [ -z $anode2 ] || [ -z $cathode2 ]
then stim_pair_2='';
fi

if ([ -z $stim_pair_1 ] || [ -z $stim_pair_2 ]) && [ $experiment == 'PS4' ];
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
 --max-amplitude=${max_amplitude}