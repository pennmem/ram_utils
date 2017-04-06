#!/usr/bin/env bash
# NOTE:
# IN CASES WHEN THE TWO STIM PAIRS HAVE DIFFERENT MAX AMPLITUDES,
# IT SHOULD BE SPECIFIED LIKE SO:
# max_amplitude='2.0 1.0'
# ADDITIONALLY, FOR FR5, anode1 AND cathode1 ARE THE PAIR THAT WILL BE STIMULATED


subject=R1111M
experiment=FR5
anode1=LPOG10
cathode1=LPOG11
anode2=LPOG11
cathode2=LPOG12
target_amplitude=1.0
min_amplitude=0.25
max_amplitude=2.0



config_dir='/scratch/pwanda/FR5_biomarkers/electrode_configs'

workspace_dir='/scratch/pwanda/FR5_biomarkers/'

pulse_frequency=200


stim_pair_1=${anode1}-${cathode1}
stim_pair_2=${anode2}-${cathode2}
if [ -z $anode1 ] || [ -z $cathode1 ];
then stim_pair_1='';
fi
if [ -z $anode2 ] || [ -z $cathode2 ]
then stim_pair_2='';
fi

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
 --max-amplitude=${max_amplitude}