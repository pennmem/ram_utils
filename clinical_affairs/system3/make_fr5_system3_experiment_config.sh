#!/usr/bin/env bash

subject=R1200T
anode1=LB6
cathode1=LB7
anode2=LB7
cathode2=LB8
target_amplitude=1.0
config_dir='/scratch/pwanda/FR5_biomarkers/electrode_configs'

workspace_dir='/scratch/pwanda/FR5_biomarkers/'

pulse_frequency=200
min_amplitude=0.25
max_amplitude=2.0



python ../../system_3_utils/odin_config_tool_generator.py --subject=${subject}\
  --contacts-json-output-dir=${config_dir}\
  --stim-channels ${anode1}-${cathode1} ${anode2}-${cathode2}


python ../../tests/fr5_biomarker/system3/fr5_util_system_3.py\
 --subject=${subject}\
 --workspace-dir=${workspace_dir}\
 --experiment=FR5\
 --electrode-config-file=${config_dir}/contacts${subject}.csv\
 --anodes ${anode1} ${anode2}\
 --cathodes ${cathode1} ${cathode2}\
 --pulse-frequency=${pulse_frequency}\
 --target-amplitude=${target_amplitude}\
 --min-amplitude=${min_amplitude}\
 --max-amplitude=${max_amplitude}