
subject=R1111M
localization=0
montage=0
stim_pair=LPOG10-LPOG11


contacts_json="/protocols/r1/subjects/${subject}/localizations/${localization}/montages/${montage}/neuroradiology/current_processed/contacts.json"
contacts_json_output_dir="/scratch/pwanda/FR3_biomarkers/electrode_config"

cd ../../
pwd
echo "---------------"
python system_3_utils/odin_config_tool_generator.py\
 --subject=$subject\
 --contacts-json=$contacts_json\
 --contacts-json-output-dir=$contacts_json_output_dir\
 --stim-channels ${stim_pair}

cd -

