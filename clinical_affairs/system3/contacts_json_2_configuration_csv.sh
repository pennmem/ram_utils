
subject=R1250N
localization=0
montage=0

contacts_json="/D/protocols/r1/subjects/${subject}/localizations/${localization}/montages/${montage}/neuroradiology/current_processed/contacts.json"
contacts_json_output_dir="/D/experiment_configs"

cd ../../
pwd
echo "---------------"
python system_3_utils/odin_config_tool_generator.py\
 --subject=$subject\
 --contacts-json=$contacts_json\
 --contacts-json-output-dir=$contacts_json_output_dir\

cd -

