

# subject=R1111M
read -p "SUBJECT:" subject
read -p "LOCALIZATION:" localization
#localization=0
read -p "MONTAGE:" montage
#montage=0
#stim_pair=LPOG10-LPOG11
read -p "STIM PAIR[S]:" stim_pair

contacts_json="/protocols/r1/subjects/${subject}/localizations/${localization}/montages/${montage}/neuroradiology/current_processed/contacts.json"
#contacts_json_output_dir="/home1/leond"
read -p "Output Directory:" contacts_json_output_dir

if [ -z $stim_pair ]
then
   stim_command=""
else
   stim_command="--stim-channels "${stim_pair}
fi

cd ../../
pwd
echo "---------------"
python system_3_utils/odin_config_tool_generator.py\
 --subject=$subject\
 --contacts-json-output-dir=$contacts_json_output_dir\
 ${stim_command}

cd -

