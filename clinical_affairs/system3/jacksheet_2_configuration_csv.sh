

# subject=R1111M
read -p "SUBJECT:" subject
read -p "LOCALIZATION:" localization
#localization=0
read -p "MONTAGE:" montage
#montage=0
#stim_pair=LPOG10-LPOG11
read -p "STIM PAIR[S]:" stim_pair

read -p "Mixed mode referencing (aka bipolar referencing) [y/n]" mixed_mode_referencing

contacts_json="/protocols/r1/subjects/${subject}/localizations/${localization}/montages/${montage}/neuroradiology/current_processed/contacts.json"
jacksheet="/data/eeg/${subject}/docs/jacksheet.txt"
leads="/data/eeg/${subject}/tal/leads.txt"

#contacts_json_output_dir="/home1/leond"
read -p "Output Directory:" output_dir

if [ -z $stim_pair ]
then
   stim_command=""
else
   stim_command="--stim-channels "${stim_pair}
fi
# TODO ADD HANDLING OFF THE BIPOLAR SWITCH -  rght now hard coded to be always ON
cd ../../
pwd
echo "---------------"
python system_3_utils/jacksheet_2_odin_config.py\
 --subject=$subject\
 --output-dir=$output_dir\
 --jacksheet=$jacksheet\
 --leads=${leads}\
 --bipolar\
 ${stim_command}

cd -

