#
# This script generates mixed-mode (aka bipolar)  or monopolar electrode config file (.csv) that is an input to  Odin Configuration Tool
# `It is a good idea to specify output dir individually for each patient/configuration
#
#
#

# subject=R1111M
read -p "SUBJECT:" subject
read -p "JACKSHEET or CONFIG FILE: " jacksheet
read -p "STIM PAIR[S]:" stim_pair

read -p 'Use mixed-mode referencing (aka bipolar ENS referencing )[y/n] : ' bipolar_referencing

leads="/data/eeg/${subject}/tal/leads.txt"

#contacts_json_output_dir="/home1/leond"
read -p "Output Directory:" output_dir

if [ $bipolar_referencing = "y" ] || [ $bipolar_referencing = "Y" ]; then
    bipolar="--bipolar"
else
    bipolar=""
fi

if [ -z $stim_pair ]
then
   stim_command=""
else
   stim_command="--stim-channels "${stim_pair}
fi

cd ../../
pwd
echo "---------------"
python system_3_utils/jacksheet_2_odin_config.py\
 --subject=$subject\
 --output-dir=$output_dir\
 --jacksheet=$jacksheet\
 --leads=${leads}\
 ${bipolar}\
 ${stim_command}

cd -

