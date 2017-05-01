ps_experiment=$2
if [ $ps_experiment = 'PS4' ]
then
read -p "MEMORY TASK [FR5,PAL5]:" memory_task;
read -p "WORKSPACE DIR:" output_dir
cd ../tests/ps4_report
python ps4_report.py --subject=$1 --task=${memory_task} --workspace-dir="$(realpath ${output_dir})" "${@:3}"
cd-
else
cd ../tests/ps_report
python ps_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/FR_reports --mount-point="" "${@:3}"
cd -
fi