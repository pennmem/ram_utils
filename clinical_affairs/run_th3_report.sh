cd ../tests/th_stim_report
python th_stim_report.py --subject=$1 --task=TH3 --workspace-dir=/scratch/RAM_reports/TH3_reports --mount-point='' "${@:2}"
cd -
