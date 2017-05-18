cd ../tests/fr_stim_report
python fr_stim_report.py --subject=$1 --task=FR3 --workspace-dir=/scratch/RAM_reports/FR3_reports --mount-point='' "${@:2}"
cd -
