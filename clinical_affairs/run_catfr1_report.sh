cd ../tests/fr1_report
python fr1_report.py --subject=$1 --task=catFR1 --workspace-dir=/scratch/RAM_reports/catFR1_reports --mount-point='' "${@:2}"
cd -
