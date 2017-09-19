cd ../tests/th1_report
python th1_report.py --subject=$1 --task=TH1 --workspace-dir=/scratch/RAM_reports/TH1_reports --mount-point='' "${@:2}"
cd -
