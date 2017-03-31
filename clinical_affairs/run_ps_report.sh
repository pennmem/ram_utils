cd ../tests/ps_report
python ps_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/FR_reports --mount-point="" "${@:3}"
cd -
