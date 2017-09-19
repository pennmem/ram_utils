cd ../tests/ps_th_report
python ps_th_report.py --subject=$1 --experiment=$2 --workspace-dir=/scratch/pwanda/TH_reports --mount-point="" "${@:3}"
cd -
