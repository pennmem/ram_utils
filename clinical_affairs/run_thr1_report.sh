
cd ../tests/thr1_report

python thr1_report.py --subject=$1 --task=THR --workspace-dir=/scratch/RAM_reports/THR1_reports --mount-point='' --python-path=/home2/losu/ram_utils --python-path=/home2/losu/python/ptsa_latest --python-path=/home1/mswat/extra_libs

cd -