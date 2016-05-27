cd ../tests/fr1_report_db
python fr1_report.py --subject=$1 --task=RAM_FR1 --workspace-dir=/scratch/pwanda/FR1_reports --mount-point='' --python-path=/home2/losu/ram_utils --python-path=/home2/losu/python/ptsa_latest --python-path=/home1/mswat/extra_libs
cd -
