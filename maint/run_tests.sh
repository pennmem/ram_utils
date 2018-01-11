#!/usr/bin/env bash
conda env remove -n ramutils_test -y
set -e
conda create -y -n ramutils_test python=3
source activate ramutils_test
conda install -y -c pennmem -c conda-forge --file=requirements.txt

git pull origin master
python setup.py install
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report.html --self-contained-html --rhino-root=/ --output-dest=/scratch/zduey

zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report.html -s "Python 3.6 Test Results" zachduey@gmail.com

conda env remove -n ramutils_test_27 -y
set -e
conda create -y -n ramutils_test_27 python=2.7
source activate ramutils_test_27
conda install -y -c pennmem -c conda-forge --file=requirements.txt

python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report.html --self-contained-html --rhino-root=/ --output-dest=/scratch/zduey

# Zip the htmlcov/ folder and email
zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report.html -s "Python 2.7 Test Results" zachduey@gmail.com




