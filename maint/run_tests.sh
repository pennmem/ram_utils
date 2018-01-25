#!/usr/bin/env bash
conda env remove -n ramutils_test -y
set -e
echo "Creating python 3 environment"
conda create -y -n ramutils_test python=3
source activate ramutils_test
conda install -y -c pennmem -c conda-forge --file=requirements.txt

echo "Pulling master branch from remote repository"
git pull origin master

# If any tests fail, non-zero error code returned by pytest, so allow shell script to contiue
echo "Running full test suite on python 3.6"
set +e
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report_36.html --self-contained-html --rhino-root=/ --output-dest=/scratch/zduey/nightly_build/python_36/
set -e

zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report.html -s "Python 3.6 Test Results" zachduey@gmail.com

echo "Creating python 2.7 environment"
set +e
conda env remove -n ramutils_test_27 -y
set -e
conda create -y -n ramutils_test_27 python=2.7
source activate ramutils_test_27
conda install -y -c pennmem -c conda-forge --file=requirements.txt

echo "Running full test suite on python 2.7"
set +e
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report_27.html --self-contained-html --rhino-root=/ --output-dest=/scratch/zduey/nightly_build/python_27/
set -e

# Zip the htmlcov/ folder and email
zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report.html -s "Python 2.7 Test Results" zachduey@gmail.com

