#!/usr/bin/env bash

BASEDIR=/home1/zduey/ram_utils/
PYTHON_36_TEST_FOLDER=/scratch/zduey/nightly_build/python_36/
PYTHON_27_TEST_FOLDER=/scratch/zduey/nightly_build/python_27/
RESULTS_DISTRIBUTION=zachduey@gmail.com,leond@sas.upenn.edu,depalati@sas.upenn.edu

cd $BASEDIR
conda env remove -n ramutils_test -y
conda env remove -n ramutils_test_27 -y

echo "Creating python 3 environment"
set -e
conda create -y -n ramutils_test python=3
source activate ramutils_test
conda install -y -c pennmem -c conda-forge --file=$BASEDIR/requirements.txt

echo "Pulling master branch from remote repository"
git stash
git checkout master
git pull origin master
git reset --hard origin/master

# If any tests fail, non-zero error code returned by pytest, so allow shell script to contiue
echo "Running full test suite on python 3.6"
set +e
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report_36.html --self-contained-html --rhino-root=/ --output-dest=$PYTHON_36_TEST_FOLDER
set -e

zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report_36.html -s "Python 3.6 Test Results" $RESULTS_DISTRIBUTION

echo "Creating python 2.7 environment"
conda create -y -n ramutils_test_27 python=2.7
source activate ramutils_test_27
conda install -y -c pennmem -c conda-forge --file=$BASEDIR/requirements.txt

echo "Running full test suite on python 2.7"
set +e
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report_27.html --self-contained-html --rhino-root=/ --output-dest=$PYTHON_27_TEST_FOLDER
set -e

# Zip the htmlcov/ folder and email
zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report_27.html -s "Python 2.7 Test Results" $RESULTS_DISTRIBUTION

# Cleanup
rm coverage.zip
rm report_36.html
rm report_27.html
rm -r $PYTHON_36_TEST_FOLDER/*
rm -r $PYTHON_27_TEST_FOLDER/*

