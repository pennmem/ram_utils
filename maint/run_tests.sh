#!/usr/bin/env bash

BASEDIR=/home2/RAM_maint/repos/ram_utils/
OUTPUT_FOLDER=/scratch/RAM_maint/nightly_testing/
RESULTS_DISTRIBUTION="zduey@sas.upenn.edu,depalati@sas.upenn.edu,leond@sas.upenn.edu"

cd $BASEDIR
conda env remove -n ramutils_test -y

echo "Creating environment"
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
echo "Running full test suite"
set +e
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=ramutils_test_results.html --self-contained-html --rhino-root=/ --output-dest=$OUTPUT_FOLDER
set -e
zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a ramutils_test_results.html -s "Ramutils Test Results" $RESULTS_DISTRIBUTION

# Cleanup
rm coverage.zip
rm ramutils_test_results.html
rm -r $OUTPUT_FOLDER/*
