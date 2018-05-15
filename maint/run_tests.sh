#!/usr/bin/env bash

CODEDIR=/home1/zduey/ram_utils
RHINO_ROOT=/
OUTPUT_LOC=/scratch/zduey/nightly_build/
RESULTS_DIST=zachduey@gmail.com

cd $CODEDIR
conda env remove -n ramutils_test -y

echo "Creating python 3 environment"
set -e
conda create -y -n ramutils_test python=3
source activate ramutils_test
conda install -y -c pennmem -c conda-forge --file=$CODEDIR/requirements.txt

echo "Pulling master branch from remote repository"
git stash
git checkout master
git pull origin master

# If any tests fail, non-zero error code returned by pytest, so allow shell script to contiue
echo "Running full test suite on python 3.6"
set +e
python -m pytest ramutils/ --cov=ramutils --cov-report html --html=report_36.html --self-contained-html --rhino-root=/ --output-dest=$OUTPUT_LOC
set -e

zip coverage.zip htmlcov/
echo "Full test suite finished running" | mail -a coverage.zip -a report_36.html -s "Ramutils test results" $RESULTS_DIST

