#!/usr/bin/env bash
set -ex

TESTDIR=test_directory
ENVDIR=`conda info --json | jq -r .active_prefix`
PROJECT=$ENVDIR/lib/python3.6/site-packages/ramutils

rm -fr $TESTDIR
mkdir -p $TESTDIR
cd $TESTDIR
python -m ramutils.test --output-dest=$PWD --cov=$PROJECT | tee test_output.txt
