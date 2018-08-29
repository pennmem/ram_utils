#!/usr/bin/env bash
set -e
mkdir test_directory
cd test_directory
PROJECT=$HOME/miniconda/envs/ramutils/lib/python3.6/site-packages/ramutils
pytest ../test -m "not rhino" --cov=$PROJECT -v
