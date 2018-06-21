#!/usr/bin/env bash
set -e
mkdir test_directory
cd test_directory
python -m ramutils.test -m "not rhino" --cov
