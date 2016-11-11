#!/bin/bash

source $HOME/.bashrc

qsub -v subject=$1 $HOME/ram_utils/tests/fr_connectivity_report.sh
