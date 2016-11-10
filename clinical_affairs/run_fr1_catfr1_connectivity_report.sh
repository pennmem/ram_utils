#!/bin/bash

source $HOME/.bashrc

qsub $HOME/ram_utils/tests/fr_connectivity_report.sh $1
