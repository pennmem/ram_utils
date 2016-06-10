#!/bin/bash

subject=R1145J_1
n_channels=128
anode=RD2
anode_num=34
cathode=RD3
cathode_num=35
pulse_frequency=200
target_amplitude=1000

#### DO NOT EDIT BELOW THIS LINE

pulse_duration=500
burst_frequency=1
burst_count=1
pulse_width=300
wait_after_word_on=1486
version=2.04

export PYTHONPATH="$PYTHONPATH:../.."

python fr4_biomarker.py --subject=$subject --n_channels=$n_channels --anode=$anode --anode_num=$anode_num --cathode=$cathode --cathode_num=$cathode_num --pulse_frequency=$pulse_frequency --target_amplitude=$target_amplitude \
    --pulse_duration=$pulse_duration --burst_frequency=$burst_frequency --burst_count=$burst_count --pulse_width=$pulse_width --wait_after_word_on=$wait_after_word_on --version=$version
