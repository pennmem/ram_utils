__author__ = 'm'

import sys
import argparse
import os
from os.path import *

def parse_command_line():

    # COMMAND LINE PARSING
    # command line example:
    # python ps_report.py --subject=R1056M --experiment=PS2 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

    parser = argparse.ArgumentParser(description='Run Parameter Search Report Generator')
    parser.add_argument('--subject', required=True, action='store')
    parser.add_argument('--experiment', required=True,  action='store')
    parser.add_argument('--workspace-dir',required=False, action='store')
    parser.add_argument('--matlab-path',required=False, action='append')

    parser.add_argument('--python-path',required=False, action='append')


    args = parser.parse_args()

    # making sure that sensible workspace directory is set if user does not provide one
    if not args.workspace_dir:
        args.workspace_dir = abspath(join(expanduser('~'),'scratch',args.experiment, args.subject))

    # Converting matlab search paths to proper format
    if not args.matlab_path:
        args.matlab_path=[os.getcwd()]

    else:
        args.matlab_path = [abspath(expanduser(path)) for path in args.matlab_path]
        args.matlab_path.insert(0,os.getcwd())


    # Converting python search paths to proper format
    if not args.python_path:
        args.python_path=[os.getcwd()]

    else:
        args.python_path = [abspath(expanduser(path)) for path in args.python_path]
        args.python_path.insert(0,os.getcwd())

    return args


def configure_python_paths(paths):
    for path in paths:
        sys.path.append(path)
