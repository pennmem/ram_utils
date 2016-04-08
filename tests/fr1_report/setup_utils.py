__author__ = 'm'


import sys
import argparse
import os
from os.path import *


class CMLParser(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='Report Generator')
        self.parser.add_argument('--subject', required=False, action='store')
        self.parser.add_argument('--task', required=False, action='store')
        self.parser.add_argument('--workspace-dir', required=False, action='store')
        self.parser.add_argument('--mount-point', required=False, action='store')
        self.parser.add_argument('--python-path', required=False, action='append')
        self.parser.add_argument('--exit-on-no-change', dest='exit_on_no_change', action='store_true')
        self.parser.add_argument('--status-output-dir', required=False, dest='status_output_dir', action='store')

        self.arg_list=[]
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, val):
        self.arg_list.append(name)
        self.arg_list.append(val)

    def configure_python_paths(self,paths):
        for path in paths:
            sys.path.append(path)

    def parse(self):
        print sys.argv
        if len(sys.argv)<=self.arg_count_threshold and len(self.arg_list):
            args = self.parser.parse_args(self.arg_list)
        else:
            args = self.parser.parse_args()

        # making sure that sensible workspace directory is set if user does not provide one
        if not args.workspace_dir:
            args.workspace_dir = abspath(join(expanduser('~'), 'scratch', args.task, args.subject))

        # Converting matlab search paths to proper format
        if not args.mount_point:
            args.mount_point = '/'
        else:
            args.mount_point = abspath(expanduser(args.mount_point))

        # Converting python search paths to proper format
        if not args.python_path:
            args.python_path = [os.getcwd()]
        else:
            args.python_path = [abspath(expanduser(path)) for path in args.python_path]
            args.python_path.insert(0, os.getcwd())

        if not args.exit_on_no_change:
            args.exit_on_no_change = False

        if args.status_output_dir:
            args.status_output_dir = abspath(join(args.workspace_dir, args.status_output_dir))
        else:
            args.status_output_dir = abspath(join(args.workspace_dir, 'status_output'))

        self.configure_python_paths(args.python_path)

        return args


def parse_command_line(command_line_emulation_argument_list=None):

    # COMMAND LINE PARSING
    # command line example:
    # python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

    parser = argparse.ArgumentParser(description='Run Parameter Search Report Generator')
    parser.add_argument('--subject', required=False, action='store')
    parser.add_argument('--task', required=True,  action='store')
    parser.add_argument('--workspace-dir',required=False, action='store')
    parser.add_argument('--mount-point',required=False, action='store')
    parser.add_argument('--python-path',required=False, action='append')
    parser.add_argument('--exit-on-no-change', dest='exit_on_no_change', action='store_true')
    parser.add_argument('--status-output-dir',required=False, dest='status_output_dir', action='store')

    if command_line_emulation_argument_list:
        args = parser.parse_args(command_line_emulation_argument_list)
    else:
        args = parser.parse_args()

    # making sure that sensible workspace directory is set if user does not provide one
    if not args.workspace_dir:
        args.workspace_dir = abspath(join(expanduser('~'),'scratch',args.task, args.subject))

    # Converting matlab search paths to proper format
    if not args.mount_point:
        args.mount_point = '/'
    else:
        args.mount_point = abspath(expanduser(args.mount_point))

    # Converting python search paths to proper format
    if not args.python_path:
        args.python_path=[os.getcwd()]
    else:
        args.python_path = [abspath(expanduser(path)) for path in args.python_path]
        args.python_path.insert(0,os.getcwd())

    if not args.exit_on_no_change:
        args.exit_on_no_change = False

    if args.status_output_dir:
        args.status_output_dir = abspath(join(args.workspace_dir,args.status_output_dir))
    else:
        args.status_output_dir = abspath(join(args.workspace_dir,'status_output'))


    return args


def configure_python_paths(paths):
    for path in paths:
        sys.path.append(path)
