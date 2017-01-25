__author__ = 'm'


import sys
import argparse
import os
from os.path import *


class CMLParserSystem3(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='Report Generator')
        self.parser.add_argument('--subject', required=True, action='store')
        self.parser.add_argument('--experiment', required=False, action='store')
        self.parser.add_argument('--workspace-dir', required=False, action='store')
        self.parser.add_argument('--mount-point', required=False, action='store')
        self.parser.add_argument('--electrode-config-file', required=True, action='store')
        # self.parser.add_argument('--stim-electrode-pair', required=True, action='store')

        # self.parser.add_argument('--python-path', required=False, action='append')
        # self.parser.add_argument('--n-channels', required=True, action='store',type=int)
        self.parser.add_argument('--anode-num', required=True, action='store',type=int)
        self.parser.add_argument('--anode', required=False, action='store',default='')
        self.parser.add_argument('--cathode-num', required=True, action='store',type=int)
        self.parser.add_argument('--cathode', required=False, action='store',default='')

        # self.parser.add_argument('--pulse-frequency', required=True, action='store',type=int)
        # self.parser.add_argument('--pulse-duration', required=True, action='store',type=int)
        # self.parser.add_argument('--target-amplitude', required=True, action='store',type=int)


        self.arg_list=[]
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, val=None):
        self.arg_list.append(name)
        if val is not None:
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
        # if not args.python_path:
        #     args.python_path = [os.getcwd()]
        # else:
        #     args.python_path = [abspath(expanduser(path)) for path in args.python_path]
        #     args.python_path.insert(0, os.getcwd())


        # self.configure_python_paths(args.python_path)

        return args


