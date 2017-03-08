__author__ = 'm'


import sys
import argparse
import os
from os.path import *


class OdinConfigToolGeneratorParser(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='Report Generator')
        self.parser.add_argument('--subject', required=True, action='store')
        self.parser.add_argument('--contacts-json', required=True, action='store',default='')
        self.parser.add_argument('--stim-channels',nargs='+',action='store')
        self.parser.add_argument('--contacts-json-output-dir',required=True,action='store',default='')


        self.arg_list=[]
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, *vals):
        self.arg_list.append(name)
        for val in vals:
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


        return args


