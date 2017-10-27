import sys
import argparse
from os.path import *


# FIXME: just use ArgumentParser directly; this isn't doing anything that it can't
class CMLParserCloseLoop3(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='Report Generator')
        self.parser.add_argument('--subject', required=True)
        self.parser.add_argument('--experiment', required=False)
        self.parser.add_argument('--workspace-dir', required=False)
        self.parser.add_argument('--mount-point', required=False)
        self.parser.add_argument('--electrode-config-file', required=True)
        self.parser.add_argument('--pulse-frequency', required=True, type=int)
        self.parser.add_argument('--target-amplitude', required=True, type=float)

        self.parser.add_argument('--anode-num', required=False, type=int)
        self.parser.add_argument('--anode', required=False, default='')
        self.parser.add_argument('--cathode-num', required=False, type=int)
        self.parser.add_argument('--cathode', required=False, default='')

        self.parser.add_argument('--anode-nums', nargs='+', type=int, metavar='ANODE_NUM')
        self.parser.add_argument('--anodes', nargs='+', metavar='ANODE')
        self.parser.add_argument('--cathode-nums', nargs='+', type=int, metavar='CATHODE_NUM')
        self.parser.add_argument('--cathodes', nargs='+', metavar='CATHODE')
        self.parser.add_argument('--min-amplitudes', nargs='+', type=float)
        self.parser.add_argument('--max-amplitudes', nargs='+', type=float)
        self.parser.add_argument('--sessions', nargs='+', type=int)
        self.parser.add_argument('--encoding-only', action='store_true')

        self.arg_list = []
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, *vals):
        self.arg_list.append(name)
        for val in vals:
            self.arg_list.append(val)

    def configure_python_paths(self, paths):
        for path in paths:
            sys.path.append(path)

    def parse(self):
        print sys.argv
        if len(sys.argv) <= self.arg_count_threshold and len(self.arg_list):
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

        # check that target amplitude is in milliamps
        try:
            assert args.target_amplitude < 2 and args.target_amplitude > 0
        except AssertionError:
            raise ValueError('Target amplitude should be between 0 and 2 milliamps')

        return args
