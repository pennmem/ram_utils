from string import Template
from pprint import pprint
import subprocess
import os
import time
import shutil
import argparse
import sys

VERSION=2.04

FILENAME = '{subject}_{date}_FR4_{anode}_{cathode}_{pulse_frequency}Hz_{target_amplitude:.2f}mA.biomarker'

def print_stim_config(order = (
        'subject', 'anode', 'anode_num', 'cathode', 'cathode_num',
        'pulse_frequency', 'pulse_duration', 'pulse_count', 'target_amplitude', 
        'burst_frequency', 'burst_count', 'pulse_width',
        'wait_after_word_on', 'version')):
        for key in order:
            n_spaces = 20-len(key)
            print('\t%s:%s%s' % (key,'.'*n_spaces, stim_config[key]))

def make_stim_control(template_file='StimControlTemplate.m', out_dir='./.biomarker_tmp', out_file='StimControl.m'):
    template = Template(open(template_file, 'r').read())
    stim_control = template.substitute(stim_config)
    open(os.path.join(out_dir, out_file), 'w').write(stim_control)

def remove_files(out_dir='./.biomarker_tmp', files=['StimControl.m', 'empty.mat']):
    for file in files:
        os.remove(os.path.join(out_dir, file))
    os.rmdir(out_dir)

def make_biomarker(stim_config, template_file='StimControlTemplate.m', out_dir='.', mat_file='empty.mat'):
    stim_config['pulse_count'] = int(stim_config['pulse_frequency'] * (stim_config['pulse_duration'] / 1000.))
    tmp_dir = os.path.join(out_dir, '.biomarker_tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    make_stim_control(out_dir=tmp_dir)
    shutil.copy(mat_file, os.path.join(tmp_dir, mat_file))
    stim_config['date'] = time.strftime('%Y-%m-%d')
    output_config = stim_config.copy()
    output_config['target_amplitude'] /= 1000.
    zip_file = os.path.join(out_dir, FILENAME.format(**output_config))
    zip_cmd = ['zip -9 -j  %s %s' % (zip_file,  os.path.join(tmp_dir, '*'))]
    subprocess.call(zip_cmd, shell=True)
    remove_files(out_dir=tmp_dir)
    print('Biomarker file %s created with parameters: ' % zip_file)
    print_stim_config()


class FR4ParserBiomarker(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='Report Generator')
        self.parser.add_argument('--subject', required=True, action='store', type=str)
        self.parser.add_argument('--anode', required=True, action='store', type=str)
        self.parser.add_argument('--anode_num', required=True, action='store',type=int)
        self.parser.add_argument('--cathode', required=True, action='store', type=str)
        self.parser.add_argument('--cathode_num', required=True, action='store',type=int)
        self.parser.add_argument('--pulse_frequency', required=True, action='store',type=int)
        self.parser.add_argument('--target_amplitude', required=True, action='store', type=int)
        self.parser.add_argument('--pulse_duration', required=False, action='store', type=int, default=500)
        self.parser.add_argument('--burst_frequency', required=False, action='store', type=int, default=1)
        self.parser.add_argument('--burst_count', required=False, action='store',type=int, default=1)
        self.parser.add_argument('--pulse_width', required=False, action='store',type=int, default=300)
        self.parser.add_argument('--wait_after_word_on', required=False, action='store',type=int, default=1486)
        self.parser.add_argument('--version', required=False, action='store',type=float, default=VERSION)
        self.parser.add_argument('--workspace_dir', required=False, action='store', type=str, default='.')


        self.arg_list=[]
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, val=None):
        self.arg_list.append(name)
        if val is not None:
            self.arg_list.append(val)

    def parse(self):
        if len(sys.argv)<=self.arg_count_threshold and len(self.arg_list):
            args = self.parser.parse_args(self.arg_list)
        else:
            args = self.parser.parse_args()

        return args



if __name__ == '__main__':
    cml_parser = FR4ParserBiomarker(arg_count_threshold=1)
    args = cml_parser.parse()
    stim_config = args.__dict__
    make_biomarker(stim_config, out_dir=args.workspace_dir)


