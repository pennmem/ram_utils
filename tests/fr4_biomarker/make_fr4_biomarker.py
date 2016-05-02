from stimulation_config import stim_config
from string import Template
from pprint import pprint
import subprocess
import os
import time
import shutil

def print_stim_config(order = (
        'subject', 'anode', 'anode_num', 'cathode', 'cathode_num',
        'pulse_frequency', 'pulse_count', 'target_amplitude', 
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

def make_biomarker(template_file='StimControlTemplate.m', out_dir='./.biomarker_tmp', mat_file='empty.mat'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    make_stim_control(out_dir=out_dir)
    shutil.copy(mat_file, os.path.join(out_dir, mat_file))
    zip_file = '%s_%d-%d_%dHz_FR4_%s.biomarker' % (stim_config['subject'], stim_config['anode_num'], stim_config['cathode_num'], stim_config['pulse_frequency'], time.strftime('%m-%d-%y'))
    zip_cmd = ['zip -9 -j  %s %s' % (zip_file,  os.path.join(out_dir, '*'))]
    subprocess.call(zip_cmd, shell=True)
    remove_files()
    print('Biomarker file %s created with parameters: ' % zip_file)
    print_stim_config()

if __name__ == '__main__':
    make_biomarker()
