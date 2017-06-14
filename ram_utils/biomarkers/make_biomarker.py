import json

import prompt_toolkit
from  prompt_toolkit.contrib import completers

from fr3_biomarker import fr3_biomarker
from fr3_biomarker.system3 import fr3_util_system_3
from fr5_biomarker import fr5_biomarker
from fr5_biomarker.system3 import fr5_util_system_3
from pal3_biomarker import pal3_biomarker
from pal3_biomarker.system3 import pal3_util_system_3
from pal5_biomarker.system3 import pal5_util_system_3
from ps4_pal5_biomarker.system3 import ps4_pal5_util_system_3
from th3_biomarker import th3_biomarker

biomarker_scripts = {
    'FR3': fr3_biomarker,
    'FR5': fr5_biomarker,
    'PAL3': pal3_biomarker,
    'TH3': th3_biomarker
}

experiment_config_scripts = {
    'FR3': fr3_util_system_3,
    'FR5': fr5_util_system_3,
    'catFR5': fr5_util_system_3,
    'PS4_FR5': fr5_util_system_3,
    'PS4_catFR5': fr5_util_system_3,
    'PAL3': pal3_util_system_3,
    'PAL5': pal5_util_system_3,
    'PS4_PAL5': ps4_pal5_util_system_3
}

system_to_method ={
    '2':biomarker_scripts,
    '3':experiment_config_scripts
}

system_completer = completers.WordCompleter(['2','3'])

two_three_args = ('n_channels','anode','anode_num','cathode','cathode_num','pulse_frequency','pulse_duration','target_amplitude')

three_ps_args = ('anode1','cathode1','min_amplitude_1',
                 'max_amplitude_1','anode2','cathode2','min_amplitude_2','max_amplitude_2','electrode_config_file')

three_five_args = ('stim_anode','stim_cathode','anode2','cathode2','target_amplitude','electrode_config_file')

args_dict ={('2',x):two_three_args for x in ['FR3','PAL3','TH3']}
args_dict.update({('3',x):three_ps_args for x in ['PS4_FR5','PS4_catFR5','PS4_PAL5']})
args_dict.update({('3',x):three_five_args for x in ['FR5','catFR5','PAL5']})

class Args(object):
    """ A perfectly generic object."""
    pass


def get_system_completer(system):
    return completers.WordCompleter(system_to_method[system].keys())

def get_arg_completer(argument):
    if 'file' in argument:
        return completers.PathCompleter(min_input_len=3)


def get_args(system_no,experiment):
    args_list = args_dict[(system_no,experiment)]
    args=Args()
    for argument in args_list:
        arg_val = prompt_toolkit.prompt(argument.uppercase().replace('_',' '),completer=get_arg_completer(argument))
        args.__setattr__(argument,arg_val)
    return args


def load_args_from_json(jsn_path):
    with open(jsn_path) as jsn_file:
        jsn_dict = json.load(jsn_file)
    args = Args()
    for k,v in jsn_dict:
        args.__setattr__(k,v)
    return args




if __name__=='__main__':
    system_no = prompt_toolkit.prompt('System #:',completer=system_completer)
    subject = prompt_toolkit.prompt('Subect: ')
    experiment = prompt_toolkit.prompt('Experiment:',completer=get_system_completer(system_to_method[system_no]))
    args = get_args(system_no,experiment)
    args.subject=subject
    args.task=experiment
    if system_no == '3':
        args.task = args.task.replace('cat','Cat')
    system_to_method[system_no][experiment].make_biomarker(args)
