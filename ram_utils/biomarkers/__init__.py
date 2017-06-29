
import json

import prompt_toolkit
from  prompt_toolkit.contrib import completers
from ptsa.data.readers import  IndexReader
from os import path

from .fr3_biomarker import fr3_biomarker
from .fr3_biomarker.system3 import fr3_util_system_3
from .fr5_biomarker import fr5_biomarker
from .fr5_biomarker.system3 import fr5_util_system_3
from .pal3_biomarker import pal3_biomarker
from .pal3_biomarker.system3 import pal3_util_system_3
from .pal5_biomarker.system3 import pal5_util_system_3
from .ps4_pal5_biomarker.system3 import ps4_pal5_util_system_3
from .th3_biomarker import th3_biomarker

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


def system_to_method(system):
    if system=='2':
        return biomarker_scripts
    elif system=='3':
        return experiment_config_scripts
    else:
        raise RuntimeError('System %s not supported'%system)

system_completer = completers.WordCompleter(['2','3'])

two_three_args = tuple(unicode(s) for s in
    ('n_channels','anode','anode_num','cathode','cathode_num','pulse_frequency','pulse_duration','target_amplitude',)
                  )

three_ps_args = tuple(unicode(s) for s in ('anode1','cathode1','min_amplitude_1',
                 'max_amplitude_1','anode2','cathode2','min_amplitude_2','max_amplitude_2','electrode_config_file',)
                 )

three_five_args = tuple(unicode(s) for s in
    ('stim_anode','stim_cathode','anode2','cathode2','target_amplitude','electrode_config_file',)
                   )

args_dict ={('2',x):two_three_args for x in ['FR3','PAL3','TH3']}
args_dict.update({('3',x):three_ps_args for x in ['PS4_FR5','PS4_catFR5','PS4_PAL5']})
args_dict.update({('3',x):three_five_args for x in ['FR5','catFR5','PAL5']})


class Args(object):
    def __init__(self):
        for attr in (two_three_args + three_ps_args + three_five_args):
            self.__setattr__(attr,'')

def get_experiment_completer(system):
    possible_experiments_dict = system_to_method(system)
    return completers.WordCompleter(possible_experiments_dict.keys())

def get_arg_completer(argument,args):
    path_arguments = (u'electrode_config_file',u'workspace_dir',u'mount_point')
    if argument in path_arguments:
        return completers.PathCompleter()
    elif 'anode' in argument or 'cathode' in argument:
        split_subject = args.subject.split('_')
        subject = split_subject[0]
        montage = 0 if len(split_subject)==1 else split_subject[-1]
        jr = IndexReader.JsonIndexReader(path.join(args.mount_point,'protocols','r1.json'))
        with open(jr.get_value('contacts',subject=subject,montage=montage)) as cfid:
            contacts = json.load(cfid)[subject]['contacts']
        if 'num' in argument:
            completions = [unicode(contacts[args.anode if 'anode' in argument else args.cathode]['channel'])]
        else:
            completions = sorted(contacts.keys())
        return completers.WordCompleter(completions)


def get_args(system_no,experiment,args):
    args_list = (u'mount_point',)+ args_dict[(system_no,experiment)] + (u'workspace_dir',u'sessions')
    anode_args = [x for x in args_list if 'anode' in x]
    cathode_args = [x for x in args_list if 'cathode' in x]
    for argument in args_list:
        arg_val = prompt_toolkit.prompt(unicode(argument.upper().replace('_',' ')+': '),completer=get_arg_completer(argument,args))
        args.__setattr__(argument,str(arg_val))
    args.anodes = [args.__getattribute__(k) for k in anode_args]
    args.cathodes = [args.__getattribute__(k) for k in cathode_args]

def load_args_from_file(file_path):
    with open(file_path) as jsn_file:
        jsn_dict = json.load(jsn_file)
    args = Args()
    for k,v in jsn_dict:
        args.__setattr__(k,v)
    anode_args = [x for x in jsn_dict.keys() if 'anode' in x]
    cathode_args = [x for x in jsn_dict.keys() if 'cathode' in x]
    args.anodes = [args.__getattribute__(k) for k in anode_args]
    args.cathodes = [args.__getattribute__(k) for k in cathode_args]
    return args


def main():
    system_no = prompt_toolkit.prompt(u'System #: ',completer=system_completer)
    experiment = prompt_toolkit.prompt(u'Experiment: ', completer=get_experiment_completer(system_no))
    subject = prompt_toolkit.prompt(u'Subject: ')
    args = Args()
    args.subject = subject
    args.task = experiment
    args.experiment = experiment
    get_args(system_no,experiment,args)
    if system_no == '3':
        args.task = args.task.replace('cat','Cat')
    system_to_method(system_no)[experiment].make_biomarker(args)
