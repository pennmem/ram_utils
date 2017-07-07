
import json

import prompt_toolkit
from prompt_toolkit import validation
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

from ..system_3_utils.ram_tasks.CMLParserClosedLoop5 import CMLParserCloseLoop5
from ..system_3_utils.ram_tasks.CMLParserClosedLoop3 import CMLParserCloseLoop3

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

def biomarker_parser(experiment):
    if experiment.endswith('3'):
        return CMLParserCloseLoop3()
    elif experiment.endswith('5'):
        return CMLParserCloseLoop5()

def system_to_method(system):
    if system=='2':
        return biomarker_scripts
    elif system=='3':
        return experiment_config_scripts
    else:
        raise RuntimeError('System %s not supported'%system)


class WordValidator(validation.Validator):
    def __init__(self,words):
        self.words=words

    def validate(self, document):
        if document.text not in self.words:
            raise validation.ValidationError


class DummyValidator(validation.Validator):
    def validate(self, document):
        return


def complete_and_validate(words):
    return {u'completer':completers.WordCompleter(words),
            u'validator':WordValidator(words)
            }


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
args_dict[('3','PS4_PAL5')] = args_dict[('3','PS4_PAL5')]+('classifier_type_to_output',)

path_arguments = (u'electrode_config_file', u'workspace_dir', u'mount_point')


class Args(object):
    def __init__(self):
        for attr in (two_three_args + three_ps_args + three_five_args):
            self.__setattr__(attr,'')


def get_experiment_completer_validator(system):
    possible_experiments_dict = system_to_method(system)
    return complete_and_validate(possible_experiments_dict.keys())


def get_arg_completer_validator(index_reader,argument,args):
    cv_dict = {u'completer':None,
               u'validator':None}
    if argument in path_arguments:
        cv_dict[u'completer']=completers.PathCompleter()
        cv_dict[u'validator']=DummyValidator()
    elif 'anode' in argument or 'cathode' in argument:
        split_subject = args['subject'].split('_')
        subject = split_subject[0]
        montage = 0 if len(split_subject)==1 else split_subject[-1]
        with open(index_reader.get_value('contacts',subject=subject,montage=montage)) as cfid:
            contacts = json.load(cfid)[subject]['contacts']
        if 'num' in argument:
            completions = [unicode(contacts[args['anode'] if 'anode' in argument else args['cathode']]['channel'])]
        else:
            completions = sorted(contacts.keys())
        cv_dict = complete_and_validate(completions)
    return cv_dict

def get_args(system_no,experiment,args,index_reader):
    args_list = args_dict[(system_no,experiment)] + (u'workspace_dir',u'sessions')
    anode_args = [x for x in args_list if 'anode' in x]
    cathode_args = [x for x in args_list if 'cathode' in x]
    parser = biomarker_parser(experiment)
    parser.arg('--mount-point',args['mount_point'])

    for argument in args_list:
        if 'mount_point' in args and argument in path_arguments:
            default = args['mount_point']
        else:
            default=u''
        arg_val = prompt_toolkit.prompt(unicode(argument.upper().replace('_', ' ') + ': '),
                                        default=default,
                                        **get_arg_completer_validator(index_reader,argument, args))
        args[argument] = arg_val
        if argument not in anode_args+cathode_args and arg_val and not any([s in argument for s in ['min','max']]):
            parser.arg('--%s'%argument.replace('_','-'),arg_val)
    if 'min_amplitude_1' in args:
        parser.arg('--min-amplitudes',args['min_amplitude_1'],args['min_amplitude_2'])
        parser.arg('--max-amplitudes',args['max_amplitude_1'],args['max_amplitude_2'])
    for a in ('subject', 'experiment'):
        parser.arg('--%s'%a,args[a])
    if experiment.endswith('5') and 'target_amplitude' not in args_list:
        parser.arg('--target-amplitude','1.0')
    parsed_args = parser.parse()
    parsed_args.anodes = [args[k] for k in anode_args]
    parsed_args.cathodes = [args[k] for k in cathode_args]
    return parsed_args

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

def prelim_experiment(experiment):
    return experiment.split('_')[-1].replace('5','1').replace('3','1').replace('C','c')

def main():
    system_no = prompt_toolkit.prompt(u'System #: ',**complete_and_validate(['2','3']))
    experiment = prompt_toolkit.prompt(u'Classifier Experiment: ', **get_experiment_completer_validator(system_no))
    mount_point = prompt_toolkit.prompt(u'Mount Point: ',default=u'/',completer=completers.PathCompleter())
    biomarker_maker = system_to_method(system_no)[experiment]
    jr = IndexReader.JsonIndexReader(path.join(mount_point,'protocols/r1.json'))
    new_experiment = False
    if experiment=='PS4_PAL5':
        new_experiment= prompt_toolkit.prompt(u'Task Laptop Experiment: ',default=u'PS4_CatFR5')
    subject = prompt_toolkit.prompt(u'Subject: ',
                                    completer=completers.WordCompleter(jr.subjects(experiment=prelim_experiment(experiment))))
    args= {}
    args['subject'] = subject
    args['task']= new_experiment or experiment
    args['experiment'] = new_experiment or experiment
    args['mount_point']=mount_point
    parsed_args = get_args(system_no,experiment,args,jr)
    if system_no == '3':
        parsed_args.experiment = parsed_args.experiment.replace('cat','Cat')
    print(parsed_args)
    prompt_toolkit.prompt(u'Continue?')
    biomarker_maker.make_biomarker(parsed_args)


def try_pal5():
    class obj(object):
        pass
    args=obj()
    args.subject='R1312N'
    args.experiment='PAL5'
    args.stim_anode = 'G10'
    args.stim_cathode = 'G11'
    args.anode2 = 'G11'
    args.cathode2 = 'G12'
    args.anodes=[args.stim_anode,args.anode2]
    args.cathodes=[args.stim_cathode,args.cathode2]
    args.target_amplitude=1.0
    args.workspace_dir = '/Users/leond'
    args.mount_point='/Volumes/rhino_root'
    args.min_amplitude=args.max_amplitude=0.5
    args.sessions=[]
    pal5_util_system_3.make_biomarker(args)