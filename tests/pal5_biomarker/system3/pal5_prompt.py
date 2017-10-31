from __future__ import unicode_literals
import sys
from prompt_toolkit import prompt
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.token import Token
from prompt_toolkit.contrib.completers import WordCompleter
from prompt_toolkit.validation import Validator, ValidationError
from os.path import *
import re

from prompt_toolkit.auto_suggest import AutoSuggestFromHistory, AutoSuggest, Suggestion

from prompt_toolkit.contrib.completers import PathCompleter

num_stim_pairs = 2

class Args(object):
    """
    Class that stores output of the command line parsing
    """
    def __init__(self):
        self.anode_nums = None
        self.anode = ''
        self.cathode = ''
        self.anodes = []
        self.cathode_num = None
        self.cathode_nums = None
        self.cathodes = []
        self.electrode_config_file = None
        self.experiment = ''
        self.target_amplitude = None
        self.min_amplitudes = []
        self.max_amplitudes = []
        self.mount_point = ''
        self.pulse_frequency = None
        self.subject = ''
        self.workspace_dir = ''
        self.sessions = None


example_style = style_from_dict({
    # User input.
    Token: '#ff0066',

    # Prompt.
    Token.Username: '#884444 italic',
    Token.At: '#00aa00',
    Token.Colon: '#00aa00',
    Token.Pound: '#00aa00',
    Token.Host: '#000088 bg:#aaaaff',
    Token.Path: '#884444 underline',

    # Make a selection reverse/underlined.
    # (Use Control-Space to select.)
    Token.SelectedText: 'reverse underline',
})


def get_prompt_tokens(cli):
    return [
        (Token.Username, 'john'),
        (Token.At, '@'),
        (Token.Host, 'localhost'),
        (Token.Colon, ':'),
        (Token.Path, '/user/john'),
        (Token.Pound, '# '),
    ]


path_completer = PathCompleter()

experiment_list = ['PAL5', 'PS4_PAL5']

experiment_completer = WordCompleter(experiment_list)


class ExperimentValidator(Validator):
    def __init__(self, experiment_list):
        self.experiment_list = experiment_list

    def validate(self, document):
        exp = document.text
        if not exp in self.experiment_list:
            raise ValidationError(message='Not a valid experiment.'
                                          ' Choose from %s' % ', '.join(self.experiment_list),
                                  cursor_position=len(document.text))  # Move cursor to end of input.


class DirValidator(Validator):
    def validate(self, document):
        dir_path = document.text
        if not exists(dir_path) or not isdir(dir_path):
            raise ValidationError(message='Not a valid directory.'
                                          ' Make sure you select existing directory to which you have write permissions',
                                  cursor_position=len(document.text))  # Move cursor to end of input.


class CSVFileValidator(Validator):
    def validate(self, document):
        file_path = document.text
        if exists(file_path) and isfile(file_path):
            core_name, ext = splitext(file_path)
            if ext.lower() == '.csv':
                return
        raise ValidationError(message='Not a valid file.'
                                      ' Select .csv electrode file',
                              cursor_position=len(document.text))  # Move cursor to end of input.


class TypedNumberValidator(Validator):
    def __init__(self, conversion_fcn, number_type_name):

        self.conversion_fcn = conversion_fcn
        self.number_type_name = number_type_name

    def validate(self, document):
        number_str = document.text
        try:
            self.conversion_fcn(number_str)
        except:

            raise ValidationError(message='Please enter a %s number.' % self.number_type_name,
                                  cursor_position=len(document.text))  # Move cursor to end of input.


class AmplitudeValidator(Validator):
    def __init__(self, min_ampl=0.01, max_ampl=2.0):

        self.min_ampl = min_ampl
        self.max_ampl = max_ampl

    def validate(self, document):
        try:
            ampl = float(document.text)
        except:
            raise ValidationError(message='Please enter a floating point number.' ,
                                  cursor_position=len(document.text))  # Move cursor to end of input.

        if ampl >= self.min_ampl and ampl < self.max_ampl:
            return

        raise ValidationError(message='Amplitude should be between %.3f and %3.f .' % (self.min_ampl, self.max_ampl),
                              cursor_position=len(document.text))  # Move cursor to end of input.


class MaxAmplitudeValidator(Validator):
    def __init__(self, min_ampl_input, min_ampl=0.01, max_ampl=2.0):
        self.min_ampl = min_ampl
        self.max_ampl = max_ampl
        self.min_ampl_input = min_ampl_input
        self.ampl_validator = AmplitudeValidator(self.min_ampl, self.max_ampl)

    def validate(self, document):
        self.ampl_validator.validate(document)

        max_ampl = float(document.text)
        self.min_ampl_input = float(self.min_ampl_input)

        if self.min_ampl_input > max_ampl:
            raise ValidationError(
                message='Max amplitude has to be greater than min amplitude %.3f.' % (self.min_ampl_input),
                cursor_position=len(document.text))  # Move cursor to end of input.

class ElectrodeLabelValidator(Validator):

    def validate(self, document):

        if not len(document.text):
            raise ValidationError(
                message='Electrode label cannot be empty. Please enter a valid labels e.g. LPOG10' ,
                cursor_position=len(document.text))  # Move cursor to end of input.

        try:
            float(document.text)
            raise ValidationError(
                message='Electrode label cannot be a number. Please enter a valid labels e.g. LPOG10' ,
                cursor_position=len(document.text))  # Move cursor to end of input.
        except ValueError:
            pass

class YesNoValidator(Validator):
    def validate(self, document):
        if re.match(r'yes|y',document.text.lower()) is None or re.match(r'no|n',document.text.lower()) is None:
            raise ValidationError


def parse_command_line():
    """
    Parses command line using prompt_toolkit
    :return: Instance of Args class
    """
    args_obj = Args()

    args_obj.subject = prompt('Subject: ', default='R1250N')

    args_obj.experiment = prompt('Experiment: ', completer=experiment_completer, validator=ExperimentValidator(experiment_list),
                        default='PS4_PAL5')

    if sys.platform.startswith('win'):
        workspace_dir = prompt('Workspace directory: ', validator=DirValidator(), completer=path_completer,
                                     default='D:/scratch')
        mount_point = prompt('Mount Point (do not modify): ', validator=DirValidator(), completer=path_completer,
                             default='D:/')
    else:
        # workspace_dir = prompt('Workspace directory: ', validator=DirValidator(), completer=path_completer,
        #                              default='/scratch')
        workspace_dir = prompt('Workspace directory: ', completer=path_completer,
                                     default='/scratch/system3_configs/PAL5/')
        mount_point = prompt('Mount Point (do not modify): ', validator=DirValidator(), completer=path_completer,
                             default='/')

    args_obj.workspace_dir = workspace_dir
    args_obj.mount_point = mount_point

    args_obj.electrode_config_file = prompt('Electrode Configuration file (.csv): ', validator=CSVFileValidator(),
                                   # completer=path_completer, default='d:/experiment_configs/R1284N_FromJson.csv')
                                   completer=path_completer,)

    args_obj.pulse_frequency = prompt('Stimulation Frequency (Hz) -  DO NOT MODIFY ',
                            validator=TypedNumberValidator(int, 'integer'), default='200')


    for stim_pair_num in xrange(num_stim_pairs):

        anode = prompt('Anode label for stim_pair %d: ' % stim_pair_num, validator=ElectrodeLabelValidator())
        args_obj.anodes.append(anode)

        cathode = prompt('Cathode label for stim_pair %d: ' % stim_pair_num, validator=ElectrodeLabelValidator())
        args_obj.cathodes.append(cathode)

        min_ampl = prompt('Min stim amplitude (in mA ) for stim_pair %d: ' % stim_pair_num,
                          validator=AmplitudeValidator())
        args_obj.min_amplitudes.append(min_ampl)

        max_ampl = prompt('Max stim amplitude (in mA ) for stim_pair %d: ' % stim_pair_num,
                          validator=MaxAmplitudeValidator(min_ampl))

        args_obj.max_amplitudes.append(max_ampl)

    encoding = prompt('Use encoding classifier? (yes/no)',)
    args_obj.encoding = 'yes' in encoding.lower()

    return args_obj

if __name__ == '__main__':

    args = parse_command_line()





