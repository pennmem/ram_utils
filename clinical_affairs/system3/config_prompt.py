"""Functions for prompting common experiment configuration options."""

from __future__ import unicode_literals, print_function

import os.path as osp
from functools import partial

from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.token import Token
from prompt_toolkit.contrib.completers import WordCompleter
from prompt_toolkit import prompt


def tbmsg(msg):
    """Used for displaying a message in the bottom toolbar."""
    return [(Token.Toolbar, msg)]


class StimPairParams(object):
    """Return type for stim pair settings."""
    anode = 1
    cathode = 2

    # Variable-amplitude experiments
    min_amplitude = 0.1
    max_amplitude = 1.0

    # Fixed-amplitude experiments
    stim_amplitude = 0.1


class ListValidator(Validator):
    """Validate input based on a list of possibilities."""
    def __init__(self, valid_items):
        self.valid_items = [item.strip() for item in valid_items]

    def validate(self, document):
        text = document.text.strip()
        if text not in self.valid_items:
            raise ValidationError(message="Valid choices: " + ', '.join([s for s in self.valid_items]),
                                  cursor_position=len(text))


class YesNoValidator(Validator):
    """Validate a response is yes or no (and variations thereof)."""
    def validate(self, document):
        text = document.text.lower()
        if text not in ['y', 'n', 'yes', 'no']:
            raise ValidationError(message='Answer yes or no', cursor_position=len(text))


class PathValidator(Validator):
    """Validate a path.

    :param bool isfile: Allow file paths.
    :param bool isdir: Allow directory paths.

    """
    def __init__(self, isfile=True, isdir=True):
        self.isfile = isfile
        self.isdir = isdir
        assert self.isfile or self.isdir, "at least one of isfile or isdir must be true"

    def validate(self, document):
        path = document.text
        found = osp.exists(path)

        if found and self.isfile:
            if not osp.isfile(path):
                raise ValidationError(message="path {} is not a file".format(path))

        elif found and self.isdir:
            if not osp.isdir(path):
                raise ValidationError(message="path {} is not a directory".format(path))

        else:
            raise ValidationError(message="path {} does not exist".format(path))

def get_subject():
    """Prompt for the subject ID."""
    return prompt("Subject ID: ")


def get_yes_or_no(msg):
    """Prompt for a yes or no response.

    :param str msg: Message to prompt with.

    """
    while True:
        try:
            resp = prompt(msg, validator=YesNoValidator()).lower()
            return resp[0] == 'y'
        except ValidationError as e:
            print(e.message)


def get_path(msg, isfile=True, isdir=True):
    """Prompt for a path.

    :param str msg: Prompt message.
    :param bool isfile: Allow file paths.
    :param bool isdir: Allow directory paths.

    """
    while True:
        try:
            return prompt(msg, validator=PathValidator(isfile, isdir))
        except ValidationError as e:
            print(e.message)


def get_experiment(allowed_experiments):
    """Get the experiment to generate the config for.

    :param list allowed_experiments:

    """
    validator = ListValidator(allowed_experiments)
    completer = WordCompleter(allowed_experiments)
    msg = 'Tab to see options'
    while True:
        try:
            return prompt("Experiment: ", validator=validator, completer=completer,
                          get_bottom_toolbar_tokens=lambda cli: tbmsg(msg))
        except ValidationError as e:
            msg = e.message


def get_stim_pair(experiment, pair_index=0):
    """Get stim pair parameters.

    :param str experiment: Experiment we're configuring for.
    :param int pair_index: Zero-indexed stim pair number.
    :rtype: StimPairParams

    """
    msg = 'Stim pair #{} configuration'.format(pair_index + 1)
    pprompt = partial(prompt, get_bottom_toolbar_tokens=lambda cli: tbmsg(msg))
    result = StimPairParams()

    result.anode = int(pprompt('Anode: '))
    result.cathode = int(pprompt('Cathode: '))

    if 'PS4' in experiment:
        result.min_amplitude = float(pprompt('Minimum amplitude [mA]: '))
        result.max_amplitude = float(pprompt('Maximum amplitude [mA]: '))
    else:
        result.stim_amplitude = float(pprompt('Target amplitude [mA]: '))

    return result


if __name__ == "__main__":
    # get_subject()
    # get_experiment(['FR6', 'CatFR6'])
    # get_stim_pair('PS4_FR5')
    get_yes_or_no("Use retrieval data? (y/n) ")
