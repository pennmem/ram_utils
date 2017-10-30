"""Main script for generating Ramulator configuration files."""

# FIXME: generalize this for all experiments (only FR6 now)
# FIXME: handle command-line options and don't prompt for passed options

from __future__ import unicode_literals, print_function

import os
import os.path as osp
from subprocess import Popen

from .prompts import *


def main():
    # FIXME for reorganization
    repo_root = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            '..',  # ramutils package
            '..',  # ramutils repo root
        )
    )
    cwd = osp.join(repo_root, 'tests', 'fr5_biomarker', 'system3')
    python_path = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = b':'.join([repo_root, python_path])

    subject = get_subject()
    experiment = get_experiment(['FR6', 'CatFR6'])
    use_retrieval = get_yes_or_no("Use retrieval data? (y/n) ")
    odin_config_filename = get_path("Odin electrode configuration file: ")

    stim_pairs = []
    for pair in range(2):
        stim_pairs.append(get_stim_pair(experiment, pair))

    cmd = [
        "python", "fr5_util_system_3.py",
        "--subject={}".format(subject),
        "--experiment={}".format(experiment),
        "--electrode-config-file={}".format(odin_config_filename),
        "--pulse-frequency=200",
    ]

    cmd += ["--anodes"] + [pair.anode for pair in stim_pairs]
    cmd += ["--cathodes"] + [pair.cathode for pair in stim_pairs]
    cmd += ["--target-amplitude"] + [str(pair.stim_amplitude) for pair in stim_pairs]
    cmd += ["--min-amplitudes"] + [str(pair.min_amplitude) for pair in stim_pairs]
    cmd += ["--max-amplitudes"] + [str(pair.max_amplitude) for pair in stim_pairs]

    if not use_retrieval:
        cmd += ["--encoding-only"]

    print(' '.join(cmd))

    Popen(cmd, cwd=cwd, env=os.environ).wait()
