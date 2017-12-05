from __future__ import print_function

from collections import namedtuple
import functools
import json
import os.path as osp
from zipfile import ZipFile

from pkg_resources import resource_string, resource_filename

import pytest

from classiflib import ClassifierContainer

from ramutils.parameters import StimParameters, FilePaths, FRParameters
from ramutils.tasks.odin import generate_ramulator_config, generate_pairs_from_electrode_config
from ramutils.test import Mock, patch
import ramutils.test.test_data
from ramutils.utils import touch


datafile = functools.partial(resource_filename, 'ramutils.test.test_data')


def jsondata(s):
    return json.loads(resource_string('ramutils.test.test_data', s))


def test_generate_pairs_from_electrode_config():
    paths = FilePaths(root=datafile(''),
                      electrode_config_file='/input/configs/R1354E_26OCT2017L0M0STIM.csv',
                      pairs='/input/montage/R1354E_pairs.json')
    config_pairs = generate_pairs_from_electrode_config('R1354E', paths).compute()
    assert config_pairs['R1354E']['pairs'].keys() > 0
    with open(datafile('/input/configs/R1354E_pairs_from_ec.json'), 'w') as f:
        json_pairs = json.dumps(config_pairs)
        f.write(json_pairs)
    return


@pytest.mark.parametrize('experiment', ['AmplitudeDetermination', 'FR6'])
def test_generate_ramulator_config(experiment):
    subject = 'R1354E'

    root = osp.join(osp.dirname(ramutils.test.test_data.__file__))

    classifier_path = osp.join(root, 'output', subject, experiment,
                               'config_files',
                               '{}-classifier.zip'.format(subject))

    container = Mock(ClassifierContainer)

    Pairs = namedtuple('Pairs', 'label,anode,cathode')
    pairs = [Pairs('1Ld1-1Ld2', 1, 2), Pairs('1Ld3-1Ld4', 3, 4)]

    stim_params = [
        StimParameters(
            label=pair.label,
            anode=pair.anode,
            cathode=pair.cathode
        )
        for pair in pairs
    ]

    ec_conf_prefix = 'R1354E_26OCT2017L0M0STIM'
    paths = FilePaths(
        root=root,
        electrode_config_file='/input/configs/R1354E_26OCT2017L0M0STIM' +
                              '.csv',

        # Since we're not actually reading the pairs files in this test, we
        # don't have to worry about the fact that the subjects aren't the same.
        # All we are really doing in this test is verifying that stuff is saved.
        pairs='/input/configs/R1328E_pairs.json',
        dest='output'
    )

    getpath = functools.partial(resource_filename, 'ramutils.test.test_data')
    with open(getpath('/input/configs/R1328E_excluded_pairs.json'), 'r') as f:
        excluded_pairs = json.load(f)

    if "FR" in experiment:
        exp_params = FRParameters()
    elif experiment == "AmplitudeDetermination":
        exp_params = None
    else:
        raise RuntimeError("invalid experiment")

    with patch.object(container, 'save', side_effect=lambda *args, **kwargs: touch(classifier_path)):
        path = generate_ramulator_config(subject, experiment, container,
                                         stim_params, paths,
                                         excluded_pairs=excluded_pairs,
                                         params=exp_params).compute()

    with ZipFile(path) as zf:
        members = zf.namelist()

    assert 'experiment_config.json' in members
    if experiment != 'AmplitudeDetermination':
        assert 'exp_params.h5' in members
        assert 'config_files/{}-classifier.zip'.format(subject) in members
    assert 'config_files/pairs.json' in members
    assert 'config_files/excluded_pairs.json' in members
    assert 'config_files/' + ec_conf_prefix + '.csv' in members
    assert 'config_files/' + ec_conf_prefix + '.bin' in members
