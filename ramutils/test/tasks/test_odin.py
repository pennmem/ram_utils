from __future__ import print_function

from collections import namedtuple
import functools
import json
import os.path
from zipfile import ZipFile

from pkg_resources import resource_string, resource_filename
import pytest

from classiflib import ClassifierContainer

from ramutils.parameters import StimParameters, FilePaths
from ramutils.tasks.odin import save_montage_files, generate_ramulator_config
from ramutils.test import Mock, patch
from ramutils.utils import touch


def jsondata(s):
    return json.loads(resource_string('ramutils.test.test_data', s))


@pytest.mark.parametrize('experiment', ['FR6'])
def test_generate_ramulator_config(experiment, tmpdir):
    subject = 'R1354E'
    classifier_path = str(tmpdir.join(subject).join(experiment)
        .join('config_files').join('{}-classifier.zip'.format(subject)))

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

    getpath = functools.partial(resource_filename, 'ramutils.test.test_data')
    ec_conf_prefix = 'R1354E_26OCT2017L0M0STIM'
    ec_conf_path = getpath(ec_conf_prefix + '.csv')
    paths = FilePaths(
        electrode_config_file=ec_conf_path,

        # Since we're not actually reading the pairs files in this test, we
        # don't have to worry about the fact that the subjects aren't the same.
        # All we are really doing in this test is verifying that stuff is saved.
        pairs=getpath('R1328E_pairs.json'),
        excluded_pairs=getpath('R1328E_excluded_pairs.json')
    )

    with patch.object(container, 'save', side_effect=lambda *args, **kwargs: touch(classifier_path)):
        path = generate_ramulator_config(subject, experiment, container,
                                         stim_params, paths, str(tmpdir)).compute()

    with ZipFile(path) as zf:
        members = zf.namelist()

    assert 'experiment_config.json' in members
    assert 'config_files/pairs.json' in members
    assert 'config_files/excluded_pairs.json' in members
    assert 'config_files/' + ec_conf_prefix + '.csv' in members
    assert 'config_files/' + ec_conf_prefix + '.bin' in members
    assert 'config_files/{}-classifier.zip'.format(subject) in members
