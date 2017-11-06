from collections import namedtuple
import functools
import json
import os.path

from pkg_resources import resource_string, resource_filename
import pytest

from classiflib import ClassifierContainer

from ramutils.parameters import StimParameters, FilePaths
from ramutils.tasks.odin import save_montage_files, generate_ramulator_config
from ramutils.test import Mock
from ramutils.utils import touch


def jsondata(s):
    return json.loads(resource_string('ramutils.test.test_data', s))


@pytest.mark.parametrize('mkdir', [True, False])
def test_save_montage_files(mkdir, tmpdir):
    pairs = jsondata('R1328E_pairs.json')
    excluded = jsondata('R1328E_excluded_pairs.json')

    dest = str(tmpdir.join('otherdir')) if mkdir else str(tmpdir)
    save_montage_files(pairs, excluded, dest).compute()

    with open(os.path.join(dest, 'pairs.json'), 'r') as f:
        assert json.loads(f.read()) == pairs

    with open(os.path.join(dest, 'excluded_pairs.json'), 'r') as f:
        assert json.loads(f.read()) == excluded

    # FIXME
    # with pytest.raises(AssertionError):
    #     path = tmpdir.join('file')
    #     path.write_binary(b'')
    #     save_montage_files(pairs, excluded, str(path))


@pytest.mark.parametrize('experiment', ['FR6'])
def test_generate_ramulator_config(experiment, tmpdir):
    subject = 'R1354E'
    classifier_path = tmpdir.join(subject).join(experiment)\
        .join('config_files').join('{}-classifier.zip'.format(subject))
    container = Mock(ClassifierContainer, side_effect=lambda: touch(str(classifier_path)))

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
    ec_conf_path = getpath('R1354E_26OCT2017L0M0STIM.csv')
    paths = FilePaths(
        electrode_config_file=ec_conf_path,

        # Since we're not actually reading the pairs files in this test, we
        # don't have to worry about the fact that the subjects aren't the same.
        # All we are really doing in this test is verifying that stuff is saved.
        pairs=getpath('R1328E_pairs.json'),
        excluded_pairs=getpath('R1328E_excluded_pairs.json')
    )

    path = generate_ramulator_config(subject, experiment, container,
                                     stim_params, paths, str(tmpdir)).compute()

    # Ensure we saved a container
    container.save.assert_called_once()

    assert os.path.exists(path)
