""" Blackbox functional tests for pipelines """
import os
import functools
import pytest
from pkg_resources import resource_filename
from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.report import make_report

getpath = functools.partial(resource_filename, 'ramutils.test.test_data')

subject = 'R1354E'


@pytest.mark.slow
@pytest.mark.rhino
@pytest.mark.parametrize("subject, experiment, joint_report, sessions", [
    ('R1354E', 'FR1', False, [0]),
    ('R1354E', 'FR1', False, [0,1]),
    ('R1354E', 'FR1', False, None),
    ('R1354E', 'catFR1', False, [0]),
    ('R1354E', 'FR1', True, None),
])
def test_make_fr1_report(subject, experiment, joint_report, sessions):
    params = FRParameters()
    rhino = os.path.expanduser('/Volumes/rhino')
    pairs_path = os.path.join(rhino, 'protocols', 'r1', 'subjects', subject,
                              'localizations', str(0), 'montages', str(0),
                              'neuroradiology', 'current_processed',
                              'pairs.json')

    paths = FilePaths(root=rhino,
                      electrode_config_file='/scratch/system3_configs/'
                                            'ODIN_configs/R1354E/'
                                            'R1354E_26OCT2017L0M0STIM.csv',
                      pairs=pairs_path,
                      dest='scratch/zduey/sample_fr1_reports'
                      )
    report = make_report(subject, experiment, paths,
                         joint_report=joint_report, stim_params=None,
                         exp_params=params, sessions=sessions)

    assert os.path.exists(os.path.join(os.paths.dest, subject, 'report.html'))
    return
