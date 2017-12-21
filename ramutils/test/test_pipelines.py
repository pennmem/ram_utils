""" Blackbox functional tests for pipelines """
import os
import datetime
import functools
import pytest
from pkg_resources import resource_filename
from ramutils.parameters import FilePaths, FRParameters
from ramutils.tasks import memory
from ramutils.pipelines.report import make_report

getpath = functools.partial(resource_filename, 'ramutils.test.test_data')
memory.clear(warn=False)


@pytest.mark.slow
@pytest.mark.rhino
@pytest.mark.parametrize("subject, experiment, joint_report, sessions", [
    ('R1354E', 'FR1', False, [0]),
    ('R1354E', 'FR1', False, [0, 1]),
    ('R1354E', 'FR1', False, None),
    ('R1354E', 'catFR1', False, [0]),
    ('R1354E', 'FR1', True, None),
])
def test_make_fr1_report(subject, experiment, joint_report, sessions,
                         rhino_root, output_dest):
    params = FRParameters()
    rhino = os.path.expanduser('/Volumes/rhino')
    pairs_path = os.path.join(rhino, 'protocols', 'r1', 'subjects', subject,
                              'localizations', str(0), 'montages', str(0),
                              'neuroradiology', 'current_processed',
                              'pairs.json')

    paths = FilePaths(root=rhino_root,
                      electrode_config_file='/scratch/system3_configs/'
                                            'ODIN_configs/R1354E/'
                                            'R1354E_26OCT2017L0M0STIM.csv',
                      pairs=pairs_path,
                      dest=output_dest
                      )

    report = make_report(subject, experiment, paths,
                         joint_report=joint_report, stim_params=None,
                         exp_params=params, sessions=sessions)

    today = datetime.datetime.today().strftime('%Y_%m_%d')
    file_name = '_'.join([subject, experiment, today]) + ".html"
    assert os.path.exists(os.path.join(paths.dest, file_name))
    return
