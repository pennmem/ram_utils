from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ramutils.parameters import FilePaths, FRParameters
from ramutils.pipelines.ramulator_config import make_ramulator_config
from ramutils.montage import make_stim_params

from ramutils.tasks import memory

memory.cachedir = "/Users/zduey/tmp/"


getpath = functools.partial(resource_filename, 'ramutils.test.test_data')


subject = 'R1401J'
paths = FilePaths(
    root='/Volumes/RHINO/',
    dest='scratch/zduey/'
)

params = FRParameters()
stim_params = make_stim_params(subject, ['STG3'], ['STG4'],
                               target_amplitudes=[0.5],
                               root=paths.root)

make_ramulator_config(subject, "CatFR5", paths, stim_params, exp_params=params, use_classifier_excluded_leads=False,
                      default_surface_area=5.024)

