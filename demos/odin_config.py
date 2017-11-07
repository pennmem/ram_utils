from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from dask import delayed

from ramutils.parameters import FilePaths, StimParameters
from ramutils.tasks.montage import *
from ramutils.tasks.odin import *
from ramutils.test import Mock

getpath = functools.partial(resource_filename, 'ramutils.test.test_data')

subject = 'R1354E'
rhino = os.path.expanduser('~/mnt/rhino')
pairs_path = os.path.join(
    rhino, 'protocols', 'r1', 'subjects', subject,
    'localizations', str(0),
    'montages', str(0),
    'neuroradiology', 'current_processed', 'pairs.json')

paths = FilePaths(
    root=os.path.expanduser('~/tmp'),
    electrode_config_file=getpath('R1354E_26OCT2017L0M0STIM.csv'),
    pairs=pairs_path,
    dest='output',
)

stim_params = [
    StimParameters(
        label='1Ld9-1Ld10',
        anode=9,
        cathode=10
    ),
    StimParameters(
        label='5Ld7-5Ld8',
        anode=27,
        cathode=28
    )
]


@delayed
def consume(reduced, excluded):
    print(reduced, excluded)


### Pipeline

pairs = load_pairs(pairs_path)
# reduced_pairs = reduce_pairs(pairs, stim_params, False)
excluded_pairs = reduce_pairs(pairs, stim_params, True)

ec_pairs = generate_pairs_from_electrode_config('R1354E', paths)
# FIXME: needs classifier container
config_path = generate_ramulator_config(subject, 'FR6', Mock(), stim_params,
                                        paths, ec_pairs, excluded_pairs)

config_path.visualize()
config_path.compute()
