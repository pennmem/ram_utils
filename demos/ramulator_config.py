from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ptsa.data.readers import JsonIndexReader

from ramutils.parameters import FilePaths, StimParameters, FRParameters
from ramutils.tasks.events import *
from ramutils.tasks.montage import *
from ramutils.tasks.classifier import *
from ramutils.tasks.odin import *
from ramutils.tasks.powers import *

getpath = functools.partial(resource_filename, 'ramutils.test.test_data')

subject = 'R1354E'
rhino = os.path.expanduser('~/mnt/rhino')
jr = JsonIndexReader(os.path.join(rhino, "protocols", "r1.json"))
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

params = FRParameters()


### Pipeline

fr_events = read_fr_events(jr, subject, cat=False)
catfr_events = read_fr_events(jr, subject, cat=True)
events = concatenate_events(fr_events, catfr_events)

pairs = load_pairs(pairs_path)
# reduced_pairs = reduce_pairs(pairs, stim_params, False)
excluded_pairs = reduce_pairs(pairs, stim_params, True)

ec_pairs = generate_pairs_from_electrode_config(subject, paths)

powers = compute_powers(events, params)
classifier, xval, sample_weights = compute_classifier(events, powers, params, paths)
container = serialize_classifier(classifier, pairs, powers, events, sample_weights, xval, subject)

config_path = generate_ramulator_config(subject, 'FR6', container, stim_params,
                                        paths, ec_pairs, excluded_pairs)

config_path.visualize()
config_path.compute()
