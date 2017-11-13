from __future__ import print_function

import functools
import os.path
from pkg_resources import resource_filename

from ptsa.data.readers import JsonIndexReader

from ramutils.parameters import FilePaths, StimParameters, FRParameters
# TODO: Don't use * for imports
from ramutils.tasks.events import *
from ramutils.tasks.montage import *
from ramutils.tasks.classifier import *
from ramutils.tasks.odin import *
from ramutils.tasks.powers import *

getpath = functools.partial(resource_filename, 'ramutils.test.test_data')

subject = 'R1354E'
rhino = os.path.expanduser('/Volumes/rhino')
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


### Pipeline ###

#TODO: Add remove_bad_events() and remove_negative_offsets() functions to
# process events
fr_events = read_fr_events(jr, subject, cat=False)
catfr_events = read_fr_events(jr, subject, cat=True)
raw_events = concatenate_events(fr_events, catfr_events)
all_events = create_baseline_events(raw_events, 1000, 29000)
word_events = select_word_events(all_events, include_retrieval=True)
encoding_events = select_encoding_events(word_events)
retrieval_events = select_retrieval_events(word_events)

ec_pairs = generate_pairs_from_electrode_config(subject, paths)
excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)


# FIXME: If PTSA is updated to not remove events behind this scenes, this
# won't be necessary. Or, if we can remove bad events before passing to
# compute powers, then we won't have to catch the events
encoding_powers, good_encoding_events = compute_powers(encoding_events,
                                                       params)

retrieval_powers, good_retrieval_events = compute_powers(retrieval_events,
                                                         params)

normalized_encoding_powers = normalize_powers_by_session(encoding_powers,
                                                         good_encoding_events)

normalized_retrieval_powers = normalize_powers_by_session(retrieval_powers,
                                                          good_retrieval_events)

task_events = combine_events([good_encoding_events, good_retrieval_events])
powers = combine_encoding_retrieval_powers(task_events,
                                           normalized_encoding_powers,
                                           normalized_retrieval_powers)
reduced_powers = reduce_powers(powers, used_pair_mask, len(params.freqs))

sample_weights = get_sample_weights(task_events, params)
classifier = train_classifier(powers, task_events, sample_weights, params)
cross_validation_results = perform_cross_validation(classifier, reduced_powers,
                                                    task_events, params)

container = serialize_classifier(classifier, final_pairs, reduced_powers,
                                 task_events, sample_weights,
                                 cross_validation_results,
                                 subject)

config_path = generate_ramulator_config(subject, 'FR6', container, stim_params,
                                        paths, ec_pairs, excluded_pairs)

# config_path.visualize()
config_path.compute()
