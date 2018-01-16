from ._wrapper import task
from ramutils.montage import generate_pairs_for_classifier as \
    generate_pairs_for_classifier_core
from ramutils.montage import reduce_pairs as reduce_pairs_core
from ramutils.montage import get_used_pair_mask as get_used_pair_mask_core
from ramutils.montage import build_montage_metadata_table
from ramutils.montage import get_pairs as get_pairs_core

__all__ = [
    'generate_pairs_for_classifier',
    'reduce_pairs',
    'get_used_pair_mask',
    'generate_montage_metadata_table',
    'get_pairs',
]


@task()
def generate_pairs_for_classifier(pairs, excluded_pairs):
    return generate_pairs_for_classifier_core(pairs, excluded_pairs)


@task()
def reduce_pairs(pairs, stim_params, return_excluded=False):
    return reduce_pairs_core(pairs, stim_params,
                             return_excluded=return_excluded)


@task(cache=False)
def get_used_pair_mask(all_pairs, excluded_pairs):
    return get_used_pair_mask_core(all_pairs, excluded_pairs)


@task()
def generate_montage_metadata_table(subject, all_pairs, root):
    return build_montage_metadata_table(subject, all_pairs, root=root)


@task()
def get_pairs(subject, experiment, paths, localization=0, montage=0):
    return get_pairs_core(subject, experiment, paths, localization=localization,
                          montage=montage)
