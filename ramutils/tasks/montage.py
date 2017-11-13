import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from classiflib import dtypes
from ramutils.tasks import task

__all__ = ['load_pairs_from_json', 'reduce_pairs',
           'generate_pairs_for_classifier', 'get_used_pair_mask',
           'extract_pairs_dict']


# FIXME: document
def _atlas_location(bp_data):
    atlases = bp_data['atlases']

    if 'stein' in atlases:
        loc_tag = atlases['stein']['region']
        if (loc_tag is not None) and (loc_tag != '') and (loc_tag != 'None'):
            return loc_tag

    if (bp_data['type_1'] == 'D') and ('wb' in atlases):
        wb_loc = atlases['wb']['region']
        if (wb_loc is not None) and (wb_loc != '') and (wb_loc != 'None'):
            return wb_loc

    if 'ind' in atlases:
        ind_loc = atlases['ind']['region']
        if (ind_loc is not None) and (ind_loc != '') and (ind_loc != 'None'):
            return ('Left ' if atlases['ind']['x'] < 0.0 else 'Right ') + ind_loc

    return '--'


@task()
def load_pairs_from_json(path):
    """Load pairs.json.

    :param str path: Path to pairs.json
    :returns: pairs.json as a dict of bipolar pairs
    :rtype: dict

    """
    with open(path, 'r') as f:
        pairs = json.load(f)

    return pairs


@task(cache=False)
def reduce_pairs(pairs, stim_params, return_excluded=False):
    """Remove stim pairs from the pairs.json dict.

    FIXME: do these need to be sorted by channel_1?

    :param dict pairs: Full pairs.json as a dict
    :param List[StimParameters] stim_params:
    :param bool return_excluded:
    :returns: excluded pairs if not return_excluded else reduced pairs
    :rtype: dict

    """
    pairs = extract_pairs_dict(pairs)
    contacts = [(p.anode, p.cathode) for p in stim_params]
    reduced_pairs = OrderedDict()
    excluded_pairs = OrderedDict()

    for label, pair in pairs.items():
        if (pair['channel_1'], pair['channel_2']) not in contacts:
            reduced_pairs[label] = pair
        else:
            excluded_pairs[label] = pair

    if return_excluded:
        reduced_pairs = excluded_pairs
    return reduced_pairs


@task(cache=False)
def get_monopolar_channels(bp_tal_structs):
    """Get all monopolar channels.

    :param pd.DataFrame bp_tal_structs:
    :returns: Array of monopolar channels
    :rtype: np.ndarray

    """
    return np.unique(np.hstack((bp_tal_structs.channel_1.values, bp_tal_structs.channel_2.values)))


@task(cache=False)
def get_bipolar_pairs(bp_tal_structs):
    """Get all bipolar pairs.

    :param pd.DataFrame bp_tal_structs:
    :rtype: list

    """
    return list(zip(bp_tal_structs.channel_1.values, bp_tal_structs.channel_2.values))


@task()
def generate_pairs_for_classifier(pairs, excluded_pairs):
    """ Create recarray of electrode pairs for the classifier container

    :param pairs: JSON-format object containing all electrode pairs in the
    montage
    :param excluded_pairs: array-like containing pairs excluded from the montage
    :returns: recarray containing all pairs minus excluded pairs
    :rtype: np.recarray

    """
    pairs = extract_pairs_dict(pairs)
    used_pairs = {
        key: value for key, value in pairs.items()
        if key not in excluded_pairs
        }

    pairs = np.rec.fromrecords([(item['channel_1'], item['channel_2'],
                                 pair.split('-')[0], pair.split('-')[1])
                                 for pair, item in used_pairs.items()],
                               dtype=dtypes.pairs)

    pairs.sort(order='contact0')

    return pairs


@task()
def get_used_pair_mask(all_pairs, excluded_pairs):
    """ Get a mask of whether to include a pair in all_pairs based on
    excluded pairs

    :param all_pairs: standard json-format pairs object
    :param excluded_pairs: list containing pairs to exclude
    :return: np.array containing a mask to identify excluded pairs from all
    pairs

    """
    all_pairs = extract_pairs_dict(all_pairs).keys()
    mask = [False if (label in excluded_pairs) else True for
            label in all_pairs]
    mask = np.array(mask)

    return mask


def extract_pairs_dict(pairs):
    """ Extract a dictionary of pairs from the standard json structure

    :param pairs: raw json pairs structure
    :return:  dict containing just the pairs data

    """
    # Handle empty dictionary case
    if len(pairs.keys()) == 0:
        return pairs

    subject = list(pairs.keys())[0]
    pairs = pairs[subject]['pairs']

    return pairs
