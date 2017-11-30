""" Set of utility functions for working with electrode-related data """

import numpy as np

from collections import OrderedDict
from classiflib import dtypes


def generate_pairs_for_classifier(pairs, excluded_pairs):
    """Create recarray of electrode pairs for the classifier container

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


def reduce_pairs(pairs, stim_params, return_excluded=False):
    """Remove stim pairs from the pairs.json dict.

    :param dict pairs: Full pairs.json as a dict
    :param List[StimParameters] stim_params:
    :param bool return_excluded:
    :returns: excluded pairs if not return_excluded else reduced pairs
    :rtype: dict

    Parameters
    ----------
    pairs: OrderedDict
    stim_params:
    return_excluded:  bool
        Whether excluded pairs should be returned instead of reduced pairs

    Returns
    -------
    OrderedDict
        pairs with stim pairs removed, or removed pairs if return_excluded is True

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


def get_used_pair_mask(all_pairs, excluded_pairs):
    """ Create a dictionary mapping electrode names to a boolean for if they
    should be included or not in classifier training/evaluation

    Parameters
    ----------
    all_pairs: OrderedDict
    excluded_pairs: OrderedDict

    Returns
    -------
    dict
        Mapping between pair names in all_pairs and a boolean to
        identify if the contact should be excluded

    """
    extracted_pairs = extract_pairs_dict(all_pairs)
    if type(extracted_pairs) != OrderedDict:
        raise RuntimeError("all pairs must be an orderd dict so that the "
                           "ordering can be correctly preserved when creating "
                           "the mask")

    pair_list = extracted_pairs.keys()
    mask = [False if (label in excluded_pairs) else True for
            label in pair_list]

    return mask


def extract_pairs_dict(pairs):
    """ Extract a dictionary of pairs from the standard json structure

    Parameters
    ----------
    pairs: OrderedDict

    Returns
    -------
    OrderedDict
        Dictionary of pairs that will preserve ordering

    """
    # Handle empty dictionary case
    if len(pairs.keys()) == 0:
        return pairs

    subject = list(pairs.keys())[0]
    pairs = pairs[subject]['pairs']

    return pairs
