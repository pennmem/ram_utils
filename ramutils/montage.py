""" Set of utility functions for working with electrode-related data """

import os
import json
import numpy as np
import pandas as pd

from collections import OrderedDict
from classiflib import dtypes
from ptsa.data.readers import JsonIndexReader
from bptools.util import standardize_label
from ramutils.utils import extract_subject_montage


def build_montage_metadata_table(subject, all_pairs, root='/'):
    """ Create a dataframe containing atlas labels, locations, and coordinates

    Parameters
    ----------
    subject: str
        Subject ID
    all_pairs: OrderedDict
        Full set of bipolar pairs that will be augmented with their metadata
    root: str
        Base path for RHINO

    """
    pairs_from_json = load_pairs_from_json(subject, rootdir=root)

    # Standardize labels from json so that lookup will be easier
    pairs_from_json = {standardize_label(key): val for key, val in pairs_from_json.items()}

    # If all_pairs is an ordered dict, so this loop will preserve the ordering
    all_pair_labels = all_pairs[subject]['pairs'].keys()
    for pair in all_pair_labels:
        standardized_pair = standardize_label(pair)
        if standardized_pair not in pairs_from_json:
            # Log some warning here about not finding the contact
            continue
        channel_1 = pairs_from_json[standardized_pair]['channel_1']
        channel_2 = pairs_from_json[standardized_pair]['channel_2']
        all_pairs[subject]['pairs'][pair]['channel_1'] = str(channel_1)
        all_pairs[subject]['pairs'][pair]['channel_2'] = str(channel_2)
        # types should be same for both electrodes
        all_pairs[subject]['pairs'][pair]['type'] = pairs_from_json[standardized_pair]['type_1']
        all_pairs[subject]['pairs'][pair]['location'] = extract_atlas_location(pairs_from_json[standardized_pair])
        all_pairs[subject]['pairs'][pair]['label'] = pair

    # Constructing the dataframe will not preserve the order from the OrderedDict
    pairs_metadata = pd.DataFrame.from_dict(all_pairs[subject]['pairs'], orient='index')
    pairs_metadata = pairs_metadata.reindex(all_pair_labels)
    pairs_metadata = pairs_metadata[['type', 'channel_1', 'channel_2', 'label', 'location']]

    return pairs_metadata


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
    if stim_params is None:
        stim_params = []

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


def compare_recorded_with_all_pairs(all_pairs, classifier_pairs):
    """ Returns a mask for if an electrode in all_pairs is present in
    classifier_pairs

    Parameters
    ----------
    all_pairs: OrderedDict
        The full set of possible pairs based on the electrode config
    classifier_pairs: np.recarray
        Pairs used for classification (usually extracted from classifier
        container)

    Returns
    -------
    array_like
        Boolean array of the same size as all_pairs indicating if each pair
        was used for classification

    """
    used_pairs = classifier_pairs[["contact0", "contact1"]]
    used_pairs = np.array([(int(a), int(b)) for a, b in used_pairs])

    recorded_pairs = []
    subject = all_pairs.keys()[0]
    for pair in all_pairs[subject]['pairs'].keys():
        channel_1 = all_pairs[subject]['pairs'][pair]['channel_1']
        channel_2 = all_pairs[subject]['pairs'][pair]['channel_2']
        pair_nums = (int(channel_1), int(channel_2))
        recorded_pairs.append(pair_nums)

    recorded_pairs = np.array(recorded_pairs)
    pair_mask = np.isin(recorded_pairs, used_pairs)
    pair_mask = np.apply_along_axis(max, 1, pair_mask)

    return pair_mask


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
    keys = list(pairs.keys())
    # Handle empty dictionary case
    if len(keys) == 0:
        return pairs

    # Remove 'version' information. TODO: Make this more flexible
    if 'version' in keys:
        keys.remove('version')
    subject = keys[0]
    pairs = pairs[subject]['pairs']

    return pairs


def load_pairs_from_json(subject, rootdir='/'):
    """ Load montage information from pairs.json file

    Parameters
    ----------
    subject: str
        Subject ID
    rootdir: str
        Mount point for RHINO

    Returns
    -------
    dict
        Dictionary containing metadata for all pairs in the given subjects' montage

    """
    subject_id, montage = extract_subject_montage(subject)

    json_reader = JsonIndexReader(os.path.join(rootdir,
                                               "protocols",
                                               "r1.json"))
    all_pairs_paths = json_reader.aggregate_values('pairs', subject=subject_id,
                                                   montage=montage)

    # For simplicity, just load the first file since they *should* all be the
    # same
    bp_path = os.path.join(rootdir, list(all_pairs_paths)[0])
    with open(bp_path, 'r') as f:
        pair_data = json.load(f)
    pair_data = extract_pairs_dict(pair_data)

    return pair_data


def extract_atlas_location(bp_data):
    """ Extract atlas based on electrode type and what locations are available

    Parameters
    ----------
    bp_data: dict
        Dictionary containing metadata for a single electrode (monopolar or bipolar)

    Returns
    -------
    str
        Atlas location for the given contact
    """
    atlases = bp_data['atlases']

    # Sort of a waterfall here: Stein, then WB for depths, then ind
    if 'stein' in atlases:
        loc_tag = atlases['stein']['region']
        if (loc_tag is not None) and (loc_tag!='') and (loc_tag!='None'):
            return loc_tag

    if (bp_data['type_1']=='D') and ('wb' in atlases):
        wb_loc = atlases['wb']['region']
        if (wb_loc is not None) and (wb_loc!='') and (wb_loc!='None'):
            return wb_loc

    if 'ind' in atlases:
        ind_loc = atlases['ind']['region']
        if (ind_loc is not None) and (ind_loc!='') and (ind_loc!='None'):
            return ('Left ' if atlases['ind']['x']<0.0 else 'Right ') + ind_loc

    return '--'
