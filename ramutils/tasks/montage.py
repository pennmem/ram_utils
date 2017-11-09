import os
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from ramutils.tasks import task

__all__ = ['load_pairs', 'reduce_pairs']


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
def load_pairs(path):
    """Load pairs.json.

    :param str path: Path to pairs.json
    :returns: pairs.json as a dict
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
    subject = list(pairs.keys())[0]
    contacts = [(p.anode, p.cathode) for p in stim_params]
    all_pairs = pairs[subject]['pairs']
    reduced_pairs = OrderedDict()
    excluded_pairs = OrderedDict()

    for label, pair in all_pairs.items():
        if pair['channel_1'] not in contacts and pair['channel_2'] not in contacts:
            reduced_pairs[label] = pair
        else:
            excluded_pairs[label] = pair

    if return_excluded:
        return excluded_pairs
    else:
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
