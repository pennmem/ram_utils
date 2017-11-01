import os
import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from ramutils.tasks import task


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
def load_pairs(index, subject):
    """Load pairs.json.

    FIXME: this takes a much longer time to run than one might expect, probably
    because of the json reader.

    :param JsonIndexReader index:
    :rtype: dict

    """
    tmp = subject.split('_')
    subj_code = tmp[0]
    montage = 0 if len(tmp) == 1 else int(tmp[1])

    bp_paths = index.aggregate_values('pairs', subject=subj_code, montage=montage)
    mount_point = index.index_file.split('/protocols')[0]
    path = os.path.join(mount_point, next(iter(bp_paths)))

    with open(path, 'r') as f:
        bipolar_dict = json.load(f)[subject]['pairs']

    return bipolar_dict


@task()
def remove_stim_pairs(pairs):
    """Remove stim pairs from the pairs.json dict.

    :param dict pairs: Full pairs.json as a dict
    :returns: Dictionary of reduced pairs

    """
    # FIXME
    # if self.pipeline.args.anode_nums:
    #     stim_pairs = self.pipeline.args.anode_nums + self.pipeline.args.cathode_nums
    # else:
    #     stim_pairs = [self.pipeline.args.anode_num, self.pipeline.args.cathode_num]

    reduced_pairs = [bp for bp in pairs if int(bp[0]) not in stim_pairs and int(bp[1]) not in stim_pairs]

    # FIXME for modern Python
    sorted_ = sorted(
        pairs.items(),
        cmp=lambda x, y: cmp((x[1]['channel_1'], x[1]['channel_2']),
                             (y[1]['channel_1'], y[1]['channel_2'])))

    reduced_dict = OrderedDict()
    for (bp_tag, bp_dict) in sorted_:
        if ('%03d' % bp_dict['channel_1'], '%03d' % bp_dict['channel_2']) not in reduced_pairs:
            reduced_dict[bp_tag] = bp_dict
    return reduced_dict


@task()
def build_tal_structs(pairs_dict, anodes, cathodes):
    """Build tal structs (???)

    :param dict pairs_dict: Loaded contents from pairs.json
    :param list anodes: List of stim anode labels
    :param list cathodes: List of stim cathode labels
    :returns: tal struct data
    :rtype: pd.DataFrame

    """
    bipolar_data_stim_only = {
        bp_tag: bp_data for bp_tag, bp_data in pairs_dict.items()
        if bp_data['is_stim_only']
    }

    bipolar_data = {
        bp_tag: bp_data for bp_tag, bp_data in pairs_dict.items()
        if not bp_data['is_stim_only']
    }

    if anodes is not None:
        # FIXME: what is anode/cathode_nums?
        (args.anode_nums, args.cathode_nums) = zip(
            *[(bipolar_data['-'.join((anode.upper(), cathode.upper()))]['channel_1'],
               bipolar_data['-'.join((anode.upper(), cathode.upper()))]['channel_2'])
              for (anode, cathode) in zip(anodes, cathodes)])

    bp_tags = []
    bp_tal_structs = []
    for bp_tag, bp_data in bipolar_data.items():
        bp_tags.append(bp_tag)
        ch1 = bp_data['channel_1']
        ch2 = bp_data['channel_2']
        bp_tal_structs.append(['%03d' % ch1, '%03d' % ch2, bp_data['type_1'], _atlas_location(bp_data)])

    bp_tal_structs = pd.DataFrame(bp_tal_structs, index=bp_tags, columns=['channel_1', 'channel_2', 'etype', 'bp_atlas_loc'])
    bp_tal_structs.sort_values(by=['channel_1', 'channel_2'], inplace=True)


# FIXME
# @task()
# def build_stim_only_tal_structs():
#     """Build stim-only tal structs (???)
#
#     :param str subject: Subject ID
#     :param dict pairs_dict: Loaded contents from pairs.json
#     :param dict excluded: Excluded pairs
#     :param list anodes: List of stim anode labels
#     :param list cathodes: List of stim cathode labels
#     :rtype: pd.Series
#
#     """
#     bp_tal_stim_only_structs = pd.Series()
#
#     if bipolar_data_stim_only:
#         bp_tags_stim_only = []
#         bp_tal_stim_only_structs = []
#         for bp_tag,bp_data in bipolar_data_stim_only.items():
#             bp_tags_stim_only.append(bp_tag)
#             bp_tal_stim_only_structs.append(_atlas_location(bp_data))
#         bp_tal_stim_only_structs = pd.Series(bp_tal_stim_only_structs, index=bp_tags_stim_only)
#
#     return bp_tal_stim_only_structs


@task()
def get_monopolar_channels(bp_tal_structs):
    """Get all monopolar channels.

    :param pd.DataFrame bp_tal_structs:
    :returns: Array of monopolar channels
    :rtype: np.ndarray

    """
    return np.unique(np.hstack((bp_tal_structs.channel_1.values, bp_tal_structs.channel_2.values)))


@task()
def get_bipolar_pairs(bp_tal_structs):
    """Get all bipolar pairs.

    :param pd.DataFrame bp_tal_structs:
    :rtype: list

    """
    return list(zip(bp_tal_structs.channel_1.values, bp_tal_structs.channel_2.values))


if __name__ == "__main__":
    from ramutils.tasks import read_index

    index = read_index('~/mnt/rhino')
    pairs = load_pairs(index, 'R1354E')
    reduced = remove_stim_pairs(pairs)
    tal_structs = build_tal_structs(pairs, ['1Ld9', '5Ld7'], ['1Ld10', '5Ld8'])
    monopolars = get_monopolar_channels(tal_structs)

    monopolars.visualize()
