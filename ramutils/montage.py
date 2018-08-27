""" Set of utility functions for working with electrode-related data """

from collections import OrderedDict
import json
import os
import os.path
from glob import glob
from os import path as osp

import h5py
import numpy as np
import pandas as pd

from bptools.jacksheet import read_jacksheet
from bptools.transform import SeriesTransformation
from bptools.util import standardize_label
from classiflib import dtypes
from ptsa.data.readers import JsonIndexReader

from ramutils.parameters import StimParameters
from ramutils.utils import extract_subject_montage, touch, bytes_to_str, tempdir
from ramutils.log import get_logger
from ramutils.constants import MTL_LOC_DICT, DK_LOC_DICT

logger = get_logger()


def make_stim_params(subject, anodes, cathodes, min_amplitudes=None,
                     max_amplitudes=None, target_amplitudes=None,  root='/'):
    """Construct :class:`StimParameters` objects from anode and cathode labels
    for a specific subject.

    Parameters
    ----------
    subject : str
    anodes : List[str]
        anode labels
    cathodes : List[str]
        cathode labels
    min_amplitudes : List[float]
        Minimum stim amplitudes (when applicable)
    max_amplitudes : List[float]
        Maximum stim amplitudes (when applicable)
    target_amplitudes : List[float]
        Target stim amplitudes (when applicable)
    root : str
        root directory to search for jacksheet

    Returns
    -------
    stim_params : List[StimParameters]

    """
    path = os.path.join(root, 'data', 'eeg', subject, 'docs', 'jacksheet.txt')
    jacksheet = read_jacksheet(path)

    stim_params = []

    valid_labels = jacksheet.label.values
    for i in range(len(anodes)):
        anode = anodes[i]
        cathode = cathodes[i]

        if anode not in valid_labels:
            raise RuntimeError(
                "Label {} could not be found in the jacksheet".format(anode))

        if cathode not in valid_labels:
            raise RuntimeError(
                "Label {} could not be found in the jacksheet".format(cathode))

        anode_idx = jacksheet[jacksheet.label == anode].index[0]
        cathode_idx = jacksheet[jacksheet.label == cathode].index[0]

        params = StimParameters(
            anode_label=anode,
            cathode_label=cathode,
            anode=anode_idx,
            cathode=cathode_idx
        )

        if min_amplitudes is not None:
            params.min_amplitude = min_amplitudes[i]
            params.max_amplitude = max_amplitudes[i]
        else:
            params.target_amplitude = target_amplitudes[i]

        stim_params.append(params)

    return stim_params


def build_montage_metadata_table(subject, experiment, sessions, all_pairs,
                                 root='/'):
    """ Create a dataframe containing atlas labels, locations, and coordinates

    Parameters
    ----------
    subject: str
        Subject ID
    experiment: str
        Experiment
    all_pairs: OrderedDict
        Full set of bipolar pairs that will be augmented with their metadata
    root: str
        Base path for RHINO

    """
    pairs_from_json = load_pairs_from_json(subject, experiment,
                                           sessions, rootdir=root)

    # Check if mni coordinates are present, since this is the case only since
    # neurorad v2.0
    first_channel = list(pairs_from_json.keys())[0]
    mni_available = ('mni' in pairs_from_json[first_channel]['atlases'].keys())

    # Standardize labels from json so that lookup will be easier
    pairs_from_json = {standardize_label(
        key): val for key, val in pairs_from_json.items()}

    # If all_pairs is an ordered dict, this loop will preserve the ordering
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
        all_pairs[subject]['pairs'][pair]['location'] = extract_atlas_location(
            pairs_from_json[standardized_pair])
        all_pairs[subject]['pairs'][pair]['label'] = pair
        all_pairs[subject]['pairs'][pair]['region'] = get_region_name(
            all_pairs[subject]['pairs'][pair]['location'].partition(' ')[-1]
        )

        if mni_available:
            all_pairs[subject]['pairs'][pair]['mni_x'] = pairs_from_json[
                standardized_pair]['atlases']['mni']['x']
            all_pairs[subject]['pairs'][pair]['mni_y'] = pairs_from_json[
                standardized_pair]['atlases']['mni']['y']
            all_pairs[subject]['pairs'][pair]['mni_z'] = pairs_from_json[
                standardized_pair]['atlases']['mni']['z']

    # Constructing the dataframe will not preserve the order from the
    # OrderedDict
    pairs_metadata = pd.DataFrame.from_dict(all_pairs[subject]['pairs'],
                                            orient='index')

    # Give some sort of default value when mni coordinates are not present
    if mni_available is False:
        pairs_metadata['mni_x'] = None
        pairs_metadata['mni_y'] = None
        pairs_metadata['mni_z'] = None

    pairs_metadata = pairs_metadata.reindex(all_pair_labels)
    pairs_metadata = pairs_metadata[['type', 'channel_1', 'channel_2', 'label',
                                     'region', 'location',
                                     'mni_x', 'mni_y', 'mni_z']]

    return pairs_metadata


def get_trigger_electrode_mask(montage_metadata_table, contact_label):
    # Make sure labels are not the index, but are instead numeric
    montage_metadata_table = montage_metadata_table.reset_index()

    if contact_label not in montage_metadata_table['label'].values:
        raise RuntimeError('%s not found. Check that label exists in the '
                           'electrode config file.')
    montage_metadata_table['is_trigger_electrode'] = (montage_metadata_table[
        'label'] == contact_label)
    trigger_electrode_mask = montage_metadata_table[
        'is_trigger_electrode'].values.tolist()
    return trigger_electrode_mask


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


def generate_pairs_for_ptsa(pairs):
    """ Convert bipolar pairs into a format expected by PTSA methods """
    classifier_fmt_pairs = generate_pairs_for_classifier(pairs, {})
    final_pairs = []
    for rec in classifier_fmt_pairs:
        final_pairs.append(('{:03d}'.format(rec[0]),
                            '{:03d}'.format(rec[1])))

    final_pairs = np.rec.array(np.array(final_pairs,
                                        dtype=[('ch0', 'S3'),
                                               ('ch1', 'S3')]))
    return final_pairs


def extract_monopolar_from_bipolar(bipolar_pairs_array):
    unique_monopolar_channels = []
    for rec in bipolar_pairs_array:
        rec0 = rec[0]
        rec1 = rec[1]
        if rec0 not in unique_monopolar_channels:
            unique_monopolar_channels.append(rec0)
        if rec1 not in unique_monopolar_channels:
            unique_monopolar_channels.append(rec1)

    final_channels = np.array(unique_monopolar_channels, dtype='S3')

    return final_channels


def reduce_pairs(pairs, stim_params, return_excluded=False):
    """Remove stim pairs from the pairs.json dict.

    Parameters
    ----------
    pairs : OrderedDict
        Full pairs.json as a dict
    stim_params : List[StimParameters]
    return_excluded :  bool
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
    """ Create a boolean mask indicating which electrodes should be included
    in classifier training/evaluation.

    Parameters
    ----------
    all_pairs: OrderedDict
    excluded_pairs: OrderedDict

    Returns
    -------
    mask : List[bool]
        Boolean mask of channels to include.

    """
    extracted_pairs = extract_pairs_dict(all_pairs)
    if type(extracted_pairs) != OrderedDict:
        raise RuntimeError("all pairs must be an orderd dict so that the "
                           "ordering can be correctly preserved when creating "
                           "the mask")

    pair_list = extracted_pairs.keys()
    mask = [label not in excluded_pairs for label in pair_list]

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
    used_pairs = [(int(a), int(b)) for a, b in used_pairs]

    recorded_pairs = []
    subject = list(all_pairs.keys())[0]
    for pair in all_pairs[subject]['pairs'].keys():
        channel_1 = all_pairs[subject]['pairs'][pair]['channel_1']
        channel_2 = all_pairs[subject]['pairs'][pair]['channel_2']
        pair_nums = (int(channel_1), int(channel_2))
        recorded_pairs.append(pair_nums)

    pair_mask = [True] * len(recorded_pairs)
    for i, pair in enumerate(recorded_pairs):
        if pair not in used_pairs:
            pair_mask[i] = False

    return np.array(pair_mask)


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


def load_pairs_from_json(subject, experiment, sessions=None, just_pairs=True,
                         rootdir='/'):
    """ Load montage information from pairs.json file

    Parameters
    ----------
    subject: str
        Subject ID
    experiment: str
        Experiment
    just_pairs: Bool
        If True, strip out any other metadata contained in the json file
    rootdir: str
        Mount point for RHINO

    Returns
    -------
    dict
        Dictionary containing metadata for all pairs in the given subjects'
        montage

    """
    json_reader = JsonIndexReader(os.path.join(rootdir,
                                               "protocols",
                                               "r1.json"))
    if sessions is not None:
        if len(sessions) == 1:
            all_pairs_paths = json_reader.aggregate_values('pairs',
                                                           subject_alias=subject,
                                                           experiment=experiment,
                                                           session=sessions[0])

            # For PS4 sessions, the experiment is listed as FR5/catFR5, so we need to look it up
            # based on original_experiment field instead
            if len(all_pairs_paths) == 0:
                all_pairs_paths = json_reader.aggregate_values('pairs',
                                                               subject_alias=subject,
                                                               original_experiment=experiment,
                                                               session=sessions[0])
        else:
            all_pairs_paths = json_reader.aggregate_values('pairs',
                                                           subject_alias=subject,
                                                           experiment=experiment)
    else:
        all_pairs_paths = json_reader.aggregate_values('pairs',
                                                       subject_alias=subject)

    if len(all_pairs_paths) == 0:
        raise RuntimeError("No pairs.json found for subject {} "
                           "and experiment {}".format(subject, experiment))

    elif len(all_pairs_paths) > 1:
        raise RuntimeError('Multiple montages are possible')

    # For simplicity, just load the first file since they *should* all be the
    # same
    bp_path = os.path.join(rootdir, list(all_pairs_paths)[0])
    with open(bp_path, 'r') as f:
        pair_data = json.load(f, object_pairs_hook=OrderedDict)

    if just_pairs:
        pair_data = extract_pairs_dict(pair_data)
        return pair_data

    # Update the subject_id not be standard format
    stored_subject = list(pair_data.keys())[0]
    if stored_subject != subject:
        pair_data[subject] = pair_data[stored_subject]
        del pair_data[stored_subject]

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
        if (loc_tag is not None) and (loc_tag != '') and (loc_tag != 'None'):
            return loc_tag

    if (bp_data['type_1'] == 'D') and ('wb' in atlases):
        wb_loc = atlases['wb']['region']
        if (wb_loc is not None) and (wb_loc != '') and (wb_loc != 'None'):
            return wb_loc

    try:
        if 'ind' in atlases:
            ind_loc = atlases['ind']['region']
            if (ind_loc is not None) and (ind_loc != '') and (ind_loc != 'None'):
                return ('Left ' if atlases['ind']['x'] < 0.0 else 'Right ') + ind_loc
    except TypeError:
        logger.warning(
            'Missing coordinates for an electrode. Likely a neurorad pipeline error')

    return '--'


def get_region_name(name):

    for k in MTL_LOC_DICT:
        if name in MTL_LOC_DICT[k]:
            return k

    for k in DK_LOC_DICT:
        if name in DK_LOC_DICT[k]:
            return k

    return 'Other'


def get_pairs(subject_id, experiment, sessions, paths):
    """Determine how we should figure out what pairs to use.

    Option 1: In the case of hardware bipolar recordings with the ENS, EEG
    data is stored in the HDF5 file which contains the Odin electrode config
    data so we can use this.

    Option 2: For monopolar recordings, we can just read the pairs.json from
    localization.

    Parameters
    ----------
    subject_id : str
        Subject ID
    experiment : str
        Experiment type
    sessions: list
        List of sessions to use
    paths : FilePaths
        Object for storing important file paths
    localization : int
        Localization number
    montage : int
        Montage number

    Returns
    -------
    all_pairs : dict
        All pairs used in the experiment.

    Notes
    -----
    This should only be used for getting pairs when building a report. For
    config generation, use generate_pairs_from_electrode_config. To use
    get_pairs, you would need to determine an open loop experiment that the
    subject completed and use that experiment instead of the experiment whose
    config file is being generated.

    """
    # Use * for session so we don't have to assume session numbers start at 0
    subject,montage = extract_subject_montage(subject_id)
    reader = JsonIndexReader(osp.join(paths.root,'protocols','r1.json'))
    event_path = list(reader.aggregate_values(
        'task_events',subject=subject,montage=montage,experiment=experiment)
        )[0]
    beh_root = event_path.partition('behavioral')[0]

    eeg_dir = osp.join(beh_root,'ephys', 'current_processed', 'noreref', '*.h5')
    files = glob(eeg_dir)

    # Read HDF5 file to get pairs
    if len(files) > 0:
        filename = files[0]

        with h5py.File(filename, 'r') as hfile:
            config_str = hfile['/config_files/electrode_config'].value

        # This will create a temporary directory that is removed when the
        # program exists the scope of the 'with' statement
        with tempdir() as temp_path:
            config_path = osp.join(temp_path, 'electrode_config.csv')
            with open(config_path, 'wb') as f:
                f.write(config_str)

            paths.electrode_config_file = config_path

            # generate_pairs_from_electrode_config panics if the .bin file isn't
            # found, so we have to make sure it's there
            touch(config_path.replace('.csv', '.bin'))

            all_pairs = generate_pairs_from_electrode_config(subject_id,
                                                             experiment,
                                                             sessions,
                                                             paths)

    # No HDF5 file exists, meaning this was a monopolar recording... read
    # pairs.json instead
    else:
        all_pairs = load_pairs_from_json(subject_id,
                                         experiment,
                                         sessions=sessions,
                                         just_pairs=False,
                                         rootdir=paths.root)

    return all_pairs


def get_classifier_excluded_leads(subject, all_pairs, rootdir='/'):
    """ Identify channels to be excluded using the classifier_excluded_leads.txt file

    Parameters:
    -----------
    subject: str
        Subject identifier

    paths: FilePaths object including RHINO root directory

    Returns:
    --------
    excluded_contacts: List of contacts in the same format as what is returned by make_stim_params

    """
    classifier_excluded_leads_path = osp.join(
        rootdir, 'data', 'eeg', subject, 'tal', 'classifier_excluded_leads.txt')
    if not osp.exists(classifier_excluded_leads_path):
        raise RuntimeError(
            "No classifier_excluded_leads.txt file found for {}".format(subject))

    with open(classifier_excluded_leads_path) as f:
        file_contents = f.read()
        excluded_labels = file_contents.split('\n')

    excluded_labels = [label for label in excluded_labels if label != '']

    # Find all bipolar pairs where any of the excluded labels appear
    excluded_anodes = []
    excluded_cathodes = []
    for pair in all_pairs[subject]['pairs'].keys():
        excluded_label_mask = [(pair.find(excluded) != -1)
                               for excluded in excluded_labels]
        if any(excluded_label_mask):
            anode = pair.split('-')[0]
            excluded_anodes.append(anode)
            cathode = pair.split('-')[1]
            excluded_cathodes.append(cathode)

    # This may seem a bit odd, but it returns the labels in a format that is easy to use by other functions
    # in ramutils
    excluded_contacts = make_stim_params(subject, excluded_anodes, excluded_cathodes,
                                         target_amplitudes=[
                                             0.5] * len(excluded_labels),
                                         root=rootdir)
    return excluded_contacts


def generate_pairs_from_electrode_config(subject, experiment, session, paths):
    """Load and verify the validity of the Odin electrode configuration file.

    Parameters
    ----------
    subject : str
        Subject ID
    experiment: str
        Experiment
    paths: FilePaths
        Object containing common file paths used for building reports/configs

    Returns
    -------
    pairs_from_ec : dict
        Minimal pairs.json based on the electrode configuration

    Raises
    ------
    RuntimeError
        If the csv or bin file are not found

    """
    prefix, _ = os.path.splitext(paths.electrode_config_file)
    csv_filename = prefix + '.csv'
    bin_filename = prefix + '.bin'

    if not os.path.exists(csv_filename):
        raise RuntimeError("{} not found!".format(csv_filename))
    if not os.path.exists(bin_filename):
        raise RuntimeError("{} not found!".format(bin_filename))

    # Create SeriesTransformation object to determine if this is monopolar,
    # mixed-mode, or bipolar
    # FIXME: load excluded pairs
    xform = SeriesTransformation.create(csv_filename, paths.pairs)

    # Odin electrode configuration
    ec = xform.elec_conf

    # This will mimic pairs.json (but only with labels).
    pairs_dict = OrderedDict()

    # FIXME: move the following logic into bptools
    # Hardware bipolar mode
    pairs_from_ec = {}
    if not xform.monopolar_possible():
        contacts = ec.contacts_as_recarray()

        for ch in ec.sense_channels:
            anode, cathode = ch.contact, ch.ref
            aname = bytes_to_str(
                contacts[contacts.jack_box_num == anode].contact_name[0])
            cname = bytes_to_str(
                contacts[contacts.jack_box_num == cathode].contact_name[0])
            name = '{}-{}'.format(aname, cname)
            pairs_dict[name] = {
                'channel_1': anode,
                'channel_2': cathode
            }

        # Note this is different from neurorad pipeline pairs.json because
        # the electrode configuration trumps it
        pairs_from_ec = {subject: {'pairs': pairs_dict}}

    # For monopolar, fall back to pairs.json
    else:
        pairs_from_ec = load_pairs_from_json(subject,
                                             experiment,
                                             sessions=session,
                                             just_pairs=False,
                                             rootdir=paths.root)

    return pairs_from_ec


def get_distances(pairs):
    """Get distances as an adjacency matrix.

    Parameters
    ----------
    pairs : pd.DataFrame
        A DataFrame describing a set of bipolar pairs

    Returns
    -------
    distmat : np.ndarray
        Adjacency matrix using exp(-distance / 120).

    """
    # positions matrix shaped as N_channels x 3
    pos = np.array([
        [row["mni_{}".format(c)] for c in ("x", "y", "z")]
        for _, row in pairs.sort_values(by=['channel_1', 'channel_2']).iterrows()
    ])

    distmat = np.empty((len(pos), len(pos)))

    for i, d1 in enumerate(pos):
        for j, d2 in enumerate(pos):
            if i <= j:
                distmat[i, j] = np.linalg.norm(d1 - d2, axis=0)
                distmat[j, i] = np.linalg.norm(d1 - d2, axis=0)

    distmat = 1 / np.exp(distmat / 120.)
    return distmat

def extract_rejected_pairs(subject, used_classifiers, ec_pairs, used_pair_mask):
    used_pair_mask = np.array(used_pair_mask)
    for classifier in used_classifiers:
        if classifier is not None:
            used_pair_mask &= compare_recorded_with_all_pairs(ec_pairs,
                                                              classifier.pairs)
    rejected_pairs = generate_pairs_for_classifier(ec_pairs, {})[
        ~used_pair_mask]
    rejected_pairs_as_stim_params = make_stim_params(
        subject, rejected_pairs.label0.astype(str),
        rejected_pairs.label1.astype(str),
        target_amplitudes=[0]*len(rejected_pairs))
    rejected_pairs = reduce_pairs(ec_pairs,
                                  rejected_pairs_as_stim_params,
                                  return_excluded=True)
    return rejected_pairs