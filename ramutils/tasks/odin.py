"""Tasks specific to the Medtronic Odin ENS."""

from collections import OrderedDict
import functools
from itertools import chain, cycle
import json
import os.path
import zipfile

import numpy as np

from tornado.template import Template

from bptools.transform import SeriesTransformation
from classiflib import ClassifierContainer, dtypes

from ramutils.log import get_logger
from ramutils.tasks import task

CLASSIFIER_VERSION = "1.0.2"

logger = get_logger()


@task()
def generate_pairs_from_electrode_config(path, pairs_path, subject):
    """Load and verify the validity of the Odin electrode configuration file.

    :param str path: Path to the Odin electrode .csv or .bin config file
    :param str pairs_path: Path to pairs.json
    :param str subject: Subject ID
    :returns: minimal pairs.json based on the electrode configuration
    :rtype: dict
    :raises RuntimeError: if the csv or bin file are not found

    """
    prefix, _ = os.path.splitext(path)
    csv_filename = prefix + '.csv'
    bin_filename = prefix + '.bin'

    if not os.path.exists(csv_filename):
        raise RuntimeError("{} not found!".format(csv_filename))
    if not os.path.exists(bin_filename):
        raise RuntimeError("{} not found!".format(bin_filename))

    # Create SeriesTransformation object to determine if this is monopolar,
    # mixed-mode, or bipolar
    # FIXME: load excluded pairs
    xform = SeriesTransformation.create(csv_filename, pairs_path)

    # Odin electrode configuration
    ec = xform.elec_conf

    # This will mimic pairs.json (but only with labels).
    pairs_dict = OrderedDict()

    channels = np.array(['{:03d}'.format(contact.port) for contact in ec.contacts])

    # FIXME: move the following logic into bptools
    # Hardware bipolar mode
    if not xform.monopolar_possible():
        contacts = ec.contacts_as_recarray()

        for ch in ec.sense_channels:
            anode, cathode = ch.contact, ch.ref
            aname = contacts[contacts.jack_box_num == anode].contact_name[0]
            cname = contacts[contacts.jack_box_num == cathode].contact_name[0]
            name = '{}-{}'.format(aname, cname)
            pairs_dict[name] = {
                'channel_1': anode,
                'channel_2': cathode
            }

        # Note this is different from neurorad pipeline pairs.json because
        # the electrode configuration trumps it
        pairs_from_ec = {subject: {'pairs': pairs_dict}}

        # FIXME: new task to write pairs.json where it belongs
        # with open(self.get_path_to_resource_in_workspace('pairs.json'), 'w') as pf:
        #     json.dump(pairs_from_ec, pf, indent=2)

        # FIXME: need to validate stim pair inputs match those defined in config file

        return pairs_from_ec


@task(cache=False, log_args=False)
def save_montage_files(pairs, excluded_pairs, dest):
    """Saves the montage files (pairs.json and excluded_pairs.json) to the
    config directory.

    :param dict pairs:
    :param dic excluded_pairs:
    :param str dest: directory to write JSON files to

    """
    if not os.path.exists(dest):
        try:
            os.makedirs(dest)
        except OSError:
            pass
    else:
        assert os.path.isdir(dest), "dest must be a directory"

    dump = functools.partial(json.dump, indent=2, sort_keys=True)

    with open(os.path.join(dest, "pairs.json"), 'w') as f:
        dump(pairs, f)

    with open(os.path.join(dest, "excluded_pairs.json"), 'w') as f:
        dump(excluded_pairs, f)
