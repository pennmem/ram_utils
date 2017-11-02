"""Tasks specific to the Medtronic Odin ENS."""

import os.path
from collections import OrderedDict
# import json

import numpy as np

from bptools.transform import SeriesTransformation

from ramutils.tasks import task


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
