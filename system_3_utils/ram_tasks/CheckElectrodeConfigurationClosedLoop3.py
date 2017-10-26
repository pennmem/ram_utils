from __future__ import print_function

import sys
from os.path import *
import json
from collections import OrderedDict

import numpy as np

from bptools.odin import ElectrodeConfig

from ReportUtils import RamTask
from ramutils.log import get_logger

logger = get_logger()


class CheckElectrodeConfigurationClosedLoop3(RamTask):
    def __init__(self, params, mark_as_completed=True, force_rerun=False):
        super(CheckElectrodeConfigurationClosedLoop3, self).__init__(mark_as_completed, force_rerun=force_rerun)
        self.params = params

    def validate_montage(self, electrode_config):
        # FIXME: update this for bptools

        args = self.pipeline.args

        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        monopolar_channels_int_array = monopolar_channels.astype(np.int)
        sense_channels_array = electrode_config.sense_channels_as_recarray()
        sense_array_int = sense_channels_array.jack_box_num.astype(np.int)

        # check if monopolar channels are the same as sense channels in the .bin/.csv file
        monopolar_same_as_sense = np.array_equal(sense_array_int, monopolar_channels_int_array)

        if not monopolar_same_as_sense:
            logger.error(
                'ELECTRODE CONFIG ERROR:\n%s',
                'Sense electrodes jack_box numbers defined in .bin/.csv file do not match jack_box_numbers in contacts.json')

        anode_nums = args.anode_nums if args.anode_nums else [args.anode_num]
        cathode_nums = args.cathode_nums if args.cathode_nums else [args.cathode_num]
        for (anode_num,cathode_num) in zip(anode_nums,cathode_nums):
            logger.info('anode: %d', anode_num)
            logger.info('cathode: %d', cathode_num)

            if anode_num == cathode_num:
                logger.error(
                    'ELECTRODE CONFIG ERROR:\n%s',
                    'Anode jackbox number must be different from cathode number')

            stim_index_pair_present = np.all(np.in1d([anode_num, cathode_num],monopolar_channels_int_array))

            if not stim_index_pair_present:
                logger.error('ELECTRODE CONFIG ERROR:\n%s',
                             'Could not find requested stim pair electrode numbers in contacts.json')

            # looping over stim channels to check if there exist a channel for
            # which anode and cathode jackbox numbers match those specified by
            # the user
            stim_channel_present = False
            for stim_chan_label, stim_chan_data in electrode_config.stim_channels.items():
                if stim_chan_data.anodes[0] == anode_num and stim_chan_data.cathodes[0] == cathode_num:
                    stim_channel_present = True
                    break

            if not stim_channel_present:
                logger.error('ELECTRODE CONFIG ERROR:\n%s',
                             'Could not find requested stim pair electrode numbers in .csv/.bin electrode configuration file')

            # finally will check labels if user provided the labels
            anode_label = self.pipeline.args.anode.strip().upper()
            cathode_label = self.pipeline.args.cathode.strip().upper()

            if anode_label and cathode_label:

                anode_idx_array = np.where(sense_channels_array.jack_box_num == anode_num)
                cathode_idx_array = np.where(sense_channels_array.jack_box_num == cathode_num)

                anode_label_from_contacts = None if not len(anode_idx_array) else sense_channels_array.contact_name[anode_idx_array[0]][0]
                cathode_label_from_contacts = None if not len(cathode_idx_array) else sense_channels_array.contact_name[cathode_idx_array[0]][0]
                anode_label_from_contacts = anode_label_from_contacts.strip().upper()
                cathode_label_from_contacts = cathode_label_from_contacts.strip().upper()

                if str(anode_label_from_contacts) != anode_label or cathode_label_from_contacts != cathode_label:
                    logger.error(
                        'ELECTRODE CONFIG ERROR:\n'
                        'specified electrode labels for anode and cathode (%s, %s) do no match electrodes'
                        ' found in contacts.json (%s,%s)',
                        anode_label, cathode_label, anode_label_from_contacts,
                        cathode_label_from_contacts
                    )

            self.pass_object('stim_chan_label', stim_chan_label)

    def run(self):
        electrode_config_file = self.pipeline.args.electrode_config_file

        electrode_fname = abspath(electrode_config_file)
        electrode_core_fname, ext = splitext(electrode_fname)
        electrode_csv = electrode_core_fname + '.csv'

        if not isfile(electrode_csv):
            print('Missing .csv Electrode Config File. Please make sure that %s is stored in %s' % (
                electrode_csv, dirname(electrode_csv)))

            sys.exit(1)

        ec = ElectrodeConfig(electrode_csv)

        # FIXME: validate!
        # self.validate_montage(electrode_config=ec)

        # This will mimic pairs.json (but only with labels).
        pairs_dict = OrderedDict()

        try:
            contacts = ec.contacts_as_recarray()

            # FIXME: move this logic into bptools
            for ch in ec.sense_channels:
                anode, cathode = ch.contact, ch.ref
                aname = contacts[contacts.jack_box_num == anode].contact_name[0]
                cname = contacts[contacts.jack_box_num == cathode].contact_name[0]
                name = '{}-{}'.format(aname, cname)
                pairs_dict[name] = {
                    'channel_1': anode,
                    'channel_2': cathode
                }

            pairs_from_ec = {self.pipeline.subject: {'pairs': pairs_dict}}
            with open(self.get_path_to_resource_in_workspace('pairs.json'),'w') as pf:
                json.dump(pairs_from_ec,pf,indent=2)
            channels = np.array(['{:03d}'.format(contact.port) for contact in ec.contacts])

            # FIXME: what passed objects do we actually need here?
            self.pass_object('monopolar_channels', channels)
            self.pass_object('config_pairs_path', self.get_path_to_resource_in_workspace('pairs.json'))
            self.pass_object('config_pairs_dict', pairs_from_ec)

        # FIXME: This is leftover from the old implementation but shouldn't happen.
        except IndexError:
            raise
        finally:
            self.pass_object('config_name', ec.name)
