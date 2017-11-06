"""Tasks specific to the Medtronic Odin ENS."""

from collections import OrderedDict
import functools
from itertools import chain
import json
import os.path
from tempfile import gettempdir
import shutil

try:
    from typing import List
except ImportError:
    pass

import numpy as np

from tornado.template import Template

from bptools.transform import SeriesTransformation
from classiflib import ClassifierContainer, dtypes

from ramutils.log import get_logger
from ramutils.tasks import task
from ramutils.utils import reindent_json

CLASSIFIER_VERSION = "1.0.2"

logger = get_logger()


# FIXME: logic for generating pairs should be in bptools
@task()
def generate_pairs_from_electrode_config(subject, paths):
    """Load and verify the validity of the Odin electrode configuration file.

    :param str subject: Subject ID
    :param FilePaths paths:
    :returns: minimal pairs.json based on the electrode configuration
    :rtype: dict
    :raises RuntimeError: if the csv or bin file are not found

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


@task(cache=False)
def generate_ramulator_config(subject, experiment, container, stim_params,
                              paths, dest):
    """Create configuration files for Ramulator.

    Note that the current template format will not work for FR5 experiments
    since the config format is not the same.

    In hardware biplar mode, the neurorad pipeline generates a ``pairs.json``
    file that differs from the electrode configured pairs. It is up to the user
    of the pipeline to ensure that the path to the correct ``pairs.json`` is
    supplied (although Ramulator does not use it in this case).

    :param str subject:
    :param str experiment:
    :param ClassifierContainer container: serialized classifier
    :param List[StimParameters] stim_params: list of stimulation parameters
    :param FilePaths paths:
    :param str dest: location to write configuration files to
    :returns: path to generated configuration zip file
    :rtype: str

    """
    subject = subject.split('_')[0]

    stim_dict = {
        stim_param.label: {
            "min_stim_amplitude": stim_param.min_amplitude,
            "max_stim_amplitude": stim_param.max_amplitude,
            "stim_frequency": stim_param.frequency,
            "stim_duration": stim_param.duration,
            "stim_amplitude": stim_param.target_amplitude,
        }
        for stim_param in stim_params
    }

    dest = os.path.expanduser(dest)
    config_dir_root = os.path.join(dest, subject, experiment)
    config_files_dir = os.path.join(config_dir_root, 'config_files')
    try:
        os.makedirs(config_files_dir)
    except OSError as e:
        if e.errno != 17:  # File exists
            raise

    classifier_path = os.path.join(config_files_dir, '{}-classifier.zip'.format(subject))
    ec_prefix, _ = os.path.splitext(paths.electrode_config_file)

    template_filename = os.path.join(
        os.path.dirname(__file__), 'templates', 'ramulator_config.json')

    with open(template_filename, 'r') as f:
        experiment_config_template = Template(f.read())

    experiment_config_content = experiment_config_template.generate(
        subject=subject,
        experiment=experiment,
        classifier_file='config_files/{}'.format(os.path.basename(classifier_path)),
        classifier_version=CLASSIFIER_VERSION,
        stim_params_dict=stim_dict,
        electrode_config_file='config_files/{}'.format(os.path.basename(ec_prefix + '.bin')),
        biomarker_threshold=0.5
    )

    with open(os.path.join(config_dir_root, 'experiment_config.json'), 'w') as f:
        tmp_path = os.path.join(gettempdir(), "experiment_config.json")
        with open(tmp_path, 'w') as tmpfile:
            tmpfile.write(experiment_config_content)
        f.write(reindent_json(tmp_path))

    container.save(classifier_path, overwrite=True)

    # Save some typing below...
    conffile = functools.partial(os.path.join, config_files_dir)

    # Copy pairs.json and excluded_pairs.json to the config directory
    shutil.copy(paths.pairs, conffile('pairs.json'))
    # FIXME: this isn't saved anywhere else
    shutil.copy(paths.excluded_pairs, conffile('excluded_pairs.json'))

    # Copy electrode config files
    csv = paths.electrode_config_file
    bin = paths.electrode_config_file.replace('.csv', '.bin')
    shutil.copy(csv, conffile(os.path.basename(csv)))
    shutil.copy(bin, conffile(os.path.basename(bin)))

    zip_prefix = os.path.join(dest, '{}_{}'.format(subject, experiment))
    zip_path = zip_prefix + '.zip'
    shutil.make_archive(zip_prefix, 'zip', root_dir=config_dir_root)
    logger.info("Created experiment_config zip file: %s", zip_path)
    return zip_path
