"""Tasks specific to the Medtronic Odin ENS."""

from datetime import datetime
import functools
import json
import os.path
import shutil
import warnings

try:
    from typing import List
except ImportError:
    pass

from bptools.odin import ElectrodeConfig
from classiflib import ClassifierContainer

from ramutils.constants import EXPERIMENTS
from ramutils.exc import MissingArgumentsError
from ramutils.log import get_logger
from ramutils.tasks import task

__all__ = [
    'generate_electrode_config',
    'generate_ramulator_config',
]

CLASSIFIER_VERSION = "1.0.2"

logger = get_logger()


@task(cache=False)
def generate_electrode_config(subject, paths, anodes=None, cathodes=None,
                              localization=0, montage=0,
                              default_surface_area=0.001):
    """Generate electrode configuration files (CSV and binary).

    Parameters
    ----------
    subject : str
        Subjebct ID
    paths : FilePaths
    anodes : List[str]
        List of stim anode labels.
    cathodes : List[str]
        List of stim cathode labels.
    localization : int
        Localization number (default: 0)
    montage : int
        Montage number (default: 0)
    default_surface_area : float
        Default surface area to set all electrodes to in mm^2. Only used if no
        area file can be found.

    Returns
    -------
    paths : FilePaths
        Updated :class:`FilePaths` object with path to the electrode config file
        defined.

    Notes
    -----
    At present, this will only allow for generating hardware-bipolar electrode
    config files.

    """
    docs_dir = os.path.join(paths.root, 'data', 'eeg', subject, 'docs')
    jacksheet_filename = os.path.join(docs_dir, 'jacksheet.txt')
    area = os.path.join(docs_dir, 'area.txt')

    if not os.path.exists(area):
        if paths.area_file is not None:
            area = paths.area_file
        else:
            logger.warning("No area file found. "
                           "Using default value of %f for all electrodes!",
                           default_surface_area)
            area = default_surface_area

    ec = ElectrodeConfig.from_jacksheet(jacksheet_filename, subject, area=area)

    if anodes is not None:
        assert cathodes is not None
        assert len(cathodes) == len(anodes)
        for labels in zip(anodes, cathodes):
            ec.add_stim_channel(*labels)

    date = datetime.now().strftime('%d%b%Y').upper()
    prefix = '{subject:s}_{date:s}L{localization:d}M{montage:d}{stim:s}'.format(
        subject=subject, date=date, localization=localization, montage=montage,
        stim=('STIM' if anodes is not None else 'NOSTIM')
    )
    csv_path = os.path.join(paths.root, paths.dest, prefix + '.csv')
    bin_path = csv_path.replace('.csv', '.bin')

    ec.to_csv(csv_path)
    logger.info("Wrote CSV electrode config: %s", csv_path)
    ec.to_bin(bin_path)
    logger.info("Wrote binary electrode config: %s", bin_path)

    paths.electrode_config_file = csv_path
    return paths


def _make_experiment_specific_data_section(experiment, stim_params,
                                           classifier_file,
                                           classifier_version=CLASSIFIER_VERSION,
                                           trigger_pairs=None):
    """Return a dict containing the config section ``experiment_specific_data``.

    Parameters
    ----------
    experiment : str
    stim_params : dict
    classifier_file : str
    classifier_version : str
    trigger_pairs : List[str] or None

    Returns
    -------
    dict representation of the ``experiment_specific_data`` section.

    """
    if experiment in EXPERIMENTS['record_only']:
        return {}

    def make_stim_channel_section(params, key):
        stub = {
            "stim_duration": 500,
            "stim_frequency": 200
        }

        if experiment.startswith('PS') or experiment == 'AmplitudeDetermination':
            stub.update({
                'min_stim_amplitude': params[key]['min_stim_amplitude'],
                'max_stim_amplitude': params[key]['max_stim_amplitude']
            })
        else:
            stub.update({
                'stim_amplitude': params[key]['stim_amplitude']
            })

        # We also need num_amplitudes for PS5 experiments. Note that it is
        # fixed at 3 for now, but may become a command-line option later.
        if experiment.startswith('PS5'):
            stub.update({
                'num_amplitudes': 3,
            })

        return stub

    # Add top-level stuff for experiments that require it
    if not experiment.startswith('PS5'):
        esd = {
            "allow_classifier_generalization": True,
            "classifier_file": "config_files/{}".format(classifier_file),
            "classifier_version": classifier_version,
            "random_stim_prob": False,
            "save_debug_output": True
        }
    else:
        esd = {}

    # Why oh why must everything be a special snowflake?
    key = 'stim_electrode_pairs' if experiment == 'AmplitudeDetermination' else 'stim_channels'
    esd[key] = {
        label: make_stim_channel_section(stim_params, label)
        for label in stim_params
    }

    # Add trigger section for PS5
    if experiment.startswith('PS5'):
        esd['trigger'] = {'pairs': trigger_pairs}

    return esd


def _make_experiment_specs_section(experiment):
    """Generate the ``experiment_specs`` config section.

    Parameters
    ----------
    experiment : str

    Returns
    -------
    ``experiment_specs`` dict

    """
    if experiment in EXPERIMENTS['record_only']:
        return {}

    # FIXME: values below shouldn't be hardcoded
    specs = {
        "version": "3.0.0",
        "experiment_type": experiment,
        "biomarker_sample_start_time_offset": 0,
        "biomarker_sample_time_length": 1366,
        "buffer_time": 1365,
        "stim_duration": 500,
        "freq_min": 6,
        "freq_max": 180,
        "num_freqs": 8,
        "num_items": 300,
    }

    # FIXME: values below shouldn't be hardcoded
    if 'PS4' in experiment:
        specs.update({
            "retrieval_biomarker_sample_start_time_offset": 0,
            "retrieval_biomarker_sample_time_length": 525,
            "retrieval_buffer_time": 524,
            "post_stim_biomarker_sample_time_length": 500,
            "post_stim_buffer_time": 499,
            "post_stim_wait_time": 100,
        })

    return specs


def _make_ramulator_config_json(subject, experiment, electrode_config_file,
                                stim_params, classifier_file=None,
                                classifier_version=None, extended_blanking=True,
                                trigger_pairs=None):
    """Create the Ramulator ``experiment_config.json`` file.

    Parameters
    ----------
    subject : str
    experiment : str
    electrode_config_file : str
    stim_params : dict
    classifier_file : str
    classifier_version : str
    extended_blanking : bool
    trigger_pairs : List[str] or None

    Returns
    -------
    str

    Notes
    -----
    Not all settings are actually used in all experiments. For example, there
    are several stim-specific settings that we always write here even for
    record-only experiments. When not used, they are simply ignored by
    Ramulator.

    """
    config = {
        'subject': subject,
        'experiment': {
            'type': experiment,
            'experiment_specific_data':
                _make_experiment_specific_data_section(experiment,
                                                       stim_params,
                                                       classifier_file,
                                                       classifier_version,
                                                       trigger_pairs),
            'experiment_specs': _make_experiment_specs_section(experiment),
            'artifact_detection': {
                "allow_artifact_detection": True,
                "artifact_detection_number_of_stims_per_channel": 15,
                "artifact_detection_sample_time_length": 500,
                "artifact_detection_inter_stim_interval": 2000,
                "allow_artifact_detection_during_session": False,
            },
        },

        "biomarker_threshold": 0.5,
        "electrode_config_file": "config_files/{}".format(electrode_config_file),
        "montage_file": "config_files/pairs.json",
        "excluded_montage_file": "config_files/excluded_pairs.json",
        "global_settings": {
            "data_dir": "SET_AUTOMATICALLY_AT_A_RUNTIME",
            "experiment_config_filename": "SET_AUTOMATICALLY_AT_A_RUNTIME",
            "extended_blanking": extended_blanking,
            "plot_fps": 5,
            "plot_window_length": 20000,
            "plot_update_style": "Sweeping",
            "max_session_length": 120,
            "sampling_rate": 1000,
            "odin_lib_debug_level": 0,
            "connect_to_task_laptop": True if experiment != 'AmplitudeDetermination' else False
        }
    }

    return json.dumps(config, indent=2, sort_keys=True)


@task(cache=False)
def generate_ramulator_config(subject, experiment, container, stim_params,
                              paths, pairs=None, excluded_pairs=None,
                              exp_params=None, extended_blanking=True,
                              trigger_pairs=None):
    """Create configuration files for Ramulator.

    In hardware bipolar mode, the neurorad pipeline generates a ``pairs.json``
    file that differs from the electrode configured pairs. It is up to the user
    of the pipeline to ensure that the path to the correct ``pairs.json`` is
    supplied (although Ramulator does not use it in this case).

    Parameters
    ----------
    subject : str
    experiment : str
    container : ClassifierContainer or None
        serialized classifier
    stim_params : List[StimParameters]
        list of stimulation parameters
    paths : FilePaths
    excluded_pairs : dict
        Pairs excluded from the classifier (pairs that contain a stim contact
        and possibly some others)
    exp_params : ExperimentParameters
        All parameters used in training the classifier. This is partially
        redundant with some data stored in the ``container`` object.
    extended_blanking : bool
        Whether or not to enable the ENS extended blanking (default: True).
    trigger_pairs : List[str] or None
        Pairs to be used for triggering stim in PS5.

    Returns
    -------
    zip_path : str
        Path to generated configuration zip file

    """
    no_classifier_experiments = (
        ['AmplitudeDetermination'] + EXPERIMENTS['record_only'] +
        ['PS5_FR', 'PS5_CatFR']
    )

    if container is None and experiment not in no_classifier_experiments:
        raise MissingArgumentsError("container must not be None")

    if experiment.startswith('PS5') and trigger_pairs is None:
        raise MissingArgumentsError("trigger_pairs needed for PS5")

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

    # Put all files in a "clean" directory in the destination path. This just
    # creates a timestamped folder so that we don't end up bundling up files
    # that were around from a previous experiment config generation run.
    dest = paths.dest
    clean_dir = datetime.now().strftime('%Y%m%d_%H%m%S')

    config_dir_root = os.path.join(dest, clean_dir, subject, experiment)
    config_files_dir = os.path.join(config_dir_root, 'config_files')
    try:
        os.makedirs(config_files_dir)
    except OSError as e:
        if e.errno != 17:  # File exists
            raise

    classifier_path = os.path.join(config_files_dir, '{}-classifier.zip'.format(subject))
    ec_prefix, _ = os.path.splitext(paths.electrode_config_file)

    experiment_config_content = _make_ramulator_config_json(
        subject, experiment, os.path.basename(ec_prefix + '.bin'), stim_dict,
        os.path.basename(classifier_path), CLASSIFIER_VERSION,
        extended_blanking=extended_blanking,
        trigger_pairs=trigger_pairs,
    )

    with open(os.path.join(config_dir_root, 'experiment_config.json'), 'w') as f:
        f.write(experiment_config_content)

    if container is not None:
        container.save(classifier_path, overwrite=True)

    # Save some typing below...
    conffile = functools.partial(os.path.join, config_files_dir)

    # Write pairs.json and excluded_pairs.json to the config directory. We can
    # give pairs.json as a parameter (generally when read from the electrode
    # config file) or as a path (when using the neurorad pipeline's pairs.json).
    if pairs is not None:
        with open(conffile('pairs.json'), 'w') as f:
            json.dump(pairs, f)
    else:
        shutil.copy(paths.pairs, conffile('pairs.json'))
    if excluded_pairs is not None:
        # Make format of excluded pairs more standard
        excluded_pairs = {subject: {'pairs': excluded_pairs}}
        with open(conffile('excluded_pairs.json'), 'w') as f:
            json.dump(excluded_pairs, f)

    # Copy electrode config files
    csv = paths.electrode_config_file
    bin = csv.replace('.csv', '.bin')
    shutil.copy(csv, conffile(os.path.basename(csv)))
    shutil.copy(bin, conffile(os.path.basename(bin)))

    # Serialize experiment parameters
    if exp_params is not None:
        exp_params.to_hdf(os.path.join(config_dir_root, 'exp_params.h5'))
    else:
        if experiment not in no_classifier_experiments:
            warnings.warn("No ExperimentParameters object passed; "
                          "classifier may not be 100% reproducible", UserWarning)

    filename_tmpl = '{subject:s}_{experiment:s}{pairs:s}{date:s}'
    pair_str = '_' + "_".join([pair.label for pair in stim_params]) + '_' if len(stim_params) else '_'
    zip_prefix = os.path.join(dest, filename_tmpl.format(
        subject=subject,
        experiment=experiment,
        pairs=pair_str,
        date=datetime.now().strftime('%d%b%Y').upper()
    ))
    zip_path = zip_prefix + '.zip'
    shutil.make_archive(zip_prefix, 'zip', root_dir=config_dir_root)
    logger.info("Created experiment_config zip file: %s", zip_path)
    return zip_path
