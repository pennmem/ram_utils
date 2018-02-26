from ramutils.constants import EXPERIMENTS
from ramutils.exc import (
    MissingArgumentsError, MultistimNotAllowedException, ValidationError
)
from ramutils.montage import generate_pairs_from_electrode_config
from ramutils.tasks import *


@task(cache=False)
def validate_pairs(subject, ec_pairs, trigger_pairs=None):
    """Validate that specified pairs exist in the electrode config.

    Parameters
    ----------
    subject : str
        Subject ID
    ec_pairs : OrderedDict
        Contents of pairs.json as generated from the electrode config file.
        Pairs here are specified as ``<anode label>-<cathode label>``.
    trigger_pairs : List
        List of specified pairs to be used as triggers for PS5. Pairs here are
        specified as ``<anode label>_<cathode label>``.

    Notes
    -----
    Generating the electrode config file will already fail if anodes/cathodes
    are not spelled correctly, so we only actually check trigger pairs for PS5
    here.

    """
    pairs_json = ec_pairs[subject]['pairs']

    if trigger_pairs is not None:
        for pair in trigger_pairs:
            hyphenated_pair = pair.replace('_', '-')
            if hyphenated_pair not in pairs_json:
                raise ValidationError(
                    "trigger pair " + pair +
                    " not found in pairs.json (check for typos!)"
                )


def make_ramulator_config(subject, experiment, paths, stim_params,
                          exp_params=None, vispath=None, extended_blanking=True,
                          localization=0, montage=0, default_surface_area=0.001,
                          trigger_pairs=None, use_common_reference=False):
    """ Generate configuration files for a Ramulator experiment

    Parameters
    ----------
    subject : str
        Subject ID
    experiment : str
        Experiment to generate configuration file for
    paths : FilePaths
    stim_params : List[StimParameters]
        Stimulation parameters for this experiment.
    exp_params : ExperimentParameters
        Parameters for the experiment.
    vispath : str
        Path to save task graph visualization to if given.
    extended_blanking : bool
        Whether to enable extended blanking on the ENS (default: True).
    localization : int
        Localization number
    montage : int
        Montage number
    default_surface_area : float
        Default surface area to set all electrodes to in mm^2. Only used if no
        area file can be found.
    trigger_pairs : List[str] or None
        Pairs to use for triggering stim in PS5 experiments.
    use_common_reference : bool
        Use a common reference in the electrode configuration instead of bipolar
        referencing.

    Returns
    -------
    The path to the generated configuration zip file.

    """
    if len(stim_params) > 1 and experiment not in EXPERIMENTS['multistim']:
        raise MultistimNotAllowedException

    if trigger_pairs is None:
        if experiment.startswith('PS5'):
            raise MissingArgumentsError("PS5 requires trigger_pairs")

        # setting to empty list for validation
        trigger_pairs = []

    anodes = [c.anode_label for c in stim_params]
    cathodes = [c.cathode_label for c in stim_params]

    # If the electrode config path is defined, load it instead of creating a new
    # one. This is useful if we want to make comparisons with old referencing
    # schemes that are not currently implemented in bptools.
    if paths.electrode_config_file is None:
        paths = generate_electrode_config(subject, paths, anodes, cathodes,
                                          localization, montage,
                                          default_surface_area,
                                          use_common_reference)

    # Note: All of these pairs variables are of type OrderedDict, which is
    # crucial for preserving the initial order of the electrodes in the
    # config file
    ec_pairs = make_task(generate_pairs_from_electrode_config, subject,
                         experiment, None, paths)
    excluded_pairs = reduce_pairs(ec_pairs, stim_params, True)
    used_pair_mask = get_used_pair_mask(ec_pairs, excluded_pairs)
    final_pairs = generate_pairs_for_classifier(ec_pairs, excluded_pairs)

    # Ensure specified pairs exist. We have to call .compute here since no
    # other tasks depend on the output of this task.
    validate_pairs(subject, ec_pairs, trigger_pairs).compute()

    # Special case handling of no-classifier tasks
    no_classifier_experiments = EXPERIMENTS['record_only'] + [
        'AmplitudeDetermination',
        'PS5_FR',
        'PS5_CatFR',
    ]
    if experiment in no_classifier_experiments:
        container = None
        config_path = generate_ramulator_config(subject=subject,
                                                experiment=experiment,
                                                container=container,
                                                stim_params=stim_params,
                                                paths=paths,
                                                pairs=ec_pairs,
                                                excluded_pairs=excluded_pairs,
                                                extended_blanking=extended_blanking,
                                                trigger_pairs=trigger_pairs)
        return config_path.compute()

    if ("FR" not in experiment) and ("PAL" not in experiment):
        raise RuntimeError("Only PAL, FR, and catFR experiments are currently"
                           "implemented")
    kwargs = exp_params.to_dict()

    all_task_events = build_training_data(subject, experiment, paths, **kwargs)

    # FIXME: If PTSA is updated to not remove events behind this scenes, this
    # won't be necessary. Or, if we can remove bad events before passing to
    # compute powers, then we won't have to catch the events
    powers, final_task_events = compute_normalized_powers(all_task_events,
                                                          bipolar_pairs=ec_pairs,
                                                          **kwargs)
    reduced_powers = reduce_powers(powers, used_pair_mask, len(kwargs['freqs']))

    sample_weights = get_sample_weights(final_task_events, **kwargs)

    classifier = train_classifier(reduced_powers,
                                  final_task_events,
                                  sample_weights,
                                  kwargs['C'],
                                  kwargs['penalty_type'],
                                  kwargs['solver'])

    cross_validation_results = perform_cross_validation(classifier,
                                                        reduced_powers,
                                                        final_task_events,
                                                        kwargs['n_perm'],
                                                        'Trained Classifier',
                                                        **kwargs)

    container = serialize_classifier(classifier,
                                     final_pairs,
                                     reduced_powers,
                                     final_task_events,
                                     sample_weights,
                                     cross_validation_results,
                                     subject)

    config_path = generate_ramulator_config(subject=subject,
                                            experiment=experiment,
                                            container=container,
                                            stim_params=stim_params,
                                            paths=paths,
                                            pairs=ec_pairs,
                                            excluded_pairs=excluded_pairs,
                                            exp_params=exp_params,
                                            extended_blanking=extended_blanking)

    if vispath is not None:
        config_path.visualize(filename=vispath)

    return config_path.compute()
