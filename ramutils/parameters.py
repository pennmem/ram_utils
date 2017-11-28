"""Common experimental/model parameters."""

import os.path
import numpy as np
from traits.api import Int, Float, Bool, Array, String
from ramutils.schema import Schema


class StimParameters(Schema):
    """Single-channel stimulation parameters."""
    label = String(desc="stim channel label")
    anode = Int(desc="stim anode contact number")
    cathode = Int(desc="stim cathode contact number")
    frequency = Float(200., desc="stim pulse frequency [Hz]")
    duration = Float(500., desc="stim duration [ms]")

    # used in fixed-amplitidue experiments
    target_amplitude = Float(0.5, desc="stim amplitude [mA]")

    # used in variable-amplitude experiments
    min_amplitude = Float(0.1, desc="minimum allowable stim amplitude [mA]")
    max_amplitude = Float(2.0, desc="maximum allowable stim amplitude [mA]")


class FilePaths(object):
    """Paths to files that frequently get passed around to many tasks.

    All paths given relative to the root path but are converted to absolute
    paths on creation.

    Keyword arguments
    -----------------
    root : str
        Rhino mount point.
    dest : str
        Directory to write files to.
    electrode_config_file : str
        Path to Odin electrode configuration CSV file.
    pairs : str
        Path to ``pairs.json``.
    excluded_pairs : str
        Path to ``excluded_pairs.json``.

    """
    def __init__(self, **kwargs):
        # root is the only required kwarg
        self.root = os.path.expanduser(kwargs['root'])

        def get(key):
            return kwargs.get(key, None)

        def makepath(key):
            p = kwargs.get(key, None)
            return os.path.join(self.root, p.lstrip('/')) if p is not None else p

        self.dest = makepath('dest')
        self.electrode_config_file = makepath('electrode_config_file')
        self.pairs = makepath('pairs')
        self.excluded_pairs = makepath('excluded_pairs')


class ExperimentParameters(Schema):
    """
        Common parameters used in an experiment. Default values apply to the FR
        class of experiments.

    """
    width = Int(5, desc='wavelet width')
    freqs = Array(value=np.logspace(np.log10(6), np.log10(180), 8),
                  desc='frequencies to compute powers for')

    log_powers = Bool(True)  # FIXME: do we really need this?

    filt_order = Int(4, desc="Butterworth filter order")

    penalty_type = String('l2', desc='logistic regression penalty type')
    C = Float(7.2e-4, desc='inverse of regularization strength')
    n_perm = Int(200, desc='number of permutations to use for cross-validation')
    solver = String('liblinear', desc='algorithm to use in optimization process')

    baseline_removal_start_time = Int(1000, desc="The amount of time to "
                                                   "skip at the beginning of "
                                                   "the sessions [ms]")
    retrieval_time = Int(29000, desc="The amount of time within the recall "
                                       "period to consider")
    empty_epoch_duration = Int(500, desc="The length of desired empty "
                                           "epochs [ms]")
    pre_event_buf = Int(2000, desc="The time before each event to exclude ["
                                    "ms]")
    post_event_buf = Int(1000, desc="The time after each event to exclude ["
                                     "ms]")
    inter_response_time = Int(1000, desc="Duration between events required "
                                           "for a recall event to have "
                                           "occured after some sort of "
                                           "cognitive process [ms]")


class FRParameters(ExperimentParameters):
    """Free recall experiment parameters relevant for classification."""
    encoding_start_time = Float(0., desc="encoding start time [s]")
    encoding_end_time = Float(1.366, desc="encoding end time [s]")
    encoding_buf = Float(1.365, desc="encoding buffer time [s]")

    retrieval_start_time = Float(-0.525, desc="retrieval start time [s]")
    retrieval_end_time = Float(0., desc="retrieval end time [s]")
    retrieval_buf = Float(0.524, desc="retrieval buffer time [s]")

    encoding_only = Bool(False, desc="use encoding-only classifier")
    encoding_multiplier = Float(2.5, desc="weighting factor for encoding "
                                          "samples")
    combine_events = Bool(True, desc="combine record-only events for "
                                     "classifier training")


class PALParameters(FRParameters):
    """
        Paired associates experiment parameters relevant for classification.
        It inhertis all of the same parameters as FR experiments and adds a
        few more
    """

    pal_start_time = Float(0.3, desc="encoding start time for PAL [s]")
    pal_end_time = Float(2.00, desc="encoding end time for PAL [s]")
    pal_buf_time = Float(1.2)

    pal_retrieval_start_time = Float(-0.625, desc="retrieval start time for "
                                                  "PAL [s]")
    pal_retrieval_end_time = Float(-0.1, desc="retrieval end time for PAL [s]")
    pal_retrieval_buf = Float(0.524, desc="retrieval buffer for PAL [s]")

    encoding_multiplier = Float(7.2, desc="weighting factor for encoding "
                                          "samples in PAL")
    pal_multiplier = Float(1.93, desc="weighting factor for PAL samples")

