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


class FilePaths(Schema):
    """Paths to files that frequently get passed around to many tasks.

    All paths given are relative to :attr:`root` when accessing attributes in a
    dict-like fashion, otherwise they are absolute::

        >>> paths = FilePaths(root='/tmp', pairs='pairs.json')
        >>> paths['pairs']
        '/tmp/pairs.json'
        >>> paths.pairs
        'pairs.json'

    """
    root = String('/', desc="root path")
    dest = String(desc="location for writing files to")
    electrode_config_file = String(desc="Odin electrode config CSV file")
    pairs = String(desc="pairs.json")
    excluded_pairs = String(desc="excluded_pairs.json")

    def __getitem__(self, item):
        """Prepends a path with the root path."""
        if item in self.visible_traits() and item != 'root':
            return os.path.join(self.root, getattr(self, item))
        else:
            raise KeyError("No such trait: " + item)


class ExperimentParameters(Schema):
    """Common parameters used in an experiment. Default values apply to the FR
    class of experiments.

    """
    width = Int(5, desc='wavelet width')
    freqs = Array(value=np.logspace(np.log10(6), np.log10(180), 8),
                  desc='frequencies to compute powers for')

    log_powers = Bool(True)  # FIXME: do we really need this?

    filt_order = Int(4, desc="Butterworth filter order")

    penalty_type = String('l2', desc='logistic regression penalty type')
    C = Float('7.2e-4', desc='inverse of regularization strength')


class FRParameters(ExperimentParameters):
    """Free recall experiment parameters relevant for classification."""
    start_time = Float(0, desc="encoding start time [s]")
    end_time = Float(1.366, desc="encoding end time [s]")
    buf = Float(1.365, desc="encoding buffer time [s]")

    retrieval_start_time = Float(-0.525)
    retrieval_end_time = Float(0)
    retrieval_buf = Float(0.524)
