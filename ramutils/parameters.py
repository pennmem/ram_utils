"""Common experimental/model parameters."""

import numpy as np
from traits.api import Int, Float, Bool, Array, String
from ramutils.schema import Schema


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
