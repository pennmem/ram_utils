class TooManySessionsError(Exception):
    """Raised when too many sessions' worth of data are passed as an argument.

    """


class TooManyExperimentsError(Exception):
    """Raised when too many experiments are included with events."""


class UnsupportedExperimentError(Exception):
    """Raised when trying to do something with an experiment that is not yet
    supported.

    """


class MultistimNotAllowedException(Exception):
    """Raised when attempting to define multiple stim sites for an experiment
    which doesn't support it.

    """
