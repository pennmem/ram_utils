class RamException(Exception):
    """Base exception class for custom exceptions."""


class TooManySessionsError(RamException):
    """Raised when too many sessions' worth of data are passed as an argument.

    """


class TooManyExperimentsError(RamException):
    """Raised when too many experiments are included with events."""


class UnsupportedExperimentError(RamException):
    """Raised when trying to do something with an experiment that is not yet
    supported.

    """


class DataLoadingError(RamException):
    """ Raised when when unable to load expected files"""

# FIXME: refactor naming


class MultistimNotAllowedException(RamException):
    """Raised when attempting to define multiple stim sites for an experiment
    which doesn't support it.

    """


# FIXME: refactor naming
class UnableToReloadClassifierException(RamException):
    """Raised when processing cannot load or use the actual classifier from
    a session.

    """


class MissingFileError(RamException):
    """Raised when a required file doesn't appear to exist."""


class CommandLineError(RamException):
    """Raised when there are CLI-related errors."""


class MissingArgumentsError(RamException):
    """Raised when an optional argument is not optional for certain cases but
    is not specified.

    Example: when not giving trigger pairs for PS5 experiments.

    """


class ValidationError(RamException):
    """Raised when validation checks fail."""


class RetrievalBaselineError(RamException):
    """
    Raised when something goes wrong with adding baseline events during
     the retrieval period
    """
