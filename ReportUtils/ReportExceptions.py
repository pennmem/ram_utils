class MissingExperimentError(Exception):
    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super(MissingExperimentError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

class MissingDataError(Exception):
    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super(MissingDataError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors