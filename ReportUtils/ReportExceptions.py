import weakref

class ReportError(Exception):
    def __init__(self, message, errors=None,status=None):

        # Call the base class constructor with the parameters it needs
        super(ReportError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors
        self.status = None

        if status:
            status.add_exception(self)
            self.status = weakref.ref(status)

class MissingExperimentError(ReportError):
    def __init__(self, message, errors=None,status=None):

        # Call the base class constructor with the parameters it needs
        super(MissingExperimentError, self).__init__(message=message, errors=errors,status=status)

class MissingDataError(ReportError):
    def __init__(self, message, errors=None,status=None):

        # Call the base class constructor with the parameters it needs
        super(MissingDataError, self).__init__(message=message, errors=errors,status=status)


class NumericalError(ReportError):
    def __init__(self, message, errors=None,status=None):

        # Call the base class constructor with the parameters it needs
        super(NumericalError, self).__init__(message=message, errors=errors,status=status)
