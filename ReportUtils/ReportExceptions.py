import weakref

class MissingExperimentError(Exception):
    def __init__(self, message, errors=None,status=None):

        # Call the base class constructor with the parameters it needs
        super(MissingExperimentError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors
        self.status = weakref.ref(status)
        if status:
            status.add_exception(self)

class MissingDataError(Exception):
    def __init__(self, message, errors=None,status=None):


        # Call the base class constructor with the parameters it needs
        super(MissingDataError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors
        self.status = weakref.ref(status)
        if status:
            status.add_exception(self)
