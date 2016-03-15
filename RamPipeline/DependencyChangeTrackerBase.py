

class DependencyChangeTrackerBase(object):
    '''
    This is meant to be a general API for dependency tracking
    '''
    def __init__(self,*args,**kwds):
        pass

    def initialize(self):
        pass

    def generate_latest_data_status(self):
        pass

    def get_latest_data_status(self):
        pass

    def read_saved_data_status(self):
        pass

    def get_saved_data_status(self):
        pass

    def write_latest_data_status(self):
        pass

    def check_dependency_change(self, task):
        pass

