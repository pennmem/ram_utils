from collections import OrderedDict


class DependencyChangeTrackerBase(object):
    """This is meant to be a general API for dependency tracking"""
    def __init__(self,*args,**kwds):
        self.changed_resources = OrderedDict()

    def get_changed_resources(self):
        return self.changed_resources

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

    def is_saved_status_present(self):
        pass


class DependencyInventory(object):
    def __init__(self):
        self.__dependent_resources__ = OrderedDict()

    def add_dependent_resource(self, resource_name, access_path):
        self.__dependent_resources__[resource_name] = access_path

    def __iter__(self):
        for resource_name, access_path in self.__dependent_resources__.items():
            yield resource_name, access_path
