from collections import OrderedDict


class DependencyInventory(object):
    def __init__(self):
        self.__dependent_resources__ = OrderedDict()

    def add_dependent_resource(self, resource_name, access_path):
        self.__dependent_resources__[resource_name] = access_path

    def __iter__(self):
        for resource_name, access_path in self.__dependent_resources__.items():
            yield resource_name, access_path
