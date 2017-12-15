import functools

from pkg_resources import resource_filename

from ramutils.tasks import memory


datafile = functools.partial(resource_filename, 'ramutils.test.test_data.protocols')


# TODO Call build_test_data, build_training_data etc.
class TestEvents:
    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        memory.clear(warn=False)
