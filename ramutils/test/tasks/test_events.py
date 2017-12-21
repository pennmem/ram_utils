import pytest
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

    def test_build_training_data_regression(self):
        # TODO: Develop test cases for event processing that happens as part
        # of config generation and classifier retraining
        return

    def test_build_test_data_regression(self):
        # TODO: Develop test cases for event processing that happens for
        # building reports
        return
