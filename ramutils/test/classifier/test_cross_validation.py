import functools
from pkg_resources import resource_filename
import pytest


datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


@pytest.mark.skip(reason='not implemented')
class TestCrossValidation:
    def test_perform_lolo_cross_validation(self):
        return

    def test_perform_loso_cross_validation(self):
        return

    def test_permuted_lolo_cross_validation(self):
        return

    def test_permuted_loso_cross_validation(self):
        return
