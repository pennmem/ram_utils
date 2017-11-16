import pytest
import random
import functools

from pkg_resources import resource_filename

from ramutils.tasks import memory
from ramutils.tasks.events import *


datafile = functools.partial(resource_filename, 'ramutils.test.test_data.protocols')


class TestEvents:
    @classmethod
    def setup_class(cls):

    @classmethod
    def teardown_class(cls):
        memory.clear(warn=False)
