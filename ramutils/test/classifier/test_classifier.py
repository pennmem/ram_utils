import os
import pytest
from ramutils.classifier.utils import reload_classifier


TEST_DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/test_data/"


@pytest.mark.skip(reason="does nothing")
def test_reload_classifier():
    return
