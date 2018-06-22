import os
import sys

import pytest

here = os.path.abspath(os.path.dirname(__file__))
args = [here] + sys.argv[1:]
sys.exit(pytest.main(args))
