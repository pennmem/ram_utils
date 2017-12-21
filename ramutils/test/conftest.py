import pytest


def pytest_addoption(parser):
    parser.addoption("--rhino-root", action="store", default="/",
                     help="Mount point for RHINO")


@pytest.fixture
def rhino_root(request):
    return request.config.getoption("--rhino-root")


def pytest_generate_tests(metafunc):
    if 'rhino' in metafunc.fixturenames:
        metafunc.parametrize("rhino",
                             rhino_root)
