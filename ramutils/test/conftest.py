import pytest


def pytest_addoption(parser):
    parser.addoption("--rhino-root", action="store", default="/",
                     help="Mount point for RHINO")
    parser.addoption("--output-dest", action="store",
                     help="Where to store testing output")


@pytest.fixture
def rhino_root(request):
    return request.config.getoption("--rhino-root")


@pytest.fixture
def output_dest(request):
    return request.config.getoption("--output-dest")


def pytest_generate_tests(metafunc):
    if 'rhino' in metafunc.fixturenames:
        metafunc.parametrize("rhino",
                             rhino_root)

    if 'output' in metafunc.fixturenames:
        metafunc.parametrize("output_dest", output_dest)
