"""Ramutils global configuration. To update user settings, create a
``.ramutils/user_settings.json`` file in your user's home directory. Any
settings not defined there will instead use the defaults defined here.

"""

import json
import os.path
import tempfile

_DEFAULT_FILE = os.path.expanduser("~/.ramutils/user_settings.json")


PATHS = {
    # Root path to base relative paths on
    'root': '/',

    # Directory for caching results to
    'cachedir': tempfile.gettempdir(),

    # Directory to store logs to
    'logdir': os.path.expanduser('~/.ramutils/logs')
}


def load_user_settings(path=_DEFAULT_FILE):
    """Load user settings from a JSON file.

    Keyword arguments
    -----------------
    path : str
        Path to ``user_settings.json``.

    """
    global PATHS

    if os.path.exists(path):
        with open(path, 'r') as f:
            config = json.loads(f.read())

        try:
            PATHS.update(config['PATHS'])
        except KeyError:
            pass


def save_user_settings(path=_DEFAULT_FILE):
    """Save user settings as a JSON file.

    Keyword arguments
    -----------------
    path : str
        Path to ``user_settings.json``.

    """
    global PATHS

    with open(path, 'w') as f:
        json.dump({
            'PATHS': PATHS,
        }, f)
