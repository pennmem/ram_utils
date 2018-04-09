"""Constants that may be used throughout :mod:`ramutils`. Constants that are
defined here include:

* ``EXPERIMENTS`` - a dictionary of all experiments that can be processed by
  :mod:`ramutils`

"""

EXPERIMENTS = {
    'record_only': [
        'FR1',
        'CatFR1',
        'PAL1',
        'DBOY1',
    ],
    'ps': [
        'PS4_FR5',
        'PS4_CatFR5',
        'PS4_FR', # reports use different naming convention from config generation
        'PS4_catFR',
        'PS5_FR',
        'PS5_CatFR',
    ],
    'open_loop': [
        'FR2',
        'CatFR2'
    ],
    'closed_loop': [
        'FR3',
        'CatFR3',
        'PAL3',
        'FR5',
        'CatFR5',
        'PAL5',
        'FR6',
        'CatFR6',
    ],

    # Experiments that allow multiple stim locations
    'multistim': [
        'AmplitudeDetermination',
        'PS4_FR5',
        'PS4_CatFR5',
        'FR6',
        'CatFR6',
    ]
}
