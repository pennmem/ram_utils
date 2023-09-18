"""Constants that may be used throughout :mod:`ramutils`. Constants that are
defined here include:

* ``EXPERIMENTS`` - a dictionary of all experiments that can be processed by
  :mod:`ramutils`

"""

EXPERIMENTS = {
    'record_only': [
        'FR1',
        'CatFR1',
        'ICatFR1',
        'PAL1',
        'DBOY1',
        'RepFR1',
        'EFRCourierReadOnly'
    ],
    'ps': [
        'PS4_FR5',
        'PS4_CatFR5',
        'PS4_FR',  # reports use different naming convention from config generation
        'PS4_catFR',
        'PS5_FR',
        'PS5_CatFR',
        "LocationSearch",
    ],
    'open_loop': [
        'FR2',
        'CatFR2',
        'ICatFR2',
        'RepFR2',
        'EFRCourierOpenLoop'
    ],
    'closed_loop': [
        'FR3',
        'CatFR3',
        'ICatFR3',
        'PAL3',
        'FR5',
        'CatFR5',
        'ICatFR5',
        'PAL5',
        'FR6',
        'CatFR6',
        'ICatFR6',
        'TICL_FR',
        'TICL_CatFR',
    ],

    # Experiments that allow multiple stim locations
    'multistim': [
        'AmplitudeDetermination',
        'LocationSearch',
        'PS4_FR5',
        'PS4_CatFR5',
        'FR6',
        'CatFR6',
        'ICatFR6',
    ]
}

MTL_LOC_DICT = {
    'Hipp': ['CA1','CA2','CA3','DG','Sub'],
    'MTL': ['PRC', 'EC', 'PHC', 'EC', 'PHC'],
}

DK_LOC_DICT = {
    'IFG': ['parsopercularis', 'parsorbitalis', 'parstriangularis'],
    'MFG': ['caudalmiddlefrontal', 'rostralmiddlefrontal'],
    'SFG': ['superiorfrontal'],
    'TC': ['middletemporal', 'inferiortemporal', 'superiortemporal'],
    'IPC': ['inferiorparietal', 'supramarginal'],
    'SPC': ['superiorparietal', 'precuneus'],
    'OC': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine'],
}
