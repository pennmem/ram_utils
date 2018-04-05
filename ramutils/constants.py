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

MTL_LOC_DICT = {
    'Hipp': ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub',
             'Right CA1', 'Right CA2', 'Right CA3', 'Right DG', 'Right Sub'],
    'MTL': ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC']
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
