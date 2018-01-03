ramutils
========

.. image:: https://travis-ci.org/pennmem/ram_utils.svg?branch=master
    :target: https://travis-ci.org/pennmem/ram_utils

.. image:: https://codecov.io/gh/pennmem/ram_utils/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pennmem/ram_utils

.. image:: https://img.shields.io/badge/docs-here-blue.svg
    :target: https://pennmem.github.io/ram_utils/html/index.html

Bootstrapping a conda environment
---------------------------------

.. code-block:: shell-session

    conda create -y -n environment_name python=3
    conda install -c pennmem --file=requirements.txt

Usage with the RAM_clinical account
-----------------------------------

For generating Ramulator configuration files, the ``RAM_clinical`` account on
``rhino2`` should be used. This account provides a managed conda environment
and should only be updated by a responsible person from the Systems Development
team.

The general workflow for users is the following:

1. Login to rhino using your normal account
2. Use ``sudo`` to open a shell with the ``RAM_clinical`` user: ``sudo -s -u RAM_clinical``
3. Use ``qlogin`` to open an interactive session on an availabel node: ``qlogin``
4. Activate the ``ramutils`` environment: ``source activate ramutils``
5. Run the provided command line scripts

Info for account maintainers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update the account to a new release:

1. ``git fetch``
2. ``git checkout <release tag>``
3. ``maint/conda_update.sh``

Ramulator experiment config generation
--------------------------------------

Ramulator experiment configuration files for stimulation experiments are
generated via the ``ramulator-conf`` script::

    usage: ramulator-conf [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                          --subject SUBJECT [--force-rerun] --experiment
                          {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1}
                          [--vispath VISPATH] [--localization LOCALIZATION]
                          [--montage MONTAGE]
                          [--electrode-config-file ELECTRODE_CONFIG_FILE]
                          [--anodes ANODES [ANODES ...]]
                          [--cathodes CATHODES [CATHODES ...]]
                          [--min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]]
                          [--max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]]
                          [--target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]]
                          [--no-extended-blanking]
                          [--default-area DEFAULT_AREA | --area-file AREA_FILE]
                          [--clear-log]

    Generate experiment configs for Ramulator

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root (default: /)
      --dest DEST, -d DEST  directory to write output to (default:
                            scratch/ramutils)
      --cachedir CACHEDIR   absolute path for caching dir
      --subject SUBJECT, -s SUBJECT
                            subject ID
      --force-rerun         force re-running all tasks
      --experiment {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1}, -x {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --localization LOCALIZATION, -l LOCALIZATION
                            localization number (default: 0)
      --montage MONTAGE, -m MONTAGE
                            montage number (default: 0)
      --electrode-config-file ELECTRODE_CONFIG_FILE, -e ELECTRODE_CONFIG_FILE
                            path to existing electrode config CSV file
      --anodes ANODES [ANODES ...], -a ANODES [ANODES ...]
                            stim anode labels
      --cathodes CATHODES [CATHODES ...], -c CATHODES [CATHODES ...]
                            stim cathode labels
      --min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]
                            minimum stim amplitudes
      --max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]
                            maximum stim amplitudes
      --target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...], -t TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]
                            target stim amplitudes
      --no-extended-blanking
                            disable extended blanking
      --default-area DEFAULT_AREA, -A DEFAULT_AREA
                            default surface area to use for all contacts
      --area-file AREA_FILE
                            path to area.txt file relative to root
      --clear-log           clear the log

Report generation
-----------------

.. warning:: The new reporting pipeline is not fully tested yet, so the old
             version in the ``new_ptsa`` branch may need to be used.

Generating reports on the command line::

    usage: ram-report [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                      --subject SUBJECT [--force-rerun] --experiment
                      {FR1,CatFR1,PAL1,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR6,CatFR6}
                      [--vispath VISPATH] [--sessions SESSIONS [SESSIONS ...]]
                      [--retrain]
                      [--excluded-contacts EXCLUDED_CONTACTS [EXCLUDED_CONTACTS ...]]
                      [--joint-report]

    Generate a report

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root (default: /)
      --dest DEST, -d DEST  directory to write output to (default:
                            scratch/ramutils)
      --cachedir CACHEDIR   absolute path for caching dir
      --subject SUBJECT, -s SUBJECT
                            subject ID
      --force-rerun         force re-running all tasks
      --experiment {FR1,CatFR1,PAL1,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR6,CatFR6}, -x {FR1,CatFR1,PAL1,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR6,CatFR6}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --sessions SESSIONS [SESSIONS ...], -S SESSIONS [SESSIONS ...]
                            sessions to read data from (default: use all)
      --retrain, -R         retrain classifier rather than loading from disk
      --excluded-contacts EXCLUDED_CONTACTS [EXCLUDED_CONTACTS ...], -E EXCLUDED_CONTACTS [EXCLUDED_CONTACTS ...]
                            contacts to exclude from classifier
      --joint-report, -j    include CatFR/FR for FR reports (default: off)

Testing
-------
Automated unit tests are run with every push to the remote repository or pull request. Longer running tests requiring
local files can and should be run frequently, but require to additional argument to be passed to pytest:

--rhino-root: The mount point for RHINO
--output-dest: Where output from blackbox tests will be saved

