Command-line usage
==================

.. note::

    For generating configuration files in production, always use the
    ``RAM_clinical`` account on rhino.

.. note::

    Except for ``cachedir``, which is treated differently due to an issue on macOS and sshfs-mounted filesystems,

    .. class:: center

    All paths are relative to the root path

Ramulator configuration generation
----------------------------------

Ramulator experiment configuration files for stimulation experiments are
generated via the ``ramulator-conf`` script::

    usage: ramulator-conf [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                          --subject SUBJECT [--use-cached] --experiment
                          {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1,DBOY1}
                          [--vispath VISPATH] [--version]
                          [--localization LOCALIZATION] [--montage MONTAGE]
                          [--electrode-config-file ELECTRODE_CONFIG_FILE]
                          [--anodes ANODES [ANODES ...]]
                          [--cathodes CATHODES [CATHODES ...]]
                          [--min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]]
                          [--max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]]
                          [--target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]]
                          [--trigger-pairs TRIGGER_PAIRS [TRIGGER_PAIRS ...]]
                          [--no-extended-blanking] [--use-common-reference]
                          [--default-area DEFAULT_AREA | --area-file AREA_FILE]
                          [--clear-log]

    Generate experiment configs for Ramulator

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root (default: /)
      --dest DEST, -d DEST  directory to write output to relative to ROOT
                            (default: scratch/ramutils)
      --cachedir CACHEDIR   absolute path for caching dir
      --subject SUBJECT, -s SUBJECT
                            subject ID
      --use-cached          allow cached results from previous run to be reused
      --experiment {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1,DBOY1}, -x {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1,DBOY1}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --version             show program's version number and exit
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
      --trigger-pairs TRIGGER_PAIRS [TRIGGER_PAIRS ...]
                            underscore separated trigger electrode pairs (e.g.,
                            LA1_LA2)
      --no-extended-blanking
                            disable extended blanking
      --use-common-reference, -R
                            generate common reference electrode config instead of
                            bipolar
      --default-area DEFAULT_AREA, -A DEFAULT_AREA
                            surface area to use for all contacts (default: 0.001)
      --area-file AREA_FILE
                            path to area.txt file relative to root
      --clear-log           clear the log


Anodes, cathodes, and amplitudes must all be specified in the same order. In
other words, if using stim channels ``LAD8_LAD9``, ``LAH1_LAH2`` with amplitudes
0.5 mA, the relevant options would be given as::

    --anodes LAD8 LAH1 --cathodes LAD9 LAH2 --target-amplitudes 0.5 0.5

Specifying surface areas
~~~~~~~~~~~~~~~~~~~~~~~~

Surface areas for contacts are defined in an ``area.txt`` file which by default
should live in the same directory as ``jacksheet.txt`` (for example,
``/data/eeg/R1374T/docs`` on ``rhino``). The format for this file is::

    <lead label 1> <surface area in mm**2>
    <lead label 2> <surface area in mm**2>
    <...>

where "lead labels" are the label for a contact preceding the contact number.
This assumes that contacts are labeled in such a way that a label is unique for
a given sized contact. In other words, for combined macro/micro contacts, the
macro contacts **must** be labeled differently than the micro contacts. An
example ``area.txt`` file for subject R1347D would look like::

    ROFD 6.1839
    LOFD 6.1839
    RAD 6.1839
    LAD 6.1839
    RAHCD 6.1839
    LAHCD 6.1839
    RPHCD 6.1839
    LPHCD 6.1839
    RID 6.1839
    LID 6.1839
    RMCD 6.1839
    LMCD 6.1839
    RPTD 6.1839
    LPTD 6.1839
    RACD 6.1839
    LACD 6.1839

.. note::

    Alternatively in this case, the ``--default-surface-area`` (or ``-A``)
    option could be used since all contacts share the same surface area.

The logic for determining what to use for surface areas is as follows:

* Use the file specified with the ``--area-file`` option if given
* If not found, use the default surface area (0.001 if not specified, otherwise
  the value specified with the ``--default-surface-area`` option)
* If no ``--area-file`` option is given, try to load ``area.txt`` from the docs
  directory
* If no area file is found, use the default surface area as described above

Troubleshooting
~~~~~~~~~~~~~~~

**Dimensions in powers don't seem to match**

Sometimes, there might be an error such as this:

    IndexError: boolean index did not match indexed array along dimension 1;
    dimension is 170 but corresponding boolean dimension is 168

This is likely caused by trying to autogenerate an electrode config file which
doesn't match with what was actually used in experiments. The workaround is to
explicitly pass an electrode config file that is generated manually with the
``--electrode-config-file`` option.


Report generation
-----------------

Reports are generated with the ``ram-report`` command-line script::

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
      --dest DEST, -d DEST  directory to write output to relative to ROOT
                            (default: scratch/ramutils)
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
