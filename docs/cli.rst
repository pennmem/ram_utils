Command-line usage
==================

.. note::

    For generating configuration files in production, always use the
    ``RAM_clinical`` account on rhino.

Ramulator configuration generation
----------------------------------

All stim experiments can have configuration files generated via the unified
``ramulator-conf`` script::

    usage: ramulator-conf [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                          --subject SUBJECT [--force-rerun] --experiment
                          {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6}
                          [--vispath VISPATH] [--localization LOCALIZATION]
                          [--montage MONTAGE] --electrode-config-file
                          ELECTRODE_CONFIG_FILE [--anodes ANODES [ANODES ...]]
                          [--cathodes CATHODES [CATHODES ...]]
                          [--min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]]
                          [--max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]]
                          [--target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]]
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
      --experiment {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6}, -x {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --localization LOCALIZATION, -l LOCALIZATION
                            localization number (default: 0)
      --montage MONTAGE, -m MONTAGE
                            montage number (default: 0)
      --electrode-config-file ELECTRODE_CONFIG_FILE, -e ELECTRODE_CONFIG_FILE
                            path to Odin electrode config csv file
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
      --clear-log           clear the log

.. note::

    All paths are relative to the root path *except* for ``cachedir``. This is
    treated differently due to an issue on macOS and sshfs-mounted filesystems.

Anodes, cathodes, and amplitudes must all be specified in the same order. In
other words, if using stim channels ``LAD8_LAD9``, ``LAH1_LAH2`` with amplitudes
0.5 mA, the relevant options would be given as::

    --anodes LAD8 LAH1 --cathodes LAD9 LAH2 --target-amplitudes 0.5 0.5
