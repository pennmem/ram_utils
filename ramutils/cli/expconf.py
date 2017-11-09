"""Command-line script to generate experimental configuration files for
Ramulator.

"""

from argparse import ArgumentParser
import os.path as osp

# FIXME: should this generate record-only configs?
RECORD_ONLY = []  # ['FR1', 'catFR1']
PS = ['PS4_FR5', 'PS4_catFR5']

# Supported experiments
EXPERIMENTS = ['FR5', 'FR6', 'catFR5', 'catFR6'] + RECORD_ONLY + PS

parser = ArgumentParser(description="Generate experiment configs for Ramulator")
parser.add_argument('--dest', '-d', default='.', help='directory to write output to')
parser.add_argument('--subject', '-s', required=True, type=str, help='subject ID')
parser.add_argument('--localization', '-l', default=0, type=int, help='localization number')
parser.add_argument('--montage', '-m', default=0, type=int, help='montage number')
parser.add_argument('--experiment', '-x', required=True, type=str,
                    choices=EXPERIMENTS, help='experiment')
parser.add_argument('--electrode-config-file', '-e', required=True, type=str,
                    help='path to Odin electrode config csv file')
parser.add_argument('--anodes', nargs='+', help='stim anode labels')
parser.add_argument('--cathodes', nargs='+', help='stim cathode labels')
parser.add_argument('--min-amplitudes', nargs='+', help='minimum stim amplitudes')
parser.add_argument('--max-amplitudes', nargs='+', help='maximum stim amplitudes')
parser.add_argument('--target-amplitudes', '-a', nargs='+', help='target stim amplitudes')
parser.add_argument('--pulse-frequencies', '-f', nargs='+', type=float,
                    help='stim pulse frequencies (one to use same value)')
parser.add_argument('--force-rerun', action='store_true', help='force re-running all tasks')


class ValidationError(Exception):
    """Raised when command-line arguments are invalid."""


def validate_stim_settings(args):
    """Check stim settings have the right number of arguments."""
    if args.experiment not in ['FR1', 'catFR1']:
        if not len(args.anodes) == len(args.cathodes):
            raise ValidationError("Number of anodes doesn't match number of cathodes")

        min_max_lengths = len(args.anodes) == len(args.min_amplitudes) == len(args.max_amplitudes)
        target_length = len(args.anodes) == len(args.target_amplitudes)
        if not (min_max_lengths or target_length):
            raise ValidationError("Number of stim contacts doesn't match number of amplitude settings")

        if len(args.pulse_frequencies) == 1:
            args.pulse_frequencies = [args.pulse_frequencies[0]] * len(args.anodes)

        if not len(args.pulse_frequencies) == len(args.anodes):
            raise ValidationError("Number of pulse frequencies doesn't match number of stim contacts")


def main():
    from ramutils.parameters import FilePaths, FRParameters
    from ramutils.pipelines.ramulator_config import make_ramulator_config
    from ramutils.tasks import memory

    args = parser.parse_args()
    validate_stim_settings(args)

    # cache in the same place that we're dumping files to
    memory.cachedir = args.dest
    if args.force_rerun:
        memory.clear()

    paths = FilePaths(
        root=osp.expanduser('/'),
        electrode_config_file=osp.expanduser(args.electrode_config_file),
        dest='scratch/ramutils2'  # FIXME: either always use abs paths or always use relpaths
    )

    # FIXME: rhino root?
    paths.pairs = osp.join(paths.root, 'protocols', 'subjects', args.subject,
                           'localizations', args.localization,
                           'montages', args.montage,
                           'neuroradiology', 'current_processed', 'pairs.json')

    # Get experiment-specific parameters
    # FIXME: add PAL parameters
    if 'FR' in args.experiment:
        exp_params = FRParameters()
    else:
        raise ValidationError("Only FR-like experiments supported so far")

    # Generate!
    make_ramulator_config(args.subject, args.experiment, paths,
                          args.anodes, args.cathodes, exp_params)


if __name__ == "__main__":
    main()
