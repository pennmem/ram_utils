"""Command-line script to generate experimental configuration files for
Ramulator.

"""

from datetime import datetime
import os.path as osp

from ramutils.cli import make_parser, ValidationError, configure_caching
from ramutils.constants import EXPERIMENTS
from ramutils.log import get_logger
from ramutils.utils import timer

# Supported experiments
# FIXME: ensure PAL support
experiments = ['AmplitudeDetermination'] + EXPERIMENTS['ps'] + \
              [exp for exp in EXPERIMENTS['closed_loop'] if 'PAL' not in exp]

parser = make_parser("Generate experiment configs for Ramulator", experiments)
parser.add_argument('--localization', '-l', default=0, type=int, help='localization number')
parser.add_argument('--montage', '-m', default=0, type=int, help='montage number')
parser.add_argument('--electrode-config-file', '-e', required=True, type=str,
                    help='path to Odin electrode config csv file')
parser.add_argument('--anodes', nargs='+', help='stim anode labels')
parser.add_argument('--cathodes', nargs='+', help='stim cathode labels')
parser.add_argument('--min-amplitudes', nargs='+', help='minimum stim amplitudes')
parser.add_argument('--max-amplitudes', nargs='+', help='maximum stim amplitudes')
parser.add_argument('--target-amplitudes', '-a', nargs='+', help='target stim amplitudes')
parser.add_argument('--pulse-frequencies', '-f', nargs='+', type=float,
                    help='stim pulse frequencies (one to use same value)')

logger = get_logger()


def validate_stim_settings(args):
    """Check stim settings have the right number of arguments."""
    # FIXME: check that stim channels as defined actually exist
    if args.experiment not in ['FR1', 'catFR1']:
        if not len(args.anodes) == len(args.cathodes):
            raise ValidationError("Number of anodes doesn't match number of cathodes")

        if args.experiment != "AmplitudeDetermination" and 'PS4' not in args.experiment:
            if args.target_amplitudes is None:
                raise RuntimeError("--target-amplitudes is required")
            valid = len(args.anodes) == len(args.target_amplitudes)
        else:
            valid = len(args.anodes) == len(args.min_amplitudes) == len(args.max_amplitudes)

        if not valid:
            raise ValidationError("Number of stim contacts doesn't match number of amplitude settings")

        if args.pulse_frequencies is None:
            args.pulse_frequencies = [200] * len(args.anodes)
        elif len(args.pulse_frequencies) == 1:
            args.pulse_frequencies = [args.pulse_frequencies[0]] * len(args.anodes)

        if not len(args.pulse_frequencies) == len(args.anodes):
            raise ValidationError("Number of pulse frequencies doesn't match number of stim contacts")


def main(input_args=None):
    from ramutils.parameters import FilePaths, FRParameters
    from ramutils.pipelines.ramulator_config import make_ramulator_config

    args = parser.parse_args(input_args)
    validate_stim_settings(args)

    # Write options to log file for reference
    # n.b., this relies on using the default naming for variables passed via
    # argparse; i.e., don't use the dest kwarg when defining options!
    output = []
    for arg in vars(args):
        value = getattr(args, arg)
        if value is None:
            continue

        if isinstance(value, list):
            value = ' '.join([str(v) for v in value])
        elif isinstance(value, bool):
            value = ''

        clarg = arg.replace('_', '-')
        output.append('--{} {}'.format(clarg, value))

    with open(osp.expanduser('~/.ramutils_expconf.log'), 'a') as f:
        f.write(datetime.now().strftime('[%Y-%m-%dT%H:%M:%S]\n'))
        f.write("ramulator-conf \\\n")
        f.write('\\\n'.join(output))  # add backslashes to allow copy-paste
        f.write('\n\n')

    paths = FilePaths(
        root=osp.expanduser(args.root),
        electrode_config_file=osp.expanduser(args.electrode_config_file),
        dest=args.dest
    )

    # FIXME: figure out why MacOS won't work with sshfs-relative paths only here
    cachedir = osp.join(args.cachedir, 'cache')
    logger.info("Using %s as cache dir", cachedir)
    configure_caching(cachedir, args.force_rerun)

    paths.pairs = osp.join(paths.root, 'protocols', 'subjects', args.subject,
                           'localizations', str(args.localization),
                           'montages', str(args.montage),
                           'neuroradiology', 'current_processed', 'pairs.json')

    # Determine params based on experiment type
    if args.experiment == 'AmplitudeDetermination':
        params = None
    elif "FR" in args.experiment:
        params = FRParameters()
    else:
        raise RuntimeError("FIXME: support more than FR")

    # Generate!
    with timer():
        make_ramulator_config(args.subject, args.experiment, paths,
                              args.anodes, args.cathodes, params, args.vispath)


if __name__ == "__main__":
    # main([
    #     "-s", "R1364C", "-x", "CatFR5",
    #     "-e", "scratch/system3_configs/ODIN_configs/R1364C/R1364C_06NOV2017L0M0STIM.csv",
    #     "--anodes", "AMY7", "--cathodes", "AMY8",
    #     "--target-amplitudes", "0.5",
    #     "--root", "~/mnt/rhino", "--dest", "scratch/ramutils2/demo", "--force-rerun"
    # ])

    # main([
    #     "-s", "R1364C", "-x", "AmplitudeDetermination",
    #     "-e", "scratch/system3_configs/ODIN_configs/R1364C/R1364C_06NOV2017L0M0STIM.csv",
    #     "--anodes", "AMY7", "--cathodes", "AMY8",
    #     "--min-amplitudes", "0.1", "--max-amplitudes", "1.0",
    #     "--root", "~/mnt/rhino", "--dest", "scratch/ramutils2/demo", "--force-rerun"
    # ])

    main([
        "-s", "R1364C", "-x", "PS4_FR5",
        "-e", "scratch/system3_configs/ODIN_configs/R1364C/R1364C_06NOV2017L0M0STIM.csv",
        "--anodes", "AMY7", "TOJ7", "--cathodes", "AMY8", "TOJ8",
        "--min-amplitudes", "0.1", "0.1", "--max-amplitudes", "1.0", "0.5",
        "--root", "~/mnt/rhino", "--dest", "scratch/ramutils2/demo", "--force-rerun"
    ])
