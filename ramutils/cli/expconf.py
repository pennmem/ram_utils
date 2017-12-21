"""Command-line script to generate experimental configuration files for
Ramulator.

"""

from datetime import datetime
import functools
import os.path as osp

from ramutils.cli import make_parser, ValidationError, configure_caching
from ramutils.constants import EXPERIMENTS
from ramutils.log import get_logger, get_warning_accumulator
from ramutils.utils import timer

# Supported experiments
experiments = (
    ['AmplitudeDetermination'] + EXPERIMENTS['ps'] +
    [exp for exp in EXPERIMENTS['closed_loop']] +
    EXPERIMENTS['record_only']
)

parser = make_parser("Generate experiment configs for Ramulator", experiments)
parser.add_argument('--localization', '-l', default=0, type=int, help='localization number (default: 0)')
parser.add_argument('--montage', '-m', default=0, type=int, help='montage number (default: 0)')
parser.add_argument('--electrode-config-file', '-e', type=str, help='path to existing electrode config CSV file')
parser.add_argument('--anodes', '-a', nargs='+', help='stim anode labels')
parser.add_argument('--cathodes', '-c', nargs='+', help='stim cathode labels')
parser.add_argument('--min-amplitudes', nargs='+', type=float, help='minimum stim amplitudes')
parser.add_argument('--max-amplitudes', nargs='+', type=float, help='maximum stim amplitudes')
parser.add_argument('--target-amplitudes', '-t', type=float, nargs='+', help='target stim amplitudes')
parser.add_argument('--no-extended-blanking', action='store_true', help='disable extended blanking')

# This is currently fixed so there is no need for an option
# parser.add_argument('--pulse-frequencies', '-f', type=float, nargs='+',
#                     help='stim pulse frequencies (one to use same value)')

# if we don't find area.txt in the same place as the jacksheet, then look
# at the --area-file option or use a default value
area_group = parser.add_mutually_exclusive_group(required=False)
area_group.add_argument('--default-area', '-A', type=float,
                        help='default surface area to use for all contacts (default: 0.001)')
area_group.add_argument('--area-file', type=str,
                        help='path to area.txt file relative to root')

parser.add_argument('--clear-log', action='store_true', default=False,
                    help='clear the log')

logger = get_logger()


def validate_stim_settings(args):
    """Check stim settings have the right number of arguments."""
    # FIXME: check that stim channels as defined actually exist
    if args.experiment not in EXPERIMENTS['record_only']:
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

        # We're not actually using this as an option, so it's commented out
        # if args.pulse_frequencies is None:
        #     args.pulse_frequencies = [200] * len(args.anodes)
        # elif len(args.pulse_frequencies) == 1:
        #     args.pulse_frequencies = [args.pulse_frequencies[0]] * len(args.anodes)
        #
        # if not len(args.pulse_frequencies) == len(args.anodes):
        #     raise ValidationError("Number of pulse frequencies doesn't match number of stim contacts")


def main(input_args=None):
    from ramutils.montage import make_stim_params
    from ramutils.parameters import FilePaths, FRParameters, PALParameters
    from ramutils.pipelines.ramulator_config import make_ramulator_config

    warning_accumulator = get_warning_accumulator()

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

    mode = 'w' if args.clear_log else 'a'
    with open(osp.expanduser('~/.ramutils_expconf.log'), mode) as f:
        f.write(datetime.now().strftime('[%Y-%m-%dT%H:%M:%S]\n'))
        f.write("ramulator-conf \\\n")
        f.write('\\ \n'.join(output))  # add backslashes to allow copy-paste
        f.write('\n\n')

    paths_kwargs = {
        'root': osp.expanduser(args.root),
        'dest': args.dest,
    }

    # Override generating electrode config files if we specify -e
    if args.electrode_config_file is not None:
        paths_kwargs['electrode_config_file'] = args.electrode_config_file

    paths = FilePaths(**paths_kwargs)

    # FIXME: figure out why MacOS won't work with sshfs-relative paths only here
    logger.info("Using %s as cache dir", args.cachedir)
    configure_caching(args.cachedir, args.force_rerun)

    paths.pairs = osp.join(paths.root, 'protocols', 'subjects', args.subject,
                           'localizations', str(args.localization),
                           'montages', str(args.montage),
                           'neuroradiology', 'current_processed', 'pairs.json')

    # Determine params based on experiment type
    if args.experiment == 'AmplitudeDetermination':
        exp_params = None
    elif "FR" in args.experiment:
        exp_params = FRParameters()
    elif "PAL" in args.experiment:
        exp_params = PALParameters()
    else:
        raise RuntimeError("Somehow we got an unsupported experiment")

    # Construct stim parameters
    if args.experiment not in EXPERIMENTS['record_only']:
        # FIXME: explicitly check given arguments to provide more helpful error messages when given malformed args
        if args.min_amplitudes is not None:
            make_stim_params = functools.partial(make_stim_params,
                                                 min_amplitudes=args.min_amplitudes,
                                                 max_amplitudes=args.max_amplitudes)
        else:
            make_stim_params = functools.partial(make_stim_params,
                                                 target_amplitudes=args.target_amplitudes)
        stim_params = make_stim_params(args.subject, args.anodes, args.cathodes, root=paths.root)
    else:
        stim_params = []

    # Override area file path if necessary...
    if args.area_file is not None:
        paths.area_file = osp.join(paths.root, args.area_file)

    # ... or set default surface area
    default_surface_area = 0.001 if args.default_area is None else args.default_area

    # Generate!
    with timer():
        make_ramulator_config(args.subject, args.experiment, paths, stim_params,
                              exp_params, args.vispath,
                              extended_blanking=(not args.no_extended_blanking),
                              localization=args.localization,
                              montage=args.montage,
                              default_surface_area=default_surface_area)

    warnings = '\n' + warning_accumulator.format_all()
    if warnings is not None:
        logger.info(warnings)


if __name__ == "__main__":  # pragma: nocover
    root = "~/mnt/rhino"
    dest = "scratch/ramutils2/demo"

    # main([
    #     "-s", "R1347D", "-x", "CatFR1", "-A", "0.5",
    #     "--root", root, "--dest", dest, "--force-rerun"
    # ])

    # main([
    #     "-s", "R1364C", "-x", "AmplitudeDetermination",
    #     "--anodes", "AMY7", "--cathodes", "AMY8",
    #     "--min-amplitudes", "0.1", "--max-amplitudes", "1.0",
    #     "--root", root, "--dest", dest, "--force-rerun"
    # ])

    main([
        "-s", "R1374T", "-x", "CatFR5",
        "--anodes", "LA7", "--cathodes", "LA8",
        "--target-amplitudes", "0.5",
        "-e", "/data/eeg/R1374T/behavioral/catFR5/session_0/host_pc/20171212_163330/config_files/R1374T_12DEC2017L0M0STIM.csv",
        "--root", root, "--dest", dest, "--force-rerun"
    ])

    # main([
    #     "-s", "R1365N", "-x", "PAL5",
    #     "--anodes", "LAD12", "--cathodes", "LAD13",
    #     "--target-amplitudes", "0.5",
    #     "--root", root, "--dest", "scratch/ramutils2/demo", "--force-rerun"
    # ])

    # main([
    #     "-s", "R1364C", "-x", "PS4_FR5",
    #     "--anodes", "AMY7", "TOJ7", "--cathodes", "AMY8", "TOJ8",
    #     "--min-amplitudes", "0.1", "0.1", "--max-amplitudes", "1.0", "0.5",
    #     "--root", root, "--dest", "scratch/ramutils2/demo", "--force-rerun"
    # ])

    # main([
    #     "-s", "R1364C", "-x", "FR6",
    #     "--anodes", "AMY7", "TOJ7", "--cathodes", "AMY8", "TOJ8",
    #     "--target-amplitudes", "0.5", "0.75",
    #     "--root", root, "--dest", "scratch/ramutils2/demo", "--force-rerun"
    # ])

