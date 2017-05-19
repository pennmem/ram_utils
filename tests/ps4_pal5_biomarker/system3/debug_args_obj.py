

from ps4_pal5_prompt import  Args
from os.path import *
import sys

if sys.platform.startswith('win'):

    prefix = 'd:/'

else:

    prefix = '/'

args_list = []

# ------------------------------------------------- R1250N
args_obj_R1250N = Args()
args_obj_R1250N.subject = 'R1250N'
args_obj_R1250N.anodes = ['PG10', 'PG11']
args_obj_R1250N.cathodes = ['PG11', 'PG12']
args_obj_R1250N.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv' % args_obj_R1250N.subject)
args_obj_R1250N.experiment = 'PS4_CatFR5'
args_obj_R1250N.min_amplitudes = [0.25, 0.25]
args_obj_R1250N.max_amplitudes = [1.0, 1.0]
args_obj_R1250N.mount_point = prefix
args_obj_R1250N.pulse_frequency = 200
args_obj_R1250N.workspace_dir = None
args_obj_R1250N.allow_fast_rerun = True


# ------------------------------------------------- R1095N
args_obj_R1095N = Args()
args_obj_R1095N.subject = 'R1095N'
args_obj_R1095N.anodes = ['RTT1', 'RTT3']
args_obj_R1095N.cathodes = ['RTT2', 'RTT4']
args_obj_R1095N.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1095N.subject)
args_obj_R1095N.experiment = 'PS4_CatFR5'
args_obj_R1095N.min_amplitudes = [0.25,0.25]
args_obj_R1095N.max_amplitudes = [1.0,1.0]
args_obj_R1095N.mount_point = prefix
args_obj_R1095N.pulse_frequency = 200
args_obj_R1095N.allow_fast_rerun = True

# ------------------------------------------------- R1284N
args_obj_R1284N = Args()
args_obj_R1284N.subject = 'R1284N'
args_obj_R1284N.anodes = ['LMD1', 'LMD3']
args_obj_R1284N.cathodes = ['LMD2','LMD4']
args_obj_R1284N.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1284N.subject)
args_obj_R1284N.experiment = 'PS4_CatFR5'
args_obj_R1284N.min_amplitudes = [0.25,0.25]
args_obj_R1284N.max_amplitudes = [1.0,1.0]
args_obj_R1284N.mount_point = prefix
args_obj_R1284N.pulse_frequency = 200
args_obj_R1095N.allow_fast_rerun = True


# ------------------------------------------------- R1002P
args_obj_R1002P = Args()
args_obj_R1002P.subject = 'R1002P'
args_obj_R1002P.anodes = ['LPF1', 'LPF3']
args_obj_R1002P.cathodes = ['LPF2','LPF4']
args_obj_R1002P.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1002P.subject)
args_obj_R1002P.experiment = 'PS4_CatFR5'
args_obj_R1002P.min_amplitudes = [0.25,0.25]
args_obj_R1002P.max_amplitudes = [1.0,1.0]
args_obj_R1002P.mount_point = prefix
args_obj_R1002P.pulse_frequency = 200
args_obj_R1095N.allow_fast_rerun = True

# ------------------------------------------------- R1065J
args_obj_R1065J = Args()
args_obj_R1065J.subject = 'R1065J'
args_obj_R1065J.anodes = ['LS1', 'LS3']
args_obj_R1065J.cathodes = ['LS2', 'LS4']
args_obj_R1065J.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1065J.subject)
args_obj_R1065J.experiment = 'PS4_CatFR5'
args_obj_R1065J.min_amplitudes = [0.25,0.25]
args_obj_R1065J.max_amplitudes = [1.0,1.0]
args_obj_R1065J.mount_point = prefix
args_obj_R1065J.pulse_frequency = 200
args_obj_R1095N.allow_fast_rerun = True


# ------------------------------------------------- R1162N
args_obj_R1162N = Args()
args_obj_R1162N.subject = 'R1162N'
args_obj_R1162N.anodes = ['G11', 'G13']
args_obj_R1162N.cathodes = ['G12', 'G14']
args_obj_R1162N.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1162N.subject)
args_obj_R1162N.experiment = 'PS4_CatFR5'
args_obj_R1162N.min_amplitudes = [0.25,0.25]
args_obj_R1162N.max_amplitudes = [1.0,1.0]
args_obj_R1162N.mount_point = prefix
args_obj_R1162N.pulse_frequency = 200
args_obj_R1162N.workspace_dir = join(prefix, 'scratch', args_obj_R1162N.subject)


# ------------------------------------------------- R1175N
args_obj_R1175N = Args()
args_obj_R1175N.subject = 'R1175N'
args_obj_R1175N.anodes = ['LAT1', 'LAT3']
args_obj_R1175N.cathodes = ['LAT2','LAT4']
args_obj_R1175N.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1175N.subject)
args_obj_R1175N.experiment = 'PS4_CatFR5'
args_obj_R1175N.min_amplitudes = [0.25,0.25]
args_obj_R1175N.max_amplitudes = [1.0,1.0]
args_obj_R1175N.mount_point = prefix
args_obj_R1175N.pulse_frequency = 200
args_obj_R1175N.workspace_dir = join(prefix, 'scratch', args_obj_R1175N.subject)

# ------------------------------------------------- R1212P
args_obj_R1212P = Args()
args_obj_R1212P.subject = 'R1212P'
args_obj_R1212P.anodes = ['LXB1', 'LXB3']
args_obj_R1212P.cathodes = ['LXB2','LXB4']
args_obj_R1212P.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1212P.subject)
args_obj_R1212P.experiment = 'PS4_CatFR5'
args_obj_R1212P.min_amplitudes = [0.25,0.25]
args_obj_R1212P.max_amplitudes = [1.0,1.0]
args_obj_R1212P.mount_point = prefix
args_obj_R1212P.pulse_frequency = 200
args_obj_R1212P.workspace_dir = join(prefix, 'scratch', args_obj_R1212P.subject)


    
# ------------------------------------------------- R1232N    
args_obj_R1232N = Args()
args_obj_R1232N.subject = 'R1232N'
args_obj_R1232N.anodes = ['LAT1', 'LAT3']
args_obj_R1232N.cathodes = ['LAT2','LAT4']
args_obj_R1232N.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj_R1232N.subject)
args_obj_R1232N.experiment = 'PS4_CatFR5'
args_obj_R1232N.min_amplitudes = [0.25,0.25]
args_obj_R1232N.max_amplitudes = [1.0,1.0]
args_obj_R1232N.mount_point = prefix
args_obj_R1232N.pulse_frequency = 200
args_obj_R1232N.workspace_dir = join(prefix, 'scratch', args_obj_R1232N.subject)




# args_obj_R1250N.classifier_type_to_output = 'pal'
# args_list.append(args_obj_R1250N)
args_list.append(args_obj_R1095N)
# args_list.append(args_obj_R1284N)
# args_list.append(args_obj_R1002P)
# args_list.append(args_obj_R1065J)
# args_list.append(args_obj_R1162N)
# args_list.append(args_obj_R1175N)
# args_list.append(args_obj_R1212P)
# args_list.append(args_obj_R1232N)



#
# # messed up localization
# # args_obj = Args()
# #
# # args_obj.subject = 'R1118N'
# # args_obj.anodes = ['G11', 'G13']
# # args_obj.cathodes = ['G12', 'G14']
# # args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
# # args_obj.experiment = 'PS4_PAL5'
# # args_obj.min_amplitudes = [0.25,0.25]
# # args_obj.max_amplitudes = [1.0,1.0]
# # args_obj.mount_point = prefix
# # args_obj.pulse_frequency = 200
# # args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
#
# # messed up data
# args_obj = Args()
#
# args_obj.subject = 'R1121M'
# args_obj.anodes = ['RFG1', 'RFG3']
# args_obj.cathodes = ['RFG2', 'RFG4']
# args_obj.electrode_config_file = join(prefix, 'experiment_configs', 'contacts%s.csv'%args_obj.subject)
# args_obj.experiment = 'PS4_PAL5'
# args_obj.min_amplitudes = [0.25,0.25]
# args_obj.max_amplitudes = [1.0,1.0]
# args_obj.mount_point = prefix
# args_obj.pulse_frequency = 200
# args_obj.workspace_dir = join(prefix, 'scratch', args_obj.subject)
#
# args_list.append(args_obj)