import pandas as pd
from scipy.stats import ttest_1samp

import sys
from setup_utils import parse_command_line, configure_python_paths
from os.path import *

import shutil

# -------------------------------processing command line
if len(sys.argv)>2:

    args = parse_command_line()


else: # emulate command line
    # command_line_emulation_argument_list = ['--workspace-dir','/scratch/busygin/ps_aggregator',
    #                                         '--mount-point','',
    #                                         '--python-path','/home1/busygin/ram_utils_new_ptsa'
    #                                         ]

    command_line_emulation_argument_list = ['--workspace-dir', '/scratch/mswat/automated_reports',
                                            '--mount-point', ''
                                            ]

    args = parse_command_line(command_line_emulation_argument_list)

configure_python_paths(args.python_path)






workspace_dir = args.workspace_dir


ps_table = pd.read_pickle(join(workspace_dir,'ps_table.pkl'))
ps3_table = pd.read_pickle(join(workspace_dir,'ps3_table.pkl'))

ps_table = ps_table[['Subject','Pulse_Frequency','Amplitude','Duration','Burst_Frequency','stimAnodeTag','stimCathodeTag','locTag','perf_diff']]
ps3_table = ps3_table[['Subject','Pulse_Frequency','Amplitude','Duration','Burst_Frequency','stimAnodeTag','stimCathodeTag','locTag','perf_diff']]

ps_table = pd.concat([ps_table, ps3_table], ignore_index=True)

grouped = ps_table.groupby(['Subject','Pulse_Frequency', 'Amplitude', 'Duration', 'Burst_Frequency','stimAnodeTag','stimCathodeTag','locTag'])

ttest_table = []

for params,ps_subtable in grouped:
    ttest_result = ttest_1samp(ps_subtable['perf_diff'].values, 0.0)
    ttest_table.append(list(params)+[len(ps_subtable),ttest_result.pvalue,ttest_result.statistic])

ttest_table = pd.DataFrame(ttest_table, columns=['Subject','Pulse_Frequency','Amplitude','Duration','Burst_Frequency','stimAnodeTag','stimCathodeTag','locTag','N','p','t'])

ttest_table['abst'] = ttest_table['t'].abs()
ttest_table = ttest_table.sort('abst', ascending=False)
del ttest_table['abst']

ttest_table.to_pickle(join(workspace_dir,'ttest_table_params.pkl'))
ttest_table.to_csv(join(workspace_dir,'ttest_table_params.csv'), index=False)

excel_file_path = join(workspace_dir,'ttest_table_params.xlsx')
ttest_table.to_excel(excel_file_path)


excel_dst = '/protocols/r1/reports/ps_aggregator_significance_table.xlsx'
shutil.copy(excel_file_path,excel_dst)