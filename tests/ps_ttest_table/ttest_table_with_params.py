import pandas as pd
import numpy as np
from ptsa.data.readers import TalReader,TalStimOnlyReader

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


mount_point = args.mount_point

def get_elec_data_coords_array(hemi_data):
    dtype_avgSurf = [('x_snap', '<f8'), ('y_snap', '<f8'),('z_snap', '<f8')]

    if hemi_data is not None and len(hemi_data):
        tmp_array = np.array([((hemi_data.avgSurf.x_snap,hemi_data.avgSurf.y_snap,hemi_data.avgSurf.z_snap),hemi_data.eType[0],hemi_data.tagName[0])],
                         dtype=[('avgSurf', dtype_avgSurf),('eType','|S256'),('tagName','|S256')])

        return tmp_array
    else:
        return None



def get_tal_structs_row(subject,anode_tag,cathode_tag):

        # '/Users/m/data/eeg/R1111M/tal/R1111M_talLocs_database_bipol.mat'
        # tal_path = join(mount_point,'data/eeg/',subject,'tal',subject+'_talLocs_database_bipol.mat')
        tal_path = join(mount_point,'data/eeg/',subject,'tal',subject+'_talLocs_database_monopol.mat')
        tal_reader = TalReader(filename=tal_path,struct_name='talStruct')
        tal_structs = tal_reader.read()

        # sel = tal_structs[np.where(tal_structs.tagName == anode_tag+'-'+cathode_tag)]
        #ORIGINAL CODE
        # sel = tal_structs[np.where((tal_structs.tagName == anode_tag)|(tal_structs.tagName == cathode_tag))]

        anode = tal_structs[np.where((tal_structs.tagName == anode_tag)) ]


        cathode = tal_structs[np.where((tal_structs.tagName == cathode_tag))]
        # print cathode
        # sel = np.vstack((anode,cathode))

        # if not len(sel):
        #     # sel = tal_structs[np.where((tal_structs.tagName == anode_tag)|(tal_structs.tagName == cathode_tag))]
        #     anode = tal_structs[np.where((tal_structs.tagName == anode_tag))]
        #     cathode = tal_structs[np.where((tal_structs.tagName == cathode_tag))]
        #     sel = np.vstack((anode, cathode))

        if not len(anode) and not len(cathode):

            tal_path = join(mount_point,'data/eeg/',subject,'tal',subject+'_talLocs_database_stimOnly.mat')
            tal_reader = TalStimOnlyReader(filename=tal_path)
            tal_structs = tal_reader.read()
            # ORIGINAL CODE
            # sel = tal_structs[np.where((tal_structs.tagName == anode_tag)|(tal_structs.tagName == cathode_tag))]

            anode = tal_structs[np.where((tal_structs.tagName == anode_tag))]
            cathode = tal_structs[np.where((tal_structs.tagName == cathode_tag))]
            # sel = np.vstack((anode, cathode))


            # if not len(sel):
            #     sel = tal_structs[np.where((tal_structs.tagName == anode_tag)|(tal_structs.tagName == cathode_tag))]


        anode_ret = anode[0] if len(anode) else None
        cathode_ret = cathode[0] if len(cathode) else None


        return anode_ret, cathode_ret


def extend_elec_dataframe(df):


    tal_structs = None
    subject = ''

    lh_selector = None
    rh_selector = None

    lh_data_combined = None
    rh_data_combined = None


    x0 = np.zeros(shape=(len(df['Subject'])),dtype=np.float)
    y0 = np.zeros(shape=(len(df['Subject'])),dtype=np.float)
    z0 = np.zeros(shape=(len(df['Subject'])),dtype=np.float)


    x1 = np.zeros(shape=(len(df['Subject'])), dtype=np.float)
    y1 = np.zeros(shape=(len(df['Subject'])), dtype=np.float)
    z1 = np.zeros(shape=(len(df['Subject'])), dtype=np.float)

    eType = np.zeros(shape=(len(df['Subject'])),dtype='|S256')

    for count, (index, row) in enumerate(df.iterrows()):
        # print count, index, row

        if subject != row['Subject']:

            subject = row['Subject']
            print subject

        anode, cathode = get_tal_structs_row(subject=subject,anode_tag=row['stimAnodeTag'],cathode_tag=row['stimCathodeTag'])
        # print sel


        if anode is not None:
            x0[count] = anode.avgSurf.x
            y0[count] = anode.avgSurf.y
            z0[count] = anode.avgSurf.z
            eType[count] = anode.eType
        else:

            x0[count] = np.nan
            y0[count] = np.nan
            z0[count] = np.nan

        if cathode is not None:
            x1[count] = cathode.avgSurf.x
            y1[count] = cathode.avgSurf.y
            z1[count] = cathode.avgSurf.z

            eType[count] = cathode.eType
        else:

            x1[count] = np.nan
            y1[count] = np.nan
            z1[count] = np.nan


    df['xAvgSurf_anode'] = x0
    df['yAvgSurf_anode'] = y0
    df['zAvgSurf_anode'] = z0


    df['xAvgSurf_cathode'] = x1
    df['yAvgSurf_cathode'] = y1
    df['zAvgSurf_cathode'] = z1

    df['eType'] = eType

    return df





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


# adding x,y,z electrode coordinates and the type of electrode
ttest_table = extend_elec_dataframe(ttest_table)


ttest_table.to_pickle(join(workspace_dir,'ttest_table_params.pkl'))
ttest_table.to_csv(join(workspace_dir,'ttest_table_params.csv'), index=False)

excel_file_path = join(workspace_dir,'ttest_table_params.xlsx')
ttest_table.to_excel(excel_file_path)


excel_dst = '/protocols/r1/reports/ps_aggregator_significance_table.xlsx'
shutil.copy(excel_file_path,excel_dst)