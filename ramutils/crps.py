import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd

import ramutils.pandas_to_pybeh as pb

pd.set_option('display.max_columns', 1000)
import warnings

def lag_CRP(full_evs):
    full_evs['itemno'] = full_evs['item'].astype('category').cat.codes
    
    
    pd.set_option('display.max_rows', 100)
    
    
    crp_df = full_evs.groupby(['subject']).apply(
        pb.pd_crp,
        itemno_column='itemno',
        list_index=['subject', 'session', 'trial'],
        lag_num=5).reset_index()
    
    return crp_df

def spatial_CRP(full_evs):
    spatial_crp = full_evs.groupby(['subject']).apply(pb.pd_sem_crp_list_sub, sim_columns=['storeX', 'storeZ'],
                                                  list_index=['subject', 'session', 'trial'], 
                                                  bins=[12.96160383, 41.99123618, 61.53639199, 77.73374529, 97.11224258]).reset_index()
    return spatial_crp



