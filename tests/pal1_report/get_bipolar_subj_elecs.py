import numpy as np
from glob import glob
from scipy.io import loadmat

def get_bipolar_subj_elecs(
        subjpath, leadsonly=True, exclude_bad_leads=False,
        taldir='/tal', bipolfileend='_talLocs_database_bipol.mat'):
    talfile = glob(subjpath+taldir+'/*'+bipolfileend)
    
    if len(talfile)!=1:
        raise ValueError('Invalid number of files! '+str(talfile)+'\n' +str(subjpath))
    else:
        talfile = talfile[0]
    try:
        tf = loadmat(talfile,struct_as_record=True,squeeze_me=True)['bpTalStruct']
    except KeyError:
        tf = loadmat(talfile,struct_as_record=True,squeeze_me=True)['subjTalEvents']

    dtypes_surf = [('x', np.float), ('y', np.float), ('z', np.float),
                   ('bpDistance', np.float), ('anatRegion', '|S64'),
                   ('x_snap', np.float), ('y_snap', np.float),
                   ('z_snap', np.float), ('anatRegion_snap', '|S64'),
                   ('distance_snap', np.float), ('bpDistance_snap', np.float),
                   ('x_eSnap', np.float), ('y_eSnap', np.float),
                   ('z_eSnap', np.float), ('anatRegion_eSnap', '|S64'),
                   ('distance_eSnap', np.float), ('bpDistance_eSnap', np.float),
                   ('x_dural', np.float), ('y_dural', np.float),
                   ('z_dural', np.float), ('anatRegion_dural', '|S64'),
                   ('distance_dural', np.float), ('bpDistance_dural', np.float),
                   ('path2surfL', '|S128'), ('path2surfR', '|S128')]
    dtypes = [('subject', '|S12'), ('channel', list), ('channel_str', list),
              ('tagName', '|S32'),
              ('grpName', '|S20'), ('x', np.float), ('y', np.float),
              ('z', np.float), ('Loc1', '|S64'), ('Loc2', '|S64'),
              ('Loc3', '|S64'), ('Loc4', '|S64'), ('Loc5', '|S64'),
              ('Loc6', '|S64'), ('Montage', '|S20'), ('eNames', '|S20'),
              ('eType', '|S8'), ('bpDistance', np.float), ('distance', np.float),
              ('avgSurf', dtypes_surf), ('indivSurf', dtypes_surf), ('locTag', '|S20')]

    new_tf = np.rec.recarray(len(tf),dtype=dtypes)
    new_tf_surf = np.rec.recarray(len(tf),dtype=dtypes_surf)
    # create field with channels as string names (e.g., '001' instead of 1):
    channel_str = np.zeros(len(tf['channel']),np.object)
    for c,chan in enumerate(tf['channel']):
        channel_str[c] = np.zeros(2,'|S8')
        for i,indivchan in enumerate(chan):
            if indivchan<10:
                channel_str[c][i] = '00'+str(indivchan)
            elif indivchan<100:
                channel_str[c][i] = '0'+str(indivchan)
            else:
                channel_str[c][i] = str(indivchan)          
    for field in tf.dtype.names:
        if ((field == 'avgSurf') | (field == 'indivSurf')):
            for f,field_surf in enumerate(tf[0][field].dtype.names):
                if field_surf == dtypes_surf[f][0]:
                    new_tf_surf[field_surf] =  tuple(np.array(
                        [x[field_surf] for x in tf[field]], dtypes_surf[f][1]))
                else:
                    new_tf_surf[field_surf] =  tuple(np.array(
                        [np.nan for x in tf[field]], dtypes_surf[f][1]))
            new_tf[field] = new_tf_surf
        else:
            new_tf[field] = tf[field]
    new_tf['channel_str'] = channel_str

    leadspath = subjpath+taldir+'/leads.txt'
    badleadspath = subjpath+taldir+'/bad_leads.txt'
    goodleadspath = subjpath+taldir+'/good_leads.txt'
    if leadsonly:
        leads = np.array([lead.strip() for lead in open(leadspath,'Ur').readlines()])
        leads = np.uint16(leads[leads!=''])
        # only keep electrodes with neural data (i.e., electrodes that are
        # in /data/eeg/[subj]/tal/leads.txt
        good_indices = np.arange(len(new_tf))
        good_indices = np.array(filter(lambda x: np.all(
            [chan in leads for chan in new_tf['channel'][x]]), good_indices))
        new_tf = new_tf[good_indices]
    if exclude_bad_leads:
        try:
            bad_leads = np.array([lead.strip() for lead in open(badleadspath,'r').readlines()])
            bad_leads = np.uint16(bad_leads[bad_leads!=''])
            # get rid of electrodes over epileptic regions (i.e., electrodes that are
            # in /data/eeg/[subj]/tal/bad_leads.txt
            good_indices = np.arange(len(new_tf))
            good_indices = np.array(filter(lambda x: ~np.any(
                [chan in bad_leads for chan in new_tf['channel'][x]]), good_indices))
            new_tf = new_tf[good_indices]
        except IOError:
            print(badleadspath + ' does not exist ... trying good_leads.txt')
            try:
                good_leads = np.array([lead.strip() for lead in open(goodleadspath,'r').readlines()])
                good_leads = np.uint16(good_leads[good_leads!=''])
                # only keep electrodes over good regions (i.e., electrodes that are
                # in /data/eeg/[subj]/tal/good_leads.txt
                good_indices = np.arange(len(new_tf))
                good_indices = np.array(filter(lambda x: np.all(
                    [chan in good_leads for chan in new_tf['channel'][x]]), good_indices))
                new_tf = new_tf[good_indices]  
            except IOError:
                print(goodleadspath + ' does not exist ... giving up!')
    return(new_tf)




# # # only keep electrodes over good regions (i.e., electrodes that are
# # # in /data/eeg/[subj]/tal/good_leads.txt

# # good_indices = np.arange(len(new_tf))
# # good_indices = np.array(filter(lambda x: np.all(
# #     [chan in bad_leads for chan in new_tf['channel'][x]]), good_indices))
# # new_tf = new_tf[good_indices]


        


# talfile = '/home/ctw/fusemounts/rhino/data/eeg/R1036M/tal/R1036M_talLocs_database_bipol.mat'
# tf = loadmat(talfile)
# tf = loadmat(talfile,struct_as_record=True,squeeze_me=True)['bpTalStruct']
# leadspath = '/home/ctw/fusemounts/rhino/data/eeg/R1036M/tal/leads.txt'
# badleadspath = '/home/ctw/fusemounts/rhino/data/eeg/R1036M/tal/bad_leads.txt'
# goodleadspath = '/home/ctw/fusemounts/rhino/data/eeg/R1036M/tal/good_leads.txt'
# # leads = [np.int(lead.strip()) for lead in open(leadspath,'r').readlines()]
# # bad_leads = [np.int(lead.strip()) for lead in open(badleadspath,'r').readlines()]
# # good_leads = [np.int(lead.strip()) for lead in open(goodleadspath,'r').readlines()]

# leads = np.array([lead.strip() for lead in open(leadspath,'r').readlines()])
# leads = np.uint16(leads[leads!=''])

# bad_leads = np.array([lead.strip() for lead in open(badleadspath,'r').readlines()])
# bad_leads = np.uint16(bad_leads[bad_leads!=''])

# # good_leads = np.array([lead.strip() for lead in open(goodleadspath,'r').readlines()])
# # good_leads = np.uint16(good_leads[good_leads!=''])

# dtypes = [('subject', '|S12'), ('channel', list), ('tagName', '|S32'),
#           ('grpName', '|S20'), ('x', np.float), ('y', np.float), ('z', np.float),
#           ('Loc1', '|S64'), ('Loc2', '|S64'), ('Loc3', '|S64'), ('Loc4', '|S64'),
#           ('Loc5', '|S64'), ('Loc6', '|S64'), ('Montage', '|S20'),
#           ('eNames', '|S20'), ('eType', '|S8'), ('bpDistance', np.float),
#           ('avgSurf', 'O'), ('indivSurf', 'O'), ('locTag', '|S20')]
# new_tf = np.rec.recarray(len(tf),dtype=dtypes)
# for field in tf.dtype.names:
#     new_tf[field] = tf[field]
    
# # only keep electrodes with neural data (i.e., electrodes that are
# # in /data/eeg/[subj]/tal/leads.txt
# good_indices = np.arange(len(new_tf))
# good_indices = np.array(filter(lambda x: np.all(
#     [chan in leads for chan in new_tf['channel'][x]]), good_indices))
# new_tf = new_tf[good_indices]

# # get rid of electrodes over epileptic regions (i.e., electrodes that are
# # in /data/eeg/[subj]/tal/bad_leads.txt

# good_indices = np.arange(len(new_tf))
# good_indices = np.array(filter(lambda x: ~np.any(
#     [chan in bad_leads for chan in new_tf['channel'][x]]), good_indices))
# new_tf = new_tf[good_indices]

# # # only keep electrodes over good regions (i.e., electrodes that are
# # # in /data/eeg/[subj]/tal/good_leads.txt

# # good_indices = np.arange(len(new_tf))
# # good_indices = np.array(filter(lambda x: np.all(
# #     [chan in bad_leads for chan in new_tf['channel'][x]]), good_indices))
# # new_tf = new_tf[good_indices]

