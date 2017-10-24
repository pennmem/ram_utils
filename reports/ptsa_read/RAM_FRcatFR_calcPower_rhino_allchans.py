 
import sys
import numpy as np
import os
import h5py
# import cPickle as pickle
import multiprocessing as mp
from exceptions import OSError
#import matplotlib.pyplot as plt

run_on_rhino = False

if run_on_rhino:
    rhino_mount = ''
    num_mp_procs = 0
else:
    rhino_mount = '/home/ctw/fusemounts/rhino'
    num_mp_procs = 23
    

# add to python path (for ptsa and other stuff)
sys.path.append(rhino_mount+'/home1/cweidema/lib/python')
from get_bipolar_subj_elecs import get_bipolar_subj_elecs

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper
from ptsa.wavelet import phase_pow_multi

exp = 'catFR2'
pow_params = {
    'catFR1':{
        'output_path': rhino_mount+'/data10/scratch/cweidema/RAM/RAM_catFR/RAM_catFR1_power/hdf5_files_sess',
        'ev_file': rhino_mount+'/home1/cweidema/Christoph/Analyses/RAM/RAM_catFR/data/RAM_catFR1_events20151106.npy'
    },
    'catFR2':{
        'output_path': rhino_mount+'/data10/scratch/cweidema/RAM/RAM_catFR/RAM_catFR2_power/hdf5_files_sess',
        'ev_file': rhino_mount+'/home1/cweidema/Christoph/Analyses/RAM/RAM_catFR/data/RAM_catFR2_events20151106.npy'
    },
    'FR1':{
        'output_path': rhino_mount+'/data10/scratch/cweidema/RAM/RAM_FR/RAM_FR1_power/hdf5_files_sess',
        'ev_file': rhino_mount+'/home1/cweidema/Christoph/Analyses/RAM/RAM_FR/data/RAM_FR1_events20151106.npy'
    },
    'FR2':{
        'output_path': rhino_mount+'/data10/scratch/cweidema/RAM/RAM_FR/RAM_FR2_power/hdf5_files_sess',
        'ev_file': rhino_mount+'/home1/cweidema/Christoph/Analyses/RAM/RAM_FR/data/RAM_FR2_events20151106.npy'
    }
}

ev = Events(np.load(pow_params[exp]['ev_file']))
ev = ev.add_fields(esrc=np.dtype(RawBinWrapper))
ev = ev[ev['type']=='WORD']
good_indices = np.ones(len(ev),np.bool)
for e,event in enumerate(ev):
    try:
        event['esrc'] = RawBinWrapper(rhino_mount+event['eegfile'])
    except IOError:
        print('No EEG files for',event['subject'],event['session'],event['eegfile'])
        good_indices[e] = False
        
ev = ev[good_indices]

start_time = -0.6
end_time = 1.6
buf = 1
baseline = (-.6,-.4)
# eeghz = 500
powhz = 50
freqs = np.logspace(np.log10(3),np.log10(180),12)

# def ztrans_pow(dat):
#     # ztrans the power
#     zmean = dat['time>='+repr(baseline[0]),'time<'+repr(baseline[1])].mean(
#         'time').mean('events')
#     zmean = np.repeat(zmean,len(dat['time'])*len(dat['events'])).reshape(
#         dat.shape)
#     zstd = dat['time>='+repr(baseline[0]),'time<'+repr(baseline[1])].mean(
#         'time').std('events')
#     zstd = np.repeat(zstd,len(dat['time'])*len(dat['events'])).reshape(
#         dat.shape)
#     dat -= zmean
#     dat /= zstd
#     return dat,zmean[:,0,0],zstd[:,0,0]


def proc_sess(subj, sess, talinfo, pdir, exclude_bad_leads=False):
    # pdir = os.path.join(output_path,subj)
    # pdir = output_path
    if exclude_bad_leads:
        pfile_path = os.path.join(pdir,'pow_'+subj+'_'+str(sess)+'_goodleads.hdf5')
    else:
        pfile_path = os.path.join(pdir,'pow_'+subj+'_'+str(sess)+'_allleads.hdf5')
    if not os.path.exists(pdir):
        try:
            os.makedirs(pdir)
        except OSError, e:
            print '*** no worries ***',e
    elif os.path.exists(pfile_path):
        # print pfile_path,'exists!'
        return
    # open(pfile,'wb').close()
    try:
        pfile = h5py.File(pfile_path,'w-',libver='latest')
    except IOError:
        print 'Cannot create',pfile_path
        return
    print "Processing subject, session: ", subj,sess
    # data = pfile.create_dataset(
    #     'data',(len(talinfo),len(freqs),
    #             np.sum((ev['subject']==subj)&(ev['session']==sess)),
    #             np.int(np.round(powhz*(end_time-start_time)))),
    #     compression="gzip",compression_opts=9,dtype=np.float16)
    # zmeans = pfile.create_dataset(
    #     'zmeans',(len(talinfo),len(freqs)), compression="gzip",
    #     compression_opts=9,dtype=np.float16)
    # zstds = pfile.create_dataset(
    #     'zstd',(len(talinfo),len(freqs)), compression="gzip",
    #     compression_opts=9,dtype=np.float16)
    data = pfile.create_dataset(
        'data',(len(talinfo),len(freqs),
                np.sum((ev['subject']==subj)&(ev['session']==sess)),
                np.int(np.round(powhz*(end_time-start_time)))),
        compression='gzip',compression_opts=9)
    # zmeans = pfile.create_dataset(
    #     'zmeans',(len(talinfo),len(freqs)), compression="gzip",
    #     compression_opts=9)
    # zstds = pfile.create_dataset(
    #     'zstd',(len(talinfo),len(freqs)), compression="gzip",
    #     compression_opts=9)
    channels = []
    bipolar = None
    for t,ti in enumerate(talinfo):
        chan = ti['channel_str']
        if len(list(chan)) == 1:
            if bipolar is None:
                bipolar = False
            elif bipolar:
                raise ValueError(
                    'All channels for a given subject/session must have '+
                    'the same reference! (1) '+
                    subj+' - '+str(sess)+' - '+str(chan)+' - '+str(talinfo[t-1]['channel_str']))
            dat = ev[(ev['subject']==subj)&(ev['session']==sess)].get_data(
                channels=chan,start_time=start_time,end_time=end_time,buffer_time=buf,
                eoffset='eegoffset',keep_buffer=True)
            channels.append(chan)
        elif len(list(chan)) == 2:
            if bipolar is None:
                bipolar = True
            elif not bipolar:
                raise ValueError(
                    'All channels for a given subject/session must have '+
                    'the same reference! (2) '+
                    subj+' - '+str(sess)+' - '+str(chan)+' - '+str(talinfo[t-1]['channel_str']))
            dat1 = ev[(ev['subject']==subj)&(ev['session']==sess)].get_data(
                channels=chan[0],start_time=start_time,end_time=end_time,buffer_time=buf,
                eoffset='eegoffset',keep_buffer=True)
            dat2 = ev[(ev['subject']==subj)&(ev['session']==sess)].get_data(
                channels=chan[1],start_time=start_time,end_time=end_time,buffer_time=buf,
                eoffset='eegoffset',keep_buffer=True)
            dat = dat1-dat2
        else:
            raise ValueError('Invalid channels: '+str(chan))    
        print "Calculating power values for channel", chan
        # Notch Butterworth filter for 60Hz line noise:
        dat = dat.filtered([58,62],filt_type='stop',order=4)
        channels.append(chan[0]+'-'+chan[1])
        dat = phase_pow_multi(freqs,dat[0],to_return='power')
        #
        print "Downsampling channel", chan
        dat = dat.resampled(powhz,num_mp_procs=4)
        #
        # remove the buffer now that we have filtered and calculated power
        print "Removing buffer from channel", chan
        dat = dat.remove_buffer(buf) #.baseline_corrected(baseline)
        #
        # log transform power values
        dat[dat<=0.0] = np.finfo(dat.dtype).eps
        dat = np.log10(dat)
        #
        # print "Z-scoring the data for channel", chan
        # dat,zmean,zstd = ztrans_pow(dat)
        # data[t] = np.float16(dat)
        # zmeans[t] = np.float16(zmean)
        # zstds[t] = np.float16(zstd)
        data[t] = dat
        # zmeans[t] = zmean
        # zstds[t] = zstd
    # finalize HDF5 file:
    samplerate=dat.samplerate
    times = dat['time']
    events = dat['events']
    # data.attrs['baseline'] = baseline
    data.attrs['buffer'] = buf
    data.attrs['subject'] = subj
    data.attrs['session'] = sess
    data.attrs['samplerate'] = samplerate
    data.attrs['exp'] = exp
    for t in talinfo[0].dtype.names:
        # if ((t=='avgSurf')|(t=='indivSurf'))
        if len(talinfo[0][t].dtype)>0:
            for branch in talinfo[0][t].dtype.names:
                pfile['tal_info/'+t+'/'+branch] = [ti[t][branch] for ti in talinfo]
        else:
            pfile['tal_info/'+t] = [ti[t] for ti in talinfo]
    data.dims[0].label = 'channels'
    pfile['channels'] = channels
    data.dims.create_scale(pfile['channels'], 'channels')
    data.dims[0].attach_scale(pfile['channels'])
    #
    data.dims[1].label = 'freqs'
    pfile['freqs'] = freqs
    data.dims.create_scale(pfile['freqs'], 'freqs')
    data.dims[1].attach_scale(pfile['freqs'])
    data.dims[3].label = 'time'
    pfile['time'] = times
    data.dims.create_scale(pfile['time'], 'time')
    data.dims[3].attach_scale(pfile['time'])
    data.dims[2].label = 'events'
    for dtn in events.dtype.names:
        # print 'dtn',dtn, dat['events'][dtn].dtype
        if events[dtn].dtype == np.object:
            #print(dtn+' is object!')
            continue
        pfile['events/'+dtn] = events[dtn]
        data.dims.create_scale(pfile['events/'+dtn], dtn)
        data.dims[2].attach_scale(pfile['events/'+dtn])
    pfile.close()
    # make file read only to avoid accidental loss:
    os.chmod(pfile_path,0o444)

    


if num_mp_procs > 0:
    po = mp.Pool(num_mp_procs)
    res = []

for subj in np.unique(ev['subject']):
    for sess in np.unique(ev[ev['subject']==subj]['session']): #sessions:
        indx = ((ev['subject']==subj) & (ev['session']==sess))
        dataroot = np.unique([es.dataroot for es in ev[indx]['esrc']])
        if len(dataroot)!=1:
            raise ValueError('Invalid number of dataroots: '+str(dataroot))
        else:
            dataroot = dataroot[0]
        subjpath = os.path.dirname(os.path.dirname(dataroot))
        talinfo = get_bipolar_subj_elecs(
            subjpath,leadsonly=True,exclude_bad_leads=False)
        if num_mp_procs > 0:
            res.append(po.apply_async(
                proc_sess, [subj, sess, talinfo,
                            pow_params[exp]['output_path']]))
        else:
            proc_sess(subj, sess, talinfo, pow_params[exp]['output_path'])
    
if num_mp_procs > 0:
    po.close()
    # track results
    for i,r in enumerate(res):
        sys.stdout.write('%d '%i)
        sys.stdout.flush()
        r.get()
