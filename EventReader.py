__author__ = 'm'

from os.path import *
import numpy as np
# from scipy.io import loadmat

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper


from BaseEventReader import BaseEventReader

class EventReader(BaseEventReader):
    def __init__(self, event_file, **kwds):
        BaseEventReader.__init__(self,event_file, **kwds)


    def read(self):

        # calling base class read fcn
        evs = BaseEventReader.read(self)
        evs = evs.add_fields(esrc=np.dtype(RawBinWrapper))

        import pathlib

        for ev in evs:
            try:
                eeg_file_path = join(self.data_dir_prefix, str(pathlib.Path(str(ev.eegfile)).parts[1:]))
                ev.esrc = RawBinWrapper(eeg_file_path)
                self.raw_data_root=str(eeg_file_path)
            except TypeError:
                print 'skipping event with eegfile=',ev.eegfile
                pass

        self.subject_path = str(pathlib.Path(eeg_file_path).parts[:-2])

        # attaching
        # evs.add_fields(esrc=np.dtype(RawBinWrapper))

        self.set_output(evs)

        return self.get_output()


if __name__=='__main__':

        from EventReader import EventReader
        e_path = join('/Volumes/rhino_root', 'data/events/RAM_FR1/R1060M_events.mat')
        # e_path = '/Users/m/data/events/RAM_FR1/R1056M_events.mat'
        e_reader = EventReader(event_file=e_path, eliminate_events_with_no_eeg=True, data_dir_prefix='/Volumes/rhino_root')

        events = e_reader.read()

        events = e_reader.get_output()