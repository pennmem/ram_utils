__author__ = 'm'

from ptsa.data.readers import TalReader

from RamPipeline import *

from ReportUtils import ReportRamTask

from BrainPlotUtils import *

import numpy as np


class BrainPlotsPreparation_new(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(BrainPlotsPreparation_new, self).__init__(mark_as_completed)

    def run(self):

        w = BrainPlotOffscreenWidget()
        w.set_size(1000,1000)
        # w.set_image_format('pdf')

        mount_point = '/Users/m'
        mount_point = '/'

        from ptsa.data.readers import TalReader
        subject = 'R1060M'
        tal_path = os.path.join(mount_point,'data/eeg',subject,'tal',subject+'_talLocs_database_monopol.mat')
        tal_reader = TalReader(filename=tal_path,struct_name='talStruct')
        tal_struct = tal_reader.read()


        # axial_slice = AxialSlice(fname='axial-tal-10.0.vtk')
        # w.add_display_object('axial_slice',axial_slice)
        # w.load_camera_settings(filename='top.camera.json')
        # w.take_screenshot(filename='axial.png')

        lh = Hemisphere(hemi='l')
        rh = Hemisphere(hemi='r')

        w.add_display_object('lh',lh)
        w.add_display_object('rh',rh)


        dg_elec = Electrodes(shape='sphere')

        dg_tal_struct = tal_struct[ (tal_struct.eType=='S') | (tal_struct.eType=='G')]

        dg_elec_locs = np.vstack((dg_tal_struct.avgSurf.x,dg_tal_struct.avgSurf.y,dg_tal_struct.avgSurf.z)).T




        # elec.set_electrodes_locations(loc_array=[[0,0,0]])
        dg_elec.set_electrodes_locations(loc_array=dg_elec_locs)
        dg_elec.set_electrodes_color(c=[255, 255, 0])

        w.add_display_object('dg_elec',dg_elec)


        for camera_name in ('left','right','top','bottom'):

            w.load_camera_settings(filename=camera_name+'.camera.json')
            w.take_screenshot(filename='brain-'+camera_name+'.png')

