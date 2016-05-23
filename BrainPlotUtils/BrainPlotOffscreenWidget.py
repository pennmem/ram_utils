from collections import OrderedDict
import vtk
from vtk import (vtkSphereSource, vtkPolyDataMapper, vtkActor, vtkRenderer,
        vtkRenderWindow, vtkWindowToImageFilter, vtkPNGWriter, vtkVersion)

from JSONUtils import JSONNode
import inspect

import os,sys
from glob import glob
from os.path import *

from brain_plot_utils import *

import numpy as np

vtk_major_version = vtkVersion.GetVTKMajorVersion()

class BrainPlotOffscreenWidget(object):
    # def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
    def __init__(self):



        self.actors_dict = {}

        self.display_obj_dict = OrderedDict()

        self.ren = vtkRenderer()
        self.ren.SetBackground(1., 1., 1.)

        self.renWin = vtkRenderWindow()
        self.renWin.SetOffScreenRendering(1)
        self.renWin.AddRenderer(self.ren)

        self.renWin.SetSize(1000,1000)


        self.image_format = 'png'
        self.screenshot_fcn_dict = {
            'png':self.take_screenshot_png,
            'pdf':self.take_screenshot_pdf,
        }

        self.take_screenshot = self.screenshot_fcn_dict['png']

        self.camera_setting_dir = os.getcwd()
        self.anim_dir = os.getcwd()

    def set_image_format(self,format='png'):

        fmt = format.lower()
        try:
            self.take_screenshot = self.screenshot_fcn_dict[fmt]
            self.image_format = fmt
        except KeyError:
            print 'unsupported image format: '+self.image_format


    def set_size(self,x,y):
        self.renWin.SetSize(x,y)

    def display(self, **options):

        self.render_scene(**options)

        self.raise_()

        self.show()

        # start event processing
        self.app.exec_()

    def add_actor(self, actor_name, actor):
        self.ren.AddActor(actor)
        self.actors_dict[actor_name] = actor


    def take_screenshot_pdf(self, filename):
        exp = vtk.vtkGL2PSExporter()
        exp.SetFileFormatToPDF()
        exp.SetRenderWindow(self.renWin)

        core_file_name, ext = os.path.splitext(filename)

        exp.SetFilePrefix(core_file_name)
        # Turn off compression so PIL can read file.
        exp.CompressOff()
        exp.DrawBackgroundOn()
        exp.Write()


    def take_screenshot_png(self, filename):


        renderLarge = vtk.vtkRenderLargeImage()
        if vtk_major_version <= 5:
            renderLarge.SetInputData(self.ren)
        else:
            renderLarge.SetInput(self.ren)

        renderLarge.SetMagnification(1)

        # We write out the image which causes the rendering to occur. If you
        # watch your screen you might see the pieces being rendered right
        # after one another.
        writer = vtkPNGWriter()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        # # # print "GOT HERE fileName=",fileName
        writer.SetFileName(filename)

        writer.Write()


    # def photoshoot(self, camera_setting_dir='', output_dir=''):
    #
    #     if not output_dir:
    #         output_dir = os.getcwd()
    #
    #     if not isdir(output_dir):
    #         os.makedirs(output_dir)
    #
    #     if not camera_setting_dir:
    #         camera_setting_dir = str(self.camera_setting_LE.text())
    #
    #     if not camera_setting_dir:
    #         camera_setting_dir = join(os.getcwd(), 'camera_settings')
    #
    #     camera_files = glob.glob(join(camera_setting_dir, '*.camera.json'))
    #     print camera_files
    #
    #     screenshot_core_name = str(self.screenshot_core_name_LE.text())
    #     for c_file in camera_files:
    #         self.load_camera_settings(c_file)
    #
    #         screenshot_filename = join(output_dir, screenshot_core_name + '_' + basename(c_file) + '.png')
    #         self.take_screenshot(screenshot_filename)
    #         # self.take_screenshot(c_file+'.png')



    def load_camera_settings(self, filename=''):

        filename = locate_file(filename=filename,local_dir='camera_settings')

        if not filename:
            filename = 'default.camera.json'

        cam = self.ren.GetActiveCamera()

        cam_node = JSONNode.read(filename)
        cam.SetClippingRange(float(cam_node['clipping_range']['min']), float(cam_node['clipping_range']['max']))
        cam.SetFocalPoint(float(cam_node['focal_point']['x']), float(cam_node['focal_point']['y']),
                          float(cam_node['focal_point']['z']))
        cam.SetPosition(float(cam_node['position']['x']), float(cam_node['position']['y']),
                        float(cam_node['position']['z']))
        cam.SetViewUp(float(cam_node['view_up']['x']), float(cam_node['view_up']['y']), float(cam_node['view_up']['z']))

        self.render_scene()
        pass


    def add_display_object(self, obj_name, obj):
        self.display_obj_dict[obj_name] = obj

    def render_scene(self, **options):
        self.ren.RemoveAllViewProps()


        for disp_obj_name, disp_obj in self.display_obj_dict.items():
            mapper = disp_obj.get_polydata_mapper()


            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            if disp_obj.get_opacity() is not None:
                actor.GetProperty().SetOpacity(disp_obj.get_opacity())

            if disp_obj.get_color() is not None:
                actor.GetProperty().SetColor(disp_obj.get_color())


            self.add_actor(disp_obj_name, actor)

        self.renWin.Render()


if __name__=='__main__':
    # sys.path.append('/Users/m/PTSA_NEW_GIT')

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
