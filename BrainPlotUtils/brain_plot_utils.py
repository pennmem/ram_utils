from os.path  import *
import inspect
import sys
import numpy as np
import vtk
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm



# import Configuration

MODULENAME = '---- QVTKRenderWindowInteractor_mac.py: '


def setVTKPaths():
    import sys
    from os import environ
    import string
    import sys
    platform = sys.platform
    if platform == 'win32':
        sys.path.insert(0, environ["PYTHON_DEPS_PATH"])

setVTKPaths()


vtk_major_version = vtk.vtkVersion().GetVTKMajorVersion()




def locate_file(filename,local_dir='camera_settings'):

    if isfile(filename):
        return filename

    module_dir = dirname(inspect.getsourcefile(locate_file))

    camera_file_name = join(module_dir,local_dir,filename)

    if isfile(camera_file_name):
        return camera_file_name

    return None




def RotMat(axis, theta):
    return expm3(cross(eye(3), axis / norm(axis) * theta))


# v, axis, theta = [3,5,0], [4,4,1], 1.2
# R0 = RotMat(axis, theta)
# print(dot(R0,v))

def extract_electrode_positions(tal_path, electrode_types=['D', 'G', 'S']):
    from ptsa.data.readers import TalReader
    tal_reader = TalReader(filename=tal_path)
    tal_structs = tal_reader.read()

    lh_selector = np.array(map(lambda loc: loc.upper().startswith('L'), tal_structs.tagName))
    rh_selector = np.array(map(lambda loc: loc.upper().startswith('R'), tal_structs.tagName))

    electrode_types_lower = map(lambda t: t.lower(), electrode_types)

    electrode_type_selector = np.array(map(lambda eType: eType.lower() in electrode_types_lower, tal_structs.eType))

    lh_data = tal_structs[['avgSurf', 'eType']]
    rh_data = tal_structs[['avgSurf', 'eType']]

    lh_data = lh_data[lh_selector & electrode_type_selector]
    rh_data = rh_data[rh_selector & electrode_type_selector]

    return lh_data, rh_data


def divergent_color_lut(table_size=20, table_range=[0, 1]):
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    # Green to tan.
    ctf.AddRGBPoint(0.0, 0.085, 0.532, 0.201)
    ctf.AddRGBPoint(0.5, 0.865, 0.865, 0.865)
    ctf.AddRGBPoint(1.0, 0.677, 0.492, 0.093)

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.Build()

    for i in range(0, table_size):
        rgb = list(ctf.GetColor(float(i) / table_size)) + [1]
        lut.SetTableValue(i, rgb)

    lut.SetTableRange(table_range[0], table_range[1])

    return lut


def cut_brain_hemi(hemi_elec_data, hemi_poly_data):
    depth_elec_data = hemi_elec_data[(hemi_elec_data.eType == 'D') | (hemi_elec_data.eType == 'd')]
    print depth_elec_data
    print

    x_coords = np.array([avg_surf.x_snap for avg_surf in depth_elec_data.avgSurf], dtype=np.float)
    y_coords = np.array([avg_surf.y_snap for avg_surf in depth_elec_data.avgSurf], dtype=np.float)
    z_coords = np.array([avg_surf.z_snap for avg_surf in depth_elec_data.avgSurf], dtype=np.float)

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    clipPlane = vtk.vtkPlane()
    clipPlane.SetNormal(0.0, 0.0, 1.0)
    # clipPlane.SetOrigin(0, 0, np.max(z_coords))
    # clipPlane.SetOrigin(np.max(x_coords), np.max(y_coords), np.max(z_coords))

    clipPlane.SetOrigin(0, 0, -500)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(hemi_poly_data)
    clipper.SetClipFunction(clipPlane)

    return clipper


def get_hemi_polydata(hemi='lh'):
    vreader = vtk.vtkPolyDataReader()
    # vreader.SetFileName('lh.vtk')
    if hemi.startswith('l'):
        lh_dataset = locate_file(filename='lh.pial.vtk',local_dir='datasets')
        vreader.SetFileName(lh_dataset)
        # vreader.SetFileName('lh.pial.vtk')
    elif hemi.startswith('r'):
        rh_dataset = locate_file(filename='rh.pial.vtk',local_dir='datasets')
        vreader.SetFileName(rh_dataset)

        # vreader.SetFileName('rh.pial.vtk')
    else:
        raise RuntimeError('hemi argument must begin with letter "l" or "r" ')

    vreader.Update()

    pd = vreader.GetOutput()

    return pd


def get_electrode_vis_data(hemi_data, lut):
    electrode_points = vtk.vtkPoints()
    electrode_colors = vtk.vtkUnsignedCharArray()
    electrode_colors.SetNumberOfComponents(3)

    num_electrodes = len(hemi_data)

    for i, avg_surf in enumerate(hemi_data.avgSurf):
        # print avg_surf
        # electrode_points.InsertNextPoint(avg_surf.x_snap, avg_surf.y_snap, avg_surf.z_snap)
        electrode_points.InsertNextPoint(avg_surf.x_snap, avg_surf.y_snap, avg_surf.z_snap)

        # electrode_colors.InsertNextTupleValue((255, 0, 0))
        color_tuple = [0, 0, 0]
        lut.GetColor(i, color_tuple)
        # lut.GetColor(num_electrodes,color_tuple)

        color_tuple = map(lambda x: int(round(x * 255)), color_tuple)

        electrode_colors.InsertNextTupleValue(color_tuple)

    return electrode_points, electrode_colors


def get_electrode_glyphs(e_pts, e_colors):
    e_glyph_shape = vtk.vtkSphereSource()

    # cone.SetResolution(5)
    # cone.SetHeight(2)
    e_glyph_shape.SetRadius(3.0)

    glyphs = vtk.vtkGlyph3D()
    glyphs.SetSourceConnection(e_glyph_shape.GetOutputPort())
    glyphs.SetColorModeToColorByScalar()

    centroidsPD = vtk.vtkPolyData()
    centroidsPD.SetPoints(e_pts)
    centroidsPD.GetPointData().SetScalars(e_colors)

    if vtk_major_version == 5:

        glyphs.SetInput(centroidsPD)


    else:
        glyphs.SetInputData(centroidsPD)

    glyphs.ScalingOff()  # IMPORTANT
    glyphs.Update()

    glyphs.ScalingOff()  # IMPORTANT
    glyphs.Update()

    return glyphs


def get_brain_verts_as_numpy_array():
    l = Hemisphere(hemi='l')
    r = Hemisphere(hemi='r')
    print l.pd
    l_num_verts = l.pd.GetNumberOfPoints()
    r_num_verts = r.pd.GetNumberOfPoints()

    verts_np = np.zeros(shape=(l_num_verts + r_num_verts, 3), dtype=np.float)

    for idx in xrange(l_num_verts):
        pt = l.pd.GetPoint(idx)
        verts_np[idx, 0] = pt[0]
        verts_np[idx, 1] = pt[1]
        verts_np[idx, 2] = pt[2]

    for idx in xrange(r_num_verts):
        pt = r.pd.GetPoint(idx)
        verts_np[l_num_verts + idx, 0] = pt[0]
        verts_np[l_num_verts + idx, 1] = pt[1]
        verts_np[l_num_verts + idx, 2] = pt[2]

    return verts_np


def project_electrode_onto_plane(elec_pos, plane_pts):
    ab = plane_pts[1] - plane_pts[0]
    ac = plane_pts[2] - plane_pts[0]

    n = cross(ab, ac)
    n = n / norm(n)

    # n,a,r,p,d, pr are all vectors in 3D space
    a = plane_pts[0]
    p = elec_pos
    r = p - a
    d = dot(n,r)*n

    pr = a+(r-d)

    # print 'distance from the plane=',norm(d)

    return pr, norm(d)


def pull_electrodes_to_surface(elec_pos_array, max_distance=1.0):
    verts_np = get_brain_verts_as_numpy_array()
    print 'Found ', len(elec_pos_array), ' electrodes'

    c_close = 0
    c_far = 0

    surface_pulled_electrodes = []
    original_pos_electrodes = []

    for loc in elec_pos_array:
        dist_vec_array = verts_np - np.array([loc[0], loc[1], loc[2]], dtype=np.float)
        dist_array = (np.linalg.norm(dist_vec_array, axis=1))
        min_dist = np.min(dist_array)

        if min_dist < max_distance:
            print 'loc=', loc
            print min_dist
            min_idx = np.argmin(dist_array)
            print 'min_idx=', min_idx, ' dist = ', dist_array[min_idx]
            print ''

            surface_pulled_electrodes.append(verts_np[min_idx, :])
            # surface_pulled_electrodes.append(loc)
            c_close += 1



        else:
            original_pos_electrodes.append(loc)
            c_far += 1

            # break
            # self.electrode_points.InsertNextPoint(loc[0], loc[1], loc[2])
    print c_close
    print c_far
    return surface_pulled_electrodes, original_pos_electrodes


def pull_electrodes_to_plane(elec_pos_array,plane_points,max_distance=1.0):

    pass

def pull_electrodes_to_z_slice(elec_pos_array, z=0.0, max_distance=1.0):
    print 'Found ', len(elec_pos_array), ' electrodes'

    c_close = 0
    c_far = 0

    pulled_electrodes_mask = abs(elec_pos_array[:, 2] - z) < max_distance

    pulled_electrodes = elec_pos_array[pulled_electrodes_mask]

    pulled_electrodes[:, 2] = z  # resetting z position of the pulled electrodes

    original_pos_electrodes = elec_pos_array[~pulled_electrodes_mask]

    print 'pulled electrodes #=', len(pulled_electrodes)
    print ' electrodes with original positions#=',len(original_pos_electrodes)

    return pulled_electrodes, original_pos_electrodes


# def take_screenshot(ren,filename):
#
#     renderLarge = vtk.vtkRenderLargeImage()
#     if vtk_major_version <= 5:
#         renderLarge.SetInputData(ren)
#     else:
#         renderLarge.SetInput(ren)
#
#
#     renderLarge.SetMagnification(1)
#
#     # We write out the image which causes the rendering to occur. If you
#     # watch your screen you might see the pieces being rendered right
#     # after one another.
#     writer = vtk.vtkPNGWriter()
#     writer.SetInputConnection(renderLarge.GetOutputPort())
#     # # # print "GOT HERE fileName=",fileName
#     writer.SetFileName(filename)
#
#     writer.Write()




def BrainPlotExample(lh_elec_data=None, rh_elec_data=None, electrode_types=['D', 'G', 'S'], filename=''):
    """A simple example that uses the QVTKRenderWindowInteractor class."""

    vtk_major_version = vtk.vtkVersion().GetVTKMajorVersion()

    import numpy as np
    # every QT app needs an app
    app = QtGui.QApplication(['QVTKRenderWindowInteractor'])
    widget = QVTKRenderWindowInteractor()

    widget.setMouseInteractionSchemeTo3D()

    widget.set_screenshot_filename(filename=filename)

    widget.Initialize()
    widget.Start()
    # if you dont want the 'q' key to exit comment this.
    widget.AddObserver("ExitEvent", lambda o, e, a=app: a.quit())

    ren = vtk.vtkRenderer()
    widget.GetRenderWindow().AddRenderer(ren)

    l_pd = get_hemi_polydata(hemi='l')
    r_pd = get_hemi_polydata(hemi='r')

    # brain cutter
    # if len(electrode_types)==1 and electrode_types[0].lower()=='d':
    #     # will cut brain
    #     clipper = cut_brain_hemi(lh_elec_data,pd)

    # clipPlane = vtk.vtkPlane()
    # clipPlane.SetNormal(1.0, -1.0, -1.0)
    # clipPlane.SetOrigin(0, 0, 0)
    #
    # clipper = vtk.vtkClipPolyData()
    # clipper.SetInputData(pd)
    # clipper.SetClipFunction(clipPlane)
    #

    l_brain_mapper = vtk.vtkPolyDataMapper()
    r_brain_mapper = vtk.vtkPolyDataMapper()

    if vtk_major_version <= 5:

        l_brain_mapper.SetInput(l_pd)
        r_brain_mapper.SetInput(r_pd)

    else:
        l_brain_mapper.SetInputData(l_pd)
        r_brain_mapper.SetInputData(r_pd)

    # brain cutter
    # brain_mapper = vtk.vtkPolyDataMapper()
    #
    # if vtk_major_version <= 5:
    #
    #     # brain_mapper.SetInput(clipper)
    #     brain_mapper.SetInputConnection(clipper.GetOutputPort())
    #
    # else:
    #     brain_mapper.SetInputConnection(clipper.GetOutputPort())

    brain_opacity = 0.2

    l_brain_actor = vtk.vtkActor()
    l_brain_actor.SetMapper(l_brain_mapper)
    l_brain_actor.GetProperty().SetOpacity(brain_opacity)
    ren.AddActor(l_brain_actor)

    r_brain_actor = vtk.vtkActor()
    r_brain_actor.SetMapper(r_brain_mapper)
    r_brain_actor.GetProperty().SetOpacity(brain_opacity)

    ren.AddActor(r_brain_actor)

    #

    cam = widget.GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()

    # top view
    cam.SetClippingRange((312.385, 827.346))
    cam.SetFocalPoint(23.9803, -13.4557, 27.6483)
    cam.SetPosition(-2.03758, 20.7186, 539.993)
    cam.SetViewUp(0.0346923, 0.997298, -0.0647596)

    lut = divergent_color_lut(table_range=[0, 120])

    lh_e_pts, lh_e_colors = get_electrode_vis_data(hemi_data=lh_elec_data, lut=lut)
    rh_e_pts, rh_e_colors = get_electrode_vis_data(hemi_data=rh_elec_data, lut=lut)

    l_glyphs = get_electrode_glyphs(e_pts=lh_e_pts, e_colors=lh_e_colors)
    r_glyphs = get_electrode_glyphs(e_pts=rh_e_pts, e_colors=rh_e_colors)

    l_glyphsMapper = vtk.vtkPolyDataMapper()
    l_glyphsMapper.SetInputConnection(l_glyphs.GetOutputPort())

    r_glyphsMapper = vtk.vtkPolyDataMapper()
    r_glyphsMapper.SetInputConnection(r_glyphs.GetOutputPort())

    l_electrodes_actor = vtk.vtkActor()
    l_electrodes_actor.SetMapper(l_glyphsMapper)
    ren.AddActor(l_electrodes_actor)

    r_electrodes_actor = vtk.vtkActor()
    r_electrodes_actor.SetMapper(r_glyphsMapper)
    ren.AddActor(r_electrodes_actor)

    ####################### AXES
    axes = vtk.vtkAxesActor()
    transform = vtk.vtkTransform()
    transform.Translate(-0.0, 0.0, 0.0)
    transform.Scale(100.0, 100.0, 100.0)

    axes.SetUserTransform(transform)

    axes.AxisLabelsOn()

    ren.AddActor(axes)

    widget.resize(1000, 1000)

    # show the widget
    widget.raise_()

    widget.show()

    # start event processing
    app.exec_()


class AxialSlice(object):
    def __init__(self, fname=''):
        self.fname = locate_file(fname,local_dir='datasets')
        if not self.fname:
            raise IOError('could not locate dataset for axial slice. File ' + fname + ' does not exist' )

        self.axial_mapper = None
        self.axial_actor = None
        self.pd = None

    def get_plane_points(self):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.fname)
        reader.Update()

        plane_points = reader.GetOutput()

        points_array = plane_points.GetPoints().GetData()
        points_array_size = points_array.GetNumberOfTuples()

        if points_array_size>=3:
            select_plane_points = np.array([points_array.GetTuple3(0),points_array.GetTuple3(points_array_size/2),points_array.GetTuple3(points_array_size-1)],dtype=np.float)
            # print points_array.GetTuple3(0)
            # print points_array.GetTuple3(points_array_size/2)
            # print points_array.GetTuple3(points_array_size-1)

            return select_plane_points

        return None

    def get_polydata_mapper(self):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.fname)



        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in xrange(256):
            lut.SetTableValue(i, i, i, i, 1)
        lut.Build()

        self.axial_mapper = vtk.vtkPolyDataMapper()
        self.axial_mapper.SetInputConnection(reader.GetOutputPort())
        self.axial_mapper.SetScalarRange(0, 255)
        self.axial_mapper.SetLookupTable(lut)

        return self.axial_mapper

    def get_bounds(self):
        pass

    def get_min_max(self, axis='x'):
        pass

    def get_opacity(self):
        pass

    def get_color(self):
        pass

    def set_opacity(self, o):
        pass

    def set_color(self, c):
        pass

    def get_actor(self):
        self.axial_actor = vtk.vtkActor()
        self.axial_actor.SetMapper(self.get_polydata_mapper())

        return self.axial_actor


class Hemisphere(object):
    def __init__(self, hemi='l'):

        self.pd = get_hemi_polydata(hemi=hemi)

        self.bounds = self.pd.GetBounds()

        # self.hemi_mapper = vtk.vtkPolyDataMapper()
        #
        # if vtk_major_version <= 5:
        #
        #     self.hemi_mapper.SetInput(self.pd)
        #
        #
        # else:
        #     self.hemi_mapper.SetInputData(self.pd)

        # self.hemi_actor = vtk.vtkActor()
        # self.hemi_actor.SetMapper(self.hemi_mapper)
        # l_brain_actor.GetProperty().SetOpacity(brain_opacity)

        self.opacity = None

        self.color = None

    def get_polydata_mapper(self):

        self.hemi_mapper = vtk.vtkPolyDataMapper()
        self.hemi_mapper.SetInputData(self.pd)

        return self.hemi_mapper

    def get_bounds(self):
        return self.bounds

    def get_min_max(self, axis='x'):

        if axis.lower() == 'x':
            return self.bounds[0], self.bounds[1]
        elif axis.lower() == 'y':
            return self.bounds[2], self.bounds[3]
        elif axis.lower() == 'z':
            return self.bounds[4], self.bounds[5]

        raise RuntimeError('axis string has to be x,y,or z')

    def get_opacity(self):
        return self.opacity

    def get_color(self):
        return self.color

    def set_opacity(self, o):
        self.opacity = o
        # self.hemi_actor.GetProperty().SetOpacity(o)

    def set_color(self, c):
        self.color = c
        # self.hemi_actor.GetProperty().SetColor(c)

    def get_actor(self):
        return self.hemi_actor


def extract_electrode_positions_for_single_subject(tal_structs, electrode_types=['D', 'G', 'S']):
    lh_selector = np.array(map(lambda loc: loc.upper().startswith('L'), tal_structs.tagName))
    rh_selector = np.array(map(lambda loc: loc.upper().startswith('R'), tal_structs.tagName))

    electrode_types_lower = map(lambda t: t.lower(), electrode_types)

    electrode_type_selector = np.array(map(lambda eType: eType.lower() in electrode_types_lower, tal_structs.eType))

    lh_data = tal_structs[['avgSurf', 'eType']]
    rh_data = tal_structs[['avgSurf', 'eType']]

    lh_data = lh_data[lh_selector & electrode_type_selector]
    rh_data = rh_data[rh_selector & electrode_type_selector]

    lh_data_np = np.zeros(shape=(len(lh_data), 3), dtype=np.float)
    rh_data_np = np.zeros(shape=(len(rh_data), 3), dtype=np.float)

    lh_data_np[:, 0] = lh_data.avgSurf.x_snap
    lh_data_np[:, 1] = lh_data.avgSurf.y_snap
    lh_data_np[:, 2] = lh_data.avgSurf.z_snap

    rh_data_np[:, 0] = rh_data.avgSurf.x_snap
    rh_data_np[:, 1] = rh_data.avgSurf.y_snap
    rh_data_np[:, 2] = rh_data.avgSurf.z_snap

    return lh_data_np, rh_data_np


class Electrodes(object):
    def __init__(self, shape='sphere'):

        self.electrode_points = vtk.vtkPoints()
        self.electrode_colors = vtk.vtkUnsignedCharArray()
        self.electrode_colors.SetNumberOfComponents(3)

        if shape == 'sphere':
            self.e_glyph_shape = vtk.vtkSphereSource()
            self.e_glyph_shape.SetRadius(3.0)

        elif shape == 'cone':
            self.e_glyph_shape = vtk.vtkConeSource()
            self.e_glyph_shape.SetResolution(5)
            self.e_glyph_shape.SetHeight(5)
            self.e_glyph_shape.SetRadius(5)

        self.opacity = None

        self.color = None

    def get_opacity(self):
        return self.opacity

    def get_color(self):
        return self.color

    def set_opacity(self, o):
        self.opacity = o

    def set_color(self, c):
        self.color = c

    def get_polydata_mapper(self):
        self.glyphs = vtk.vtkGlyph3D()
        self.glyphs.SetSourceConnection(self.e_glyph_shape.GetOutputPort())
        self.glyphs.SetColorModeToColorByScalar()

        self.centroidsPD = vtk.vtkPolyData()
        self.centroidsPD.SetPoints(self.electrode_points)
        self.centroidsPD.GetPointData().SetScalars(self.electrode_colors)

        if vtk_major_version == 5:

            self.glyphs.SetInput(self.centroidsPD)


        else:
            self.glyphs.SetInputData(self.centroidsPD)

        self.glyphsMapper = vtk.vtkPolyDataMapper()
        self.glyphsMapper.SetInputConnection(self.glyphs.GetOutputPort())

        self.glyphs.ScalingOff()  # IMPORTANT
        self.glyphs.Update()

        return self.glyphsMapper

    def set_electrodes_locations(self, loc_array):

        for loc in loc_array:
            self.electrode_points.InsertNextPoint(loc[0], loc[1], loc[2])

    def set_electrodes_color(self, c=[255, 0, 0]):
        self.electrode_colors = vtk.vtkUnsignedCharArray()
        self.electrode_colors.SetNumberOfComponents(3)

        for i in range(self.electrode_points.GetNumberOfPoints()):
            self.electrode_colors.InsertNextTupleValue(c)

    def color_electrodes_by_scalars(self, scalar_array, lut=None):

        s_array = np.array(scalar_array, dtype=np.float)

        self.electrode_colors = vtk.vtkUnsignedCharArray()
        self.electrode_colors.SetNumberOfComponents(3)

        if lut is None:
            min, max = np.min(s_array), np.max(s_array)
            lut = divergent_color_lut(table_size=len(s_array), table_range=[min, max])

        for scalar in scalar_array:
            color_tuple = [0, 0, 0]
            lut.GetColor(scalar, color_tuple)

            color_tuple = map(lambda x: int(round(x * 255)), color_tuple)

            self.electrode_colors.InsertNextTupleValue(color_tuple)

    def get_actor(self):

        # cone.SetResolution(5)
        # cone.SetHeight(2)


        self.glyphs = vtk.vtkGlyph3D()
        self.glyphs.SetSourceConnection(self.e_glyph_shape.GetOutputPort())
        self.glyphs.SetColorModeToColorByScalar()

        self.centroidsPD = vtk.vtkPolyData()
        self.centroidsPD.SetPoints(self.electrode_points)
        self.centroidsPD.GetPointData().SetScalars(self.electrode_colors)

        if vtk_major_version == 5:

            self.glyphs.SetInput(self.centroidsPD)


        else:
            self.glyphs.SetInputData(self.centroidsPD)

        self.glyphsMapper = vtk.vtkPolyDataMapper()
        self.glyphsMapper.SetInputConnection(self.glyphs.GetOutputPort())

        self.glyphs.ScalingOff()  # IMPORTANT
        self.glyphs.Update()

        self.electrodes_actor = vtk.vtkActor()
        self.electrodes_actor.SetMapper(self.glyphsMapper)

        return self.electrodes_actor


class BrainPlot(object):
    def __init__(self):
        self.vtk_major_version = vtk.vtkVersion().GetVTKMajorVersion()

        import numpy as np
        # every QT app needs an app
        self.app = QtGui.QApplication(['QVTKRenderWindowInteractor'])
        self.widget = QVTKRenderWindowInteractor()

        self.widget.setMouseInteractionSchemeTo3D()

        # widget.set_screenshot_filename(filename=filename)

        self.widget.Initialize()
        self.widget.Start()
        # if you dont want the 'q' key to exit comment this.
        self.widget.AddObserver("ExitEvent", lambda o, e, a=self.app: a.quit())

        self.ren = vtk.vtkRenderer()
        self.widget.GetRenderWindow().AddRenderer(self.ren)

    # def add_hemispheres(self, hemi_list=['l','r']):
    #     for hemi in hemi_list:
    #         h = Hemisphere(hemi=hemi)


    def add_actor(self, actor):
        self.ren.AddActor(actor)

    def plot(self):
        self.widget.resize(1000, 1000)

        # show the widget
        self.widget.raise_()

        self.widget.show()

        # start event processing
        self.app.exec_()


# if __name__ == "__main__":
#     # QVTKRenderWidgetConeExample()
#     sys.path.append('/Users/m/PTSA_NEW_GIT')
#
#     tal_path = '/Users/m/data/eeg/R1111M/tal/R1111M_talLocs_database_bipol.mat'
#     # tal_path = '/Users/m/data/eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat'
#
#     electrode_types =['D','S','G']
#     lh_elec_data, rh_elec_data = extract_electrode_positions(tal_path=tal_path, electrode_types=electrode_types)
#
#     # BrainPlotExample(lh_elec_data=lh_elec_data, rh_elec_data=rh_elec_data,electrode_types=electrode_types,filename='screenshot2.png')
#     BrainPlotExample(lh_elec_data=lh_elec_data, rh_elec_data=rh_elec_data,electrode_types=electrode_types,filename='')


if __name__ == "__main__":
    # QVTKRenderWidgetConeExample()
    sys.path.append('/Users/m/PTSA_NEW_GIT')

    # tal_path = '/Users/m/data/eeg/R1111M/tal/R1111M_talLocs_database_bipol.mat'
    # # tal_path = '/Users/m/data/eeg/R1060M/tal/R1060M_talLocs_database_bipol.mat'
    #
    # electrode_types =['D','S','G']
    # lh_elec_data, rh_elec_data = extract_electrode_positions(tal_path=tal_path, electrode_types=electrode_types)
    #
    # # BrainPlotExample(lh_elec_data=lh_elec_data, rh_elec_data=rh_elec_data,electrode_types=electrode_types,filename='screenshot2.png')
    # BrainPlotExample(lh_elec_data=lh_elec_data, rh_elec_data=rh_elec_data,electrode_types=electrode_types,filename='')


    tal_path = '/Users/m/data/eeg/R1111M/tal/R1111M_talLocs_database_bipol.mat'
    depth_lh_elec_data, depth_rh_elec_data = extract_electrode_positions_for_single_subject(tal_path=tal_path,
                                                                                            electrode_types=['D'])
    strip_lh_elec_data, strip_rh_elec_data = extract_electrode_positions_for_single_subject(tal_path=tal_path,
                                                                                            electrode_types=['S'])
    grid_lh_elec_data, grid_rh_elec_data = extract_electrode_positions_for_single_subject(tal_path=tal_path,
                                                                                          electrode_types=['G'])

    bp = BrainPlot()

    lh = Hemisphere(hemi='l')
    lh.set_opacity(0.1)

    rh = Hemisphere(hemi='r')
    rh.set_color(c=[1, 0, 0])
    rh.set_opacity(0.1)

    bp.add_actor(lh.get_actor())
    bp.add_actor(rh.get_actor())

    depth_elec = Electrodes(shape='cone')
    # elec.set_electrodes_locations(loc_array=[[0,0,0]])
    depth_elec.set_electrodes_locations(loc_array=depth_lh_elec_data)
    depth_elec.set_electrodes_color(c=[0, 255, 0])
    bp.add_actor(depth_elec.get_actor())

    strip_elec = Electrodes(shape='sphere')
    # elec.set_electrodes_locations(loc_array=[[0,0,0]])
    strip_elec.set_electrodes_locations(loc_array=strip_lh_elec_data)
    strip_elec.set_electrodes_color(c=[255, 255, 0])
    bp.add_actor(strip_elec.get_actor())

    grid_elec = Electrodes(shape='sphere')
    # elec.set_electrodes_locations(loc_array=[[0,0,0]])
    grid_elec.set_electrodes_locations(loc_array=grid_lh_elec_data)

    grid_elec.color_electrodes_by_scalars(scalar_array=np.arange(len(grid_lh_elec_data)) / 10.0)
    # grid_elec.set_electrodes_color(c=[255,0,0])
    bp.add_actor(grid_elec.get_actor())

    bp.plot()
