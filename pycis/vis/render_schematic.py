import os
import numpy as np
import glob
import yaml
import vtk
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtk import vtkFeatureEdges, vtkRenderLargeImage
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkCommonCore import (
    vtkPoints,
    vtkUnsignedCharArray,
    vtkMath,
    vtkMinimalStandardRandomSequence,
)
from vtkmodules.vtkCommonColor import (
    vtkColorSeries,
    vtkNamedColors,
)
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import (
    vtkArrowSource,
    vtkConeSource,
    vtkCubeSource,
    vtkCylinderSource,
    vtkSphereSource,
    vtkLineSource,
)
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkLine,
    vtkPolyData
)
from vtkmodules.vtkFiltersGeneral import vtkAxes
from vtkmodules.vtkCommonDataModel import vtkSphere, vtkCylinder
from vtkmodules.vtkIOImage import vtkPNGWriter, vtkPostScriptWriter
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkWindowToImageFilter,
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextActor,
)

# --------
# SETTINGS
RADIUS = 2
WIDTH_POL = 0.5
WIDTH_RET = 1.5
PIX_HEIGHT = 300
PIX_WIDTH = 300
CYLINDER_RESOLUTION = 100
TUBE_RADIUS_DEFAULT = 0.02

WIDTHS = {
    'LinearPolariser': 0.25,
    'UniaxialCrystal': 1.25,
}
COLORS = {
    'LinearPolariser': 'White',
    'UniaxialCrystal': 'AliceBlue',
}
LABELS = {
    'LinearPolariser': 'POL',
    'UniaxialCrystal': 'RET',
}
COLOR_LINE_DEFAULT = 'Black'
WIDTH_SPACING = 2.6
CAMERA_X_POS = 4
USER_MATRIX = True
FPATH_ROOT = os.path.dirname(os.path.realpath(__file__))
FPATH_CONFIG = os.path.join(FPATH_ROOT, 'pycis_config.yaml')


def render_schematic(fpath_config, fpath_out, show_axes=False, ):
    """
    Render a 3-D isometric diagram of the given interferometer configuration

    THIS IS A QUICK PROTOTYPE AND A HACK!

    Useful VTK links:
    https://kitware.github.io/vtk-examples/site/Python/Rendering/OutlineGlowPass/  (GLOWING OUTLINE, UNUSED)
    https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/OrientedArrow/  (3-D ARROW, UNUSED)

    :param str fpath_config:
    :param str fpath_out:
    :param bool show_axes:
    """

    with open(fpath_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -----
    # SETUP
    colors = vtkNamedColors()
    bkg = map(lambda x: x / 255.0, [255, 255, 255, 255])
    colors.SetColor("BkgColor", *bkg)
    iren = vtkRenderWindowInteractor()
    render_window = vtkRenderWindow()
    render_window.SetMultiSamples(500)
    render_window.SetNumberOfLayers(3)
    render_window.SetAlphaBitPlanes(1)
    iren.SetRenderWindow(render_window)
    renderer = vtkRenderer()
    renderer.SetLayer(0)
    renderer.SetUseDepthPeeling(1)
    renderer.SetOcclusionRatio(0.05)
    # renderer.SetBackgroundAlpha(0.)  # add background opacity value
    renderer.SetMaximumNumberOfPeels(100)
    renderer_lines_fg = vtkRenderer()
    renderer_lines_fg.SetLayer(2)
    renderer_lines_bg = vtkRenderer()
    renderer_lines_bg.SetLayer(1)
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(renderer_lines_fg)
    render_window.AddRenderer(renderer_lines_bg)

    if show_axes:
        show_axes = vtkAxesActor()
        show_axes.SetOrigin(0, 0, 0)
        renderer.AddActor(show_axes)

    # -------------------------
    # ADD COMPONENTS ONE BY ONE
    width_total = 0
    for ii, cc in enumerate(config['interferometer']):

        component_type = list(cc.keys())[0]
        component_orientation_deg = cc[component_type]['orientation']
        component_orientation = component_orientation_deg * np.pi / 180
        component_width = WIDTHS[component_type]

        if ii != 0:
            width_total += WIDTH_SPACING

        # --------
        # CYLINDER
        cylinder = vtkCylinderSource()
        cylinder.SetResolution(CYLINDER_RESOLUTION)
        cylinder.SetRadius(RADIUS)
        cylinder.SetHeight(component_width)
        cylinderMapper = vtkPolyDataMapper()
        cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
        cylinderActor = vtkActor()
        cylinderActor.SetMapper(cylinderMapper)
        cylinderActor.GetProperty().SetColor(colors.GetColor3d(COLORS[component_type]))
        cylinderActor.GetProperty().SetRepresentationToSurface()
        cylinderActor.GetProperty().BackfaceCullingOn()
        cylinderActor.GetProperty().LightingOff()
        cylinderActor.GetProperty().SetOpacity(0.92)

        # ----------------
        # LINES
        def make_line(x1, y1, z1, x2, y2, z2, tube_radius=TUBE_RADIUS_DEFAULT, color=COLOR_LINE_DEFAULT):
            lineSource = vtkLineSource()
            lineSource.SetPoint1(x1, y1, z1)
            lineSource.SetPoint2(x2, y2, z2)
            lineMapper = vtkPolyDataMapper()
            lineMapper.SetInputConnection(lineSource.GetOutputPort())
            lineActor = vtkActor()
            lineActor.SetMapper(lineMapper)
            lineActor.GetProperty().SetColor(colors.GetColor3d(color))
            lineActor.GetProperty().SetLineWidth(1)
            tubeFilter = vtkTubeFilter()
            tubeFilter.SetInputConnection(lineSource.GetOutputPort())
            tubeFilter.SetRadius(tube_radius)
            tubeFilter.SetNumberOfSides(20)
            tubeFilter.Update()
            tubeMapper = vtkPolyDataMapper()
            tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())
            tubeActor = vtkActor()
            tubeActor.SetMapper(tubeMapper)
            tubeActor.GetProperty().SetColor(colors.GetColor3d(color))
            tubeActor.GetProperty().LightingOff()
            return tubeActor
        view_angle = 1.14 * np.pi / 4
        nubbin = 0.05
        connector_line_actors = [
            make_line(
                RADIUS * np.cos(view_angle), RADIUS * np.sin(view_angle), width_total,
                RADIUS * np.cos(view_angle), RADIUS * np.sin(view_angle), width_total + component_width + 4. * nubbin
            ),
            make_line(
                -RADIUS * np.cos(view_angle), -RADIUS * np.sin(view_angle), width_total,
                -RADIUS * np.cos(view_angle), -RADIUS * np.sin(view_angle), width_total + component_width + 2. * nubbin
            )
        ]
        component_xy_line_actors = [
            make_line(
                0., 1.00 * RADIUS, width_total,
                0., -1.00 * RADIUS, width_total,
                tube_radius=0.01,
            ),
            make_line(
                1.00 * RADIUS, 0, width_total,
                -1.00 * RADIUS, 0, width_total,
                tube_radius=0.01,
            ),
        ]

        # -----
        # ARROW
        arrowActor = make_line(
            RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation), width_total,
            -RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation), width_total,
            color='Red'
        )

        # -------------
        # CYLINDER EDGES
        feature_edges = vtkFeatureEdges()
        feature_edges.ColoringOff()
        feature_edges.SetInputConnection(cylinder.GetOutputPort())
        feature_edges.BoundaryEdgesOn()
        feature_edges.ManifoldEdgesOff()
        feature_edges.NonManifoldEdgesOff()
        feature_edges.FeatureEdgesOff()
        edge_actor = vtkActor()
        edge_actor.GetProperty().SetColor(0., 0., 0., )
        edge_actor.GetProperty().SetLineWidth(7)
        edge_actor.GetProperty().SetRenderLinesAsTubes(1)
        edge_actor.GetProperty().SetColor(colors.GetColor3d('Black'))
        edge_actor.GetProperty().LightingOff()
        edge_mapper = vtkPolyDataMapper()
        edge_mapper.SetInputConnection(feature_edges.GetOutputPort())
        edge_actor.SetMapper(edge_mapper)

        def transform_actor(actor):
            actor.SetPosition(0.0, 0.0, 0.0)
            actor.RotateX(90.0)
            actor.SetPosition(0.0, 0.0, width_total + component_width / 2)

        transform_actor(cylinderActor)
        transform_actor(edge_actor)

        # -----
        # LABEL
        txt = vtkTextActor()
        txt.SetInput(LABELS[component_type] + '\n(' + str(component_orientation_deg) + '°)')
        txtprop = txt.GetTextProperty()
        txtprop.SetJustificationToCentered()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(36)
        txtprop.SetColor(colors.GetColor3d('Black'))
        txt.SetDisplayPosition(390 + ii * 290, 1 + ii * 110)

        renderer.AddActor(cylinderActor)
        renderer.AddActor(edge_actor)
        renderer.AddActor(txt)
        [renderer_lines_fg.AddActor(a) for a in connector_line_actors]
        [renderer_lines_bg.AddActor(a) for a in component_xy_line_actors]
        renderer_lines_bg.AddActor(arrowActor)
        width_total += component_width

    # -------------
    # DEFINE CAMERA
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOn()  # orthographic projection
    camera.SetParallelScale(4.5)  # tweak as needed
    CAMERA_Z_POS = CAMERA_X_POS * np.tan(45 * np.pi / 180)
    CAMERA_Y_POS = np.sqrt(CAMERA_X_POS ** 2 + CAMERA_Z_POS ** 2) * np.tan(30 * np.pi / 180)
    camera.SetPosition(-CAMERA_X_POS, CAMERA_Y_POS, width_total / 2 - CAMERA_Z_POS)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetFocalPoint(0, 0, width_total / 2)
    renderer_lines_fg.SetActiveCamera(camera)
    renderer_lines_bg.SetActiveCamera(camera)

    renderer.SetBackground(colors.GetColor3d("BkgColor"))
    render_window.SetSize(len(config['interferometer']) * 380, 1000)  # width, height
    render_window.SetWindowName('CylinderExample')
    render_window.LineSmoothingOn()
    render_window.PolygonSmoothingOn()
    iren.Initialize()
    render_window.Render()
    w2if = vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    writer = vtkPNGWriter()
    writer.SetFileName(fpath_out)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

    # Start the event loop.
    # iren.Start()  #  <---- UNCOMMENT LINE FOR LIVE RENDER


def demo_2crystals():
    """
    splice together images to make a figure showing 3 different 2-crystal interferometer configurations
    """
    from PIL import Image
    from PIL import ImageFilter

    FLIST_CONFIG = sorted(glob.glob(os.path.join(FPATH_ROOT, 'pycis_config_demo_*delay.yaml')))
    N_IM = len(FLIST_CONFIG)
    flist_out = []
    for fpath_config in FLIST_CONFIG:
        fpath_out = fpath_config.split('.')[0] + '.png'
        render_schematic(fpath_config, fpath_out)
        flist_out.append(fpath_out)

    images = [Image.open(x) for x in flist_out]
    widths, heights = zip(*(i.size for i in images))
    OVERLAP_FRAC = 0.8
    # total_height = int(sum(heights) * OVERLAP_FRAC)
    max_width = max(widths)
    max_height = max(heights)
    total_height = int(max_height * (1 + (OVERLAP_FRAC * (N_IM - 1))))

    new_im = Image.new('RGBA', (max_width, total_height))

    y_offset = 0
    for im in images:
        im = im.convert('RGBA')
        im_blurred = im.filter(ImageFilter.GaussianBlur(30))
        data = im.getdata()
        data_blurred = im_blurred.getdata()
        newData = []
        for item, item_blurred in zip(data, data_blurred):
            if item_blurred[0] == 255 and item_blurred[1] == 255 and item_blurred[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        im.putdata(newData)
        new_im.paste(im, (0, y_offset), im)
        y_offset += int(OVERLAP_FRAC * im.size[1])

    background = Image.new('RGBA', new_im.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, new_im)
    alpha_composite.convert('RGB').save('demo_2crystals.png')
    # new_im.convert('RGB').save('demo_2crystals.png')

    # for f in flist_out:
    #     os.remove(f)


if __name__ == '__main__':
    # test()
    # hello_cylinder()
    # hello_cylinders_v1()
    demo_2crystals()
