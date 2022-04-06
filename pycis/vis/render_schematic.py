import os
import numpy as np
import glob
import yaml
import string
from PIL import Image, ImageFilter, ImageChops, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager
from vtk import vtkFeatureEdges, vtkRenderLargeImage, vtkLabeledDataMapper, vtkActor2D, vtkAngleWidget, vtkLight
from vtkmodules.vtkCommonCore import (
    vtkPoints,
)
from vtkmodules.vtkCommonColor import (
    vtkNamedColors,
)
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import (
    vtkCylinderSource,
    vtkLineSource,
    vtkArcSource,
    vtkCubeSource,
)
from vtkmodules.vtkCommonDataModel import (
    vtkPolyData,
    vtkPolyLine,
    vtkTriangle,
    vtkPolygon,
    vtkRect,
    vtkLine,
    vtkCellArray,
)
from vtkmodules.vtkIOImage import vtkPNGWriter, vtkPostScriptWriter
from vtkmodules.vtkRenderingCore import (
    vtkWindowToImageFilter,
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextActor,
)
from pycis import Instrument, get_spectrum_delta, fft2_im

# ------------
# USER OPTIONS -- things you might want to change
SYMBOL_ORIENTATION_ANGLE = '\\rho'
SYMBOL_CUT_ANGLE = '\\theta'
SYMBOL_THICKNESS = 'L'
FONTSIZE_LABEL = 58

# ----------------
# DIAGRAM SETTINGS -- things you probably don't want to change
RADIUS = 2
WIDTH_POL = 0.5
WIDTH_RET = 1.5
PIX_HEIGHT = 300
PIX_WIDTH = 300
CYLINDER_RESOLUTION = 100
CYLINDER_OPACITY = 0.82
LINEWIDTH_CYLINDER = 5.
line_width_axes = 3
TUBE_RADIUS_DEFAULT = 0.02
IMG_BORDER = 40  # whitespace around image, in pixels
WIDTHS = {
    'LinearPolariser': 0.1,
    'UniaxialCrystal': 1.,
    'QuarterWaveplate': 0.1,
}
COLORS = {
    'LinearPolariser': 'White',
    'UniaxialCrystal': 'AliceBlue',
    'QuarterWaveplate': 'AliceBlue',
}
LABELS = {
    'LinearPolariser': 'POL',
    'UniaxialCrystal': 'RET',
    'QuarterWaveplate': 'QWP',
}
COLOR_LINE_DEFAULT = 'Black'
COLOR_AXIS = 'DimGray'
WIDTH_SPACING = 2.3
CAMERA_X_POS = 4.7
SMOL = 0.01 * RADIUS  # small nudges to avoid rendering artefacts
FPATH_ROOT = os.path.dirname(os.path.realpath(__file__))
FPATH_CONFIG = os.path.join(FPATH_ROOT, 'config')
FPATH_TEMP = os.path.join(FPATH_ROOT, 'temp')

# ---------------------------------
# POLARISED SENSOR DISPLAY SETTINGS
npix = 8  # no. pixels in each dimension x & y
line_width_grid = 1
line_width_grid_bold = 3
line_width_pol = 2


def render_schematic(fpath_config, fpath_out, show_axes=True, show_cut_angle=True, show_label_details=True, title=None,
                     border=IMG_BORDER):
    """
    Render a schematic diagram of the interferometer with 3-D isometric projection using VTK.

    :param str fpath_config: \
        filepath to pycis instrument configuration .yaml file.

    :param str fpath_out: \
        filepath to use for the output image.

    :param bool show_axes:
    :param bool show_cut_angle:
    :param bool show_label_details:
    :param str title:
    """

    with open(fpath_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_components = len(config['interferometer'])

    # -----
    # SETUP
    colors = vtkNamedColors()
    bkg = map(lambda x: x / 255.0, [255, 255, 255, 255])
    colors.SetColor("BkgColor", *bkg)
    iren = vtkRenderWindowInteractor()
    render_window = vtkRenderWindow()
    render_window.SetMultiSamples(2000)
    render_window.SetNumberOfLayers(3)
    render_window.SetPolygonSmoothing(1)
    render_window.SetLineSmoothing(1)
    render_window.SetAlphaBitPlanes(1)
    iren.SetRenderWindow(render_window)
    renderer_main = vtkRenderer()
    renderer_main.SetLayer(0)
    renderer_main.SetUseDepthPeeling(1)
    renderer_main.SetOcclusionRatio(0.05)
    renderer_main.SetMaximumNumberOfPeels(1000)
    renderer_main.UseDepthPeelingForVolumesOn()
    renderer_lines_fg = vtkRenderer()
    renderer_lines_fg.SetLayer(2)
    renderer_lines_bg = vtkRenderer()
    renderer_lines_bg.SetLayer(1)
    render_window.AddRenderer(renderer_main)
    render_window.AddRenderer(renderer_lines_fg)
    render_window.AddRenderer(renderer_lines_bg)

    def add_text_3d(txt, x, y, z, color='Black', renderer=renderer_main):
        """ Add 2D text at point (x, y, z)
        """
        points = vtkPoints()
        p = [x, y, z]
        points.InsertNextPoint(p)
        point = vtkPolyData()
        point.SetPoints(points)
        text_3d_mapper = vtkLabeledDataMapper()
        text_3d_mapper.SetInputData(point)
        text_3d_mapper.SetLabelFormat(txt)
        text_3d_mapper.GetLabelTextProperty().SetColor(colors.GetColor3d(color))
        text_3d_mapper.GetLabelTextProperty().SetJustificationToCentered()
        text_3d_mapper.GetLabelTextProperty().SetFontFamilyToArial()
        text_3d_mapper.GetLabelTextProperty().SetFontSize(FONTSIZE_LABEL)
        text_3d_mapper.GetLabelTextProperty().BoldOff()
        text_3d_mapper.GetLabelTextProperty().ItalicOff()
        text_3d_mapper.GetLabelTextProperty().ShadowOff()
        text_3d_mapper.GetLabelTextProperty().SetVerticalJustificationToTop()
        text_3d_actor = vtkActor2D()
        text_3d_actor.SetMapper(text_3d_mapper)
        renderer.AddActor(text_3d_actor)

    def add_line(p1, p2, line_width=1, color=COLOR_LINE_DEFAULT, renderer=renderer_main):
        lineSource = vtkLineSource()
        lineSource.SetPoint1(*p1)
        lineSource.SetPoint2(*p2)
        lineMapper = vtkPolyDataMapper()
        lineMapper.SetInputConnection(lineSource.GetOutputPort())
        lineActor = vtkActor()
        lineActor.SetMapper(lineMapper)
        lineActor.GetProperty().SetColor(colors.GetColor3d(color))
        lineActor.GetProperty().SetLineWidth(line_width)
        renderer.AddActor(lineActor)

    def add_tube(p1, p2, tube_radius=TUBE_RADIUS_DEFAULT, color=COLOR_LINE_DEFAULT, renderer=renderer_main, ):
        lineSource = vtkLineSource()
        lineSource.SetPoint1(*p1)
        lineSource.SetPoint2(*p2)
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
        tubeActor.GetProperty().ShadingOff()
        renderer.AddActor(tubeActor)

    def add_rect(p1, p2, p3, p4, color='Black', renderer=renderer_main, opacity=1.):

        points = vtkPoints()
        [points.InsertNextPoint(*p) for p in [p1, p2, p3, p4]]
        rect = vtkPolygon()
        rect.GetPointIds().SetNumberOfIds(4)
        rect.GetPointIds().SetId(0, 0)
        rect.GetPointIds().SetId(1, 1)
        rect.GetPointIds().SetId(2, 2)
        rect.GetPointIds().SetId(3, 3)
        rects = vtkCellArray()
        rects.InsertNextCell(rect)
        rectPolyData = vtkPolyData()
        rectPolyData.SetPoints(points)
        rectPolyData.SetPolys(rects)
        rect_mapper = vtkPolyDataMapper()
        rect_mapper.SetInputData(rectPolyData)
        rect_actor = vtkActor()
        rect_actor.GetProperty().SetColor(colors.GetColor3d(color))
        rect_actor.SetMapper(rect_mapper)
        rect_actor.GetProperty().SetOpacity(opacity)
        renderer.AddActor(rect_actor)

    # -------------------------
    # ADD COMPONENTS ONE BY ONE
    width_total = 0
    for ii, cc in enumerate(config['interferometer']):
        if ii != 0:
            width_total += WIDTH_SPACING

        component_type = list(cc.keys())[0]
        component_orientation_deg = cc[component_type]['orientation']
        component_orientation = component_orientation_deg * np.pi / 180
        component_width = WIDTHS[component_type]
        if component_type == 'UniaxialCrystal':
            cut_angle_deg = cc[component_type]['cut_angle']
            cut_angle = cut_angle_deg * np.pi / 180
            thickness = cc[component_type]['thickness']
            thickness_mm = thickness * 1e3
        else:
            cut_angle_deg = cut_angle = thickness = thickness_mm = NotImplemented

        # --------
        # CYLINDER
        cylinder = vtkCylinderSource()
        cylinder.SetResolution(CYLINDER_RESOLUTION)
        cylinder.SetRadius(RADIUS)
        cylinder.SetHeight(component_width)
        cylinderMapper = vtkPolyDataMapper()
        cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
        cyl_actor = vtkActor()
        cyl_actor.SetMapper(cylinderMapper)
        cyl_actor.GetProperty().SetColor(colors.GetColor3d(COLORS[component_type]))
        cyl_actor.GetProperty().BackfaceCullingOn()
        cyl_actor.GetProperty().ShadingOn()
        cyl_actor.GetProperty().SetAmbient(0.94)
        cyl_actor.GetProperty().SetDiffuse(0.03)
        cyl_actor.GetProperty().SetSpecular(0.03)
        cyl_actor.GetProperty().LightingOn()
        cyl_actor.GetProperty().SetOpacity(CYLINDER_OPACITY)

        def transform_actor(actor):
            actor.SetPosition(0.0, 0.0, 0.0)
            actor.RotateX(90.0)
            actor.SetPosition(0.0, 0.0, width_total + component_width / 2)

        transform_actor(cyl_actor)
        cylinder.Update()

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
        edge_actor.GetProperty().SetLineWidth(LINEWIDTH_CYLINDER)
        edge_actor.GetProperty().SetRenderLinesAsTubes(1)
        edge_actor.GetProperty().SetColor(colors.GetColor3d('Black'))
        edge_actor.GetProperty().LightingOff()
        edge_mapper = vtkPolyDataMapper()
        edge_mapper.SetInputConnection(feature_edges.GetOutputPort())
        edge_actor.SetMapper(edge_mapper)
        transform_actor(edge_actor)
        cylinderMapper.Update()
        edge_mapper.Update()
        renderer_main.AddActor(cyl_actor)
        renderer_main.AddActor(edge_actor)

        # ----------------
        # LINES
        view_angle = 1.14 * np.pi / 4
        nubbin = 0.05
        rad = 1.001 * RADIUS
        add_line(
            [rad * np.cos(view_angle), rad * np.sin(view_angle), width_total],
            [rad * np.cos(view_angle), rad * np.sin(view_angle), width_total + component_width + 2. * nubbin],
            renderer=renderer_lines_fg, line_width=0.9 * LINEWIDTH_CYLINDER,
        )
        add_line(
            [-rad * np.cos(view_angle), -rad * np.sin(view_angle), width_total],
            [-rad * np.cos(view_angle), -rad * np.sin(view_angle), width_total + component_width + 2. * nubbin],
            renderer=renderer_lines_fg, line_width=0.9 * LINEWIDTH_CYLINDER,
        )

        # --------------------
        # INDICATE COMPONENT ORIENTATION
        add_line(
            [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation), width_total - SMOL],
            [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation), width_total - SMOL],
            color='Red', line_width=1.5 * LINEWIDTH_CYLINDER
        )
        # add_line(
        #     [0., 1.00 * RADIUS, width_total - SMOL],
        #     [0., -1.00 * RADIUS, width_total - SMOL],
        # )
        # add_line(
        #     [1.00 * RADIUS, 0, width_total - SMOL],
        #     [-1.00 * RADIUS, 0, width_total - SMOL],
        # )

        # ---------------
        # LABEL COMPONENT
        if component_type == 'UniaxialCrystal' and show_label_details:
            if thickness_mm < 10:
                sf = 2
            else:
                sf = 3
            thickness_mm_str = str_round(thickness_mm, sf)
            component_txt = LABELS[component_type] + \
                            '\n$' + SYMBOL_ORIENTATION_ANGLE + '=' + str(component_orientation_deg) + '$°' + \
                            '\n$' + SYMBOL_CUT_ANGLE + '=' + str(cut_angle_deg) + '$°' + \
                            '\n$' + SYMBOL_THICKNESS + '=' + thickness_mm_str + '$ mm'
        else:
            component_txt = LABELS[component_type] + \
                            '\n$' + SYMBOL_ORIENTATION_ANGLE + '=$' + str(component_orientation_deg) + '°'
        add_text_3d(component_txt, -0.5 * RADIUS, -1.2 * RADIUS, width_total + component_width / 2)

        if show_cut_angle:
            if component_type == 'UniaxialCrystal':
                add_rect(
                    [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation), width_total, ],
                    [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation), width_total, ],
                    [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation),
                     width_total + component_width, ],
                    [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation),
                     width_total + component_width, ],
                    color='Black', opacity=0.6,
                )
                rad_partial = component_width * np.tan(np.pi / 2 - cut_angle)
                if rad_partial > RADIUS:
                    x_end = RADIUS * np.cos(component_orientation)
                    y_end = RADIUS * np.sin(component_orientation)
                    z_end = width_total + RADIUS * np.tan(cut_angle)
                else:
                    x_end = rad_partial * np.cos(component_orientation)
                    y_end = rad_partial * np.sin(component_orientation)
                    z_end = width_total + component_width

                # --------------
                # MARK CUT ANGLE
                arc_rad = 0.33 * RADIUS
                arc = vtkArcSource()
                arc.SetPoint1(
                    arc_rad * np.cos(component_orientation),
                    arc_rad * np.sin(component_orientation),
                    width_total
                )
                arc.SetPoint2(
                    arc_rad * np.sin(np.pi / 2 - cut_angle) * np.cos(component_orientation),
                    arc_rad * np.sin(np.pi / 2 - cut_angle) * np.sin(component_orientation),
                    width_total + arc_rad * np.cos(np.pi / 2 - cut_angle)
                )
                arc.SetCenter(
                    0,
                    0,
                    width_total
                )
                arc.SetResolution(50)
                arc_mapper = vtkPolyDataMapper()
                arc_mapper.SetInputConnection(arc.GetOutputPort())
                arc_actor = vtkActor()
                arc_actor.SetMapper(arc_mapper)
                arc_actor.GetProperty().SetColor(colors.GetColor3d('Red'))
                renderer_lines_fg.AddActor(arc_actor)

                add_tube(
                    [0, 0, width_total],
                    [x_end, y_end, z_end],
                    color='Red',
                    tube_radius=0.4 * TUBE_RADIUS_DEFAULT,
                    renderer=renderer_lines_fg,
                )

        # --------------------------------
        # HACK: re-add top edge to foreground to avoid rendering artefacts
        top_edge = vtkPolyData()
        npts = feature_edges.GetOutput().GetPoints().GetNumberOfPoints()
        top_edge_pts = vtkPoints()
        pids = []
        for ii_pt_all in range(npts):
            pt = feature_edges.GetOutput().GetPoints().GetPoint(ii_pt_all)
            if abs(pt[1] + component_width / 2) < 0.01:
                if pt[0] + pt[2] < 0.:
                    pids.append(top_edge_pts.InsertNextPoint(*pt))
        npts_out = len(pids)
        lines = vtkCellArray()
        for ii_pt in range(npts_out - 1):
            line = vtkLine()
            line.GetPointIds().SetId(0, pids[ii_pt])
            line.GetPointIds().SetId(1, pids[ii_pt + 1])
            lines.InsertNextCell(line)

        top_edge.SetPoints(top_edge_pts)
        top_edge.SetLines(lines)
        top_edge_actor = vtkActor()
        top_edge_actor.GetProperty().SetLineWidth(LINEWIDTH_CYLINDER)
        top_edge_actor.GetProperty().SetColor(colors.GetColor3d('Black'))
        transform_actor(top_edge_actor)
        top_edge_mapper = vtkPolyDataMapper()
        top_edge_mapper.SetInputData(top_edge)
        top_edge_actor.SetMapper(top_edge_mapper)
        # renderer_lines_fg.AddActor(top_edge_actor)

        width_total += component_width

    # ----------------
    # PIXELATED SENSOR
    if config['camera']['type'] == 'monochrome_polarised':
        width_total += 1.2 * WIDTH_SPACING
        sd = 1.5 * RADIUS
        sensor_depth = WIDTHS['LinearPolariser']
        sensor = vtkCubeSource()
        sensor.SetCenter(0, 0, width_total + sensor_depth / 2)
        sensor.SetXLength(sd)
        sensor.SetYLength(sd)
        sensor.SetZLength(sensor_depth)
        sensor_mapper = vtkPolyDataMapper()
        sensor_mapper.SetInputConnection(sensor.GetOutputPort())
        sensor_actor = vtkActor()
        sensor_actor.SetMapper(sensor_mapper)
        sensor_actor.GetProperty().SetColor(colors.GetColor3d(COLORS['LinearPolariser']))
        sensor_actor.GetProperty().SetRepresentationToSurface()
        sensor_actor.GetProperty().BackfaceCullingOn()
        sensor_actor.GetProperty().LightingOff()
        sensor_actor.GetProperty().SetOpacity(CYLINDER_OPACITY)
        feature_edges = vtkFeatureEdges()
        feature_edges.ColoringOff()
        feature_edges.SetInputConnection(sensor.GetOutputPort())
        feature_edges.BoundaryEdgesOn()
        feature_edges.ManifoldEdgesOff()
        feature_edges.NonManifoldEdgesOff()
        feature_edges.FeatureEdgesOff()
        edge_actor = vtkActor()
        edge_actor.GetProperty().SetLineWidth(LINEWIDTH_CYLINDER)
        edge_actor.GetProperty().SetRenderLinesAsTubes(1)
        edge_actor.GetProperty().SetRenderPointsAsSpheres(1)
        edge_actor.GetProperty().SetColor(colors.GetColor3d('Black'))
        edge_actor.GetProperty().LightingOff()
        edge_mapper = vtkPolyDataMapper()
        edge_mapper.SetInputConnection(feature_edges.GetOutputPort())
        edge_actor.SetMapper(edge_mapper)
        renderer_main.AddActor(sensor_actor)
        renderer_main.AddActor(edge_actor)

        assert npix % 2 == 0
        z = width_total - SMOL
        pd = sd / npix
        for ii_line in range(int(-npix / 2) + 1, int(npix / 2)):
            if ii_line % 2 != 0:
                lw = line_width_grid
            else:
                lw = line_width_grid_bold
            add_line(  # HORIZONTAL
                [-sd / 2, ii_line * pd, z],
                [sd / 2, ii_line * pd, z],
                line_width=lw,
            )
            add_line(  # VERTICAL
                [ii_line * pd, -sd / 2, z],
                [ii_line * pd, sd / 2, z],
                line_width=lw,
            )
        nspix = int(npix / 2)  # no. super-pixels in each dimension x & y
        for ii_x in range(nspix):
            for ii_y in range(nspix):
                add_line(  # HORIZONTAL m=0
                    [
                        -sd / 2 + (ii_x * 2 * pd),
                        -sd / 2 + (ii_y * 2 * pd) + pd / 2,
                        z
                    ],
                    [
                        -sd / 2 + (ii_x * 2 * pd) + pd,
                        -sd / 2 + (ii_y * 2 * pd) + pd / 2,
                        z
                    ],
                    color='Red', line_width=line_width_pol,
                )
                add_line(  # DIAGONAL m=1
                    [
                        -sd / 2 + (ii_x * 2 * pd) + pd,
                        -sd / 2 + (ii_y * 2 * pd),
                        z
                    ],
                    [
                        -sd / 2 + (ii_x * 2 * pd) + 2 * pd,
                        -sd / 2 + (ii_y * 2 * pd) + pd,
                        z
                    ],
                    color='Red', line_width=line_width_pol,
                )
                add_line(  # VERTICAL m=2
                    [
                        -sd / 2 + (ii_x * 2 * pd) + 3 / 2 * pd,
                        -sd / 2 + (ii_y * 2 * pd) + 2 * pd,
                        z
                    ],
                    [
                        -sd / 2 + (ii_x * 2 * pd) + 3 / 2 * pd,
                        -sd / 2 + (ii_y * 2 * pd) + pd,
                        z
                    ],
                    color='Red', line_width=line_width_pol,
                )
                add_line(  # DIAGONAL m=3
                    [
                        -sd / 2 + (ii_x * 2 * pd),
                        -sd / 2 + (ii_y * 2 * pd) + 2 * pd,
                        z
                    ],
                    [
                        -sd / 2 + (ii_x * 2 * pd) + pd,
                        -sd / 2 + (ii_y * 2 * pd) + pd,
                        z
                    ],
                    color='Red', line_width=line_width_pol,
                )

        width_total += sensor_depth + RADIUS / 4
        n_components += 1

    # ---------
    # SHOW AXIS
    if show_axes:
        edge_distance = 1.3 * RADIUS

        def add_line_axis(p1, p2, axis='x', color='Black', renderer=renderer_main):
            assert axis in ['x', 'y', 'z']
            line_source = vtkLineSource()
            line_source.SetPoint1(p1[0], p1[1], p1[2])
            line_source.SetPoint2(p2[0], p2[1], p2[2])
            line_mapper = vtkPolyDataMapper()
            line_mapper.SetInputConnection(line_source.GetOutputPort())
            line_actor = vtkActor()
            line_actor.SetMapper(line_mapper)
            line_actor.GetProperty().SetColor(colors.GetColor3d(color))
            line_actor.GetProperty().SetLineWidth(line_width_axes)
            renderer.AddActor(line_actor)

            BASE_WIDTH = 0.2
            HEIGHT = 0.25
            points = vtkPoints()
            if axis == 'x':
                points.InsertNextPoint(p2[0] - HEIGHT / 2, p2[1] - 0.5 * BASE_WIDTH, p2[2])
                points.InsertNextPoint(p2[0] - HEIGHT / 2, p2[1] + 0.5 * BASE_WIDTH, p2[2])
                points.InsertNextPoint(p2[0] + HEIGHT / 2, p2[1], p2[2])
            elif axis == 'y':
                points.InsertNextPoint(p2[0] - 0.5 * BASE_WIDTH, p2[1] - HEIGHT / 2, p2[2])
                points.InsertNextPoint(p2[0] + 0.5 * BASE_WIDTH, p2[1] - HEIGHT / 2, p2[2])
                points.InsertNextPoint(p2[0], p2[1] + HEIGHT / 2, p2[2])
            elif axis == 'z':
                points.InsertNextPoint(p2[0] - 0.5 * BASE_WIDTH, p2[1], p2[2] - HEIGHT / 2)
                points.InsertNextPoint(p2[0] + 0.5 * BASE_WIDTH, p2[1], p2[2] - HEIGHT / 2)
                points.InsertNextPoint(p2[0], p2[1], p2[2] + HEIGHT / 2)
            else:
                raise Exception

            triangle = vtkTriangle()
            triangle.GetPointIds().SetId(0, 0)
            triangle.GetPointIds().SetId(1, 1)
            triangle.GetPointIds().SetId(2, 2)
            triangles = vtkCellArray()
            triangles.InsertNextCell(triangle)
            trianglePolyData = vtkPolyData()
            trianglePolyData.SetPoints(points)
            trianglePolyData.SetPolys(triangles)
            tri_mapper = vtkPolyDataMapper()
            tri_mapper.SetInputData(trianglePolyData)
            tri_actor = vtkActor()
            tri_actor.GetProperty().SetColor(colors.GetColor3d(color))
            tri_actor.SetMapper(tri_mapper)
            renderer.AddActor(tri_actor)

        add_line_axis(  # z-axis
            [SMOL, -SMOL, -edge_distance],
            [SMOL, -SMOL, width_total + edge_distance * 0.85],
            axis='z', color=COLOR_AXIS, renderer=renderer_main
        )
        add_line_axis(  # x-axis
            [0, 0, -edge_distance],
            [RADIUS, 0, -edge_distance],
            axis='x', color=COLOR_AXIS, renderer=renderer_main
        )
        add_line_axis(  # y-axis
            [0, 0, -edge_distance],
            [0, RADIUS, -edge_distance],
            axis='y', color=COLOR_AXIS, renderer=renderer_main
        )
        add_text_3d('x', RADIUS, 0.28 * RADIUS, -edge_distance, color=COLOR_AXIS)
        add_text_3d('y', 0,  1.33 * RADIUS, -edge_distance, color=COLOR_AXIS)
        add_text_3d('z', 0,  0.3 * RADIUS, width_total + edge_distance * 0.8, color=COLOR_AXIS)

    if title is not None:
        add_text_3d(title, 1.2 * RADIUS, 1.2 * RADIUS, 0, )

    # ------
    # CAMERA
    camera = renderer_main.GetActiveCamera()
    camera.ParallelProjectionOn()  # orthographic projection
    camera.SetParallelScale(7)  # tweak as needed
    CAMERA_Z_POS = CAMERA_X_POS * np.tan(45 * np.pi / 180)
    CAMERA_Y_POS = np.sqrt(CAMERA_X_POS ** 2 + CAMERA_Z_POS ** 2) * np.tan(30 * np.pi / 180)
    camera.SetPosition(-CAMERA_X_POS, CAMERA_Y_POS, width_total / 2 - CAMERA_Z_POS)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetFocalPoint(0, 0, width_total / 2)
    renderer_lines_fg.SetActiveCamera(camera)
    renderer_lines_bg.SetActiveCamera(camera)

    renderer_main.SetBackground(colors.GetColor3d("BkgColor"))
    render_window.SetSize(3000, 2000)  # width, height
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

    # remove white-space
    im = Image.open(fpath_out)
    im = borderfy(im, border=border)
    im.save(fpath_out)

    # Start the event loop.
    # iren.Start()  # <-- UNCOMMENT LINE FOR LIVE RENDER


def imsplice(ims, overlap_frac=0.9):
    """
    splice two images together vertically, with a given fractional overlap. Assumes both images have a white background

    :param list ims: list of PIL.Images to splice together vertically
    :param float overlap_frac:
    :return: im_spliced
    """
    widths, heights = zip(*(i.size for i in ims))
    assert widths[0] == widths[1]
    total_height = int(np.array(heights[:-1]).sum() * overlap_frac + heights[-1])
    new_im = Image.new('RGBA', (widths[0], total_height))
    y_offset = 0
    for im in ims:
        im = im.convert('RGBA')
        im_blurred = im.filter(ImageFilter.GaussianBlur(20))
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
        y_offset += int(overlap_frac * im.size[1])

    background = Image.new('RGBA', new_im.size, (255, 255, 255))
    im_out = Image.alpha_composite(background, new_im).convert('RGB')
    return im_out


def borderfy(im, border=IMG_BORDER, ):
    """
    reduce / expand the image bounding box to the specified number of pixels

    :param im:
    :param border:
    :return: bordered image.
    """
    color = im.getpixel((0, 0))
    bg = Image.new(im.mode, im.size, color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if border == 0:
        return im.crop(bbox)
    else:
        border2 = 2 * border
        size_new = (bbox[2] - bbox[0] + border2, bbox[3] - bbox[1] + border2)
        im_new = Image.new(im.mode, size_new, color)
        im_new.paste(im.crop(bbox), (border, border))
        return im_new


def make_figure_2retarder_simple_configs():
    """
    splice together images to make a figure showing 3 different 2-crystal interferometer configurations
    """
    FLIST_CONFIG = sorted(glob.glob(os.path.join(FPATH_CONFIG, '2retarder_simple', '*.yaml')))
    flist_out = []
    titles = ['(a)', '(b)', '(c)', ]
    for fpath_config, title in zip(FLIST_CONFIG, titles):
        fpath_out = fpath_config.split('.')[0] + '.png'
        render_schematic(fpath_config, fpath_out, show_label_details=False, title=title)
        flist_out.append(fpath_out)

    im_spliced = imsplice([Image.open(x) for x in flist_out], overlap_frac=0.7)
    im_spliced.save('2retarder_simple_configs.png')

    for f in flist_out:
        os.remove(f)


def make_figure_2retarder_pixelated_configs():
    """
    splice together images to make a figure showing 3 different 2-crystal interferometer configurations
    """

    FLIST_CONFIG = sorted(glob.glob(os.path.join(FPATH_CONFIG, '2retarder_pixelated', '*.yaml')))
    flist_out = []
    titles = ['(a)', '(b)', '(c)', ]
    for fpath_config, title in zip(FLIST_CONFIG, titles):
        fpath_out = fpath_config.split('.')[0] + '.png'
        render_schematic(fpath_config, fpath_out, show_label_details=False, title=title)
        flist_out.append(fpath_out)

    im_spliced = imsplice([Image.open(x) for x in flist_out], overlap_frac=0.7)
    im_spliced.save('2retarder_pixelated_configs.png')

    for f in flist_out:
        os.remove(f)


def make_figure_1retarder_configs():
    """
    splice together images to make a figure showing 3 different 2-crystal interferometer configurations
    """
    from PIL import Image
    from PIL import ImageFilter

    FLIST_CONFIG = sorted(glob.glob(os.path.join(FPATH_ROOT, 'pycis_config_1retarder_*.yaml')))[::-1]
    N_IM = len(FLIST_CONFIG)
    flist_out = []
    titles = ['(a)', '(b)', ]
    for fpath_config, title in zip(FLIST_CONFIG, titles):
        fpath_out = fpath_config.split('.')[0] + '.png'
        render_schematic(fpath_config, fpath_out, title=title)
        flist_out.append(fpath_out)

    images = [Image.open(x) for x in flist_out]
    widths, heights = zip(*(i.size for i in images))
    OVERLAP_FRAC = 0.8
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
    alpha_composite.convert('RGB').save('1retarder_configs.png')

    for f in flist_out:
        os.remove(f)


def str_round(n, sf):
    """
    convert float to string, rounding to the given number of significant figures.
    from Falken's answer at
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    """
    return '{:g}'.format(float('{:.{p}g}'.format(n, p=sf)))


def make_3panel_figure_1retarder():
    fpath_config = [
        os.path.join(FPATH_CONFIG, '1retarder', 'pycis_config_1retarder_simple.yaml'),
        os.path.join(FPATH_CONFIG, '1retarder', 'pycis_config_1retarder_pixelated.yaml'),
        ]
    fpath_out = '3panel_1retarder.png'
    make_3panel_figure(fpath_config, fpath_out)


def make_3panel_figure_2retarder_simple():
    fpath_config = [
        os.path.join(FPATH_CONFIG, '2retarder_simple', 'pycis_config_2retarder_simple_2delay.yaml'),
        os.path.join(FPATH_CONFIG, '2retarder_simple', 'pycis_config_2retarder_simple_3delay.yaml'),
        os.path.join(FPATH_CONFIG, '2retarder_simple', 'pycis_config_2retarder_simple_4delay.yaml'),
        ]
    fpath_out = '3panel_2retarder_simple.png'
    make_3panel_figure(fpath_config, fpath_out)


def make_3panel_figure_2retarder_pixelated():
    fpath_config = [
        os.path.join(FPATH_CONFIG, '2retarder_pixelated', 'pycis_config_2retarder_pixelated_2delay.yaml'),
        os.path.join(FPATH_CONFIG, '2retarder_pixelated', 'pycis_config_2retarder_pixelated_3delay.yaml'),
        ]
    fpath_out = '3panel_2retarder_pixelated.png'
    make_3panel_figure(fpath_config, fpath_out)


def make_3panel_figure(fpath_config, fpath_out, label_subplots=True):
    """
    make figure showing the instrument schematic diagram + modelled interferogram + interferogram FFT.

    :param list or str fpath_config: \
        filepath to pycis instrument configuration .yaml file. Alternatively, a list of filepaths to config files.

    :param str fpath_out: \
        filepath to use for the output image.

    :param width_schematic: \
        resize schematic to a specific height, in pixels, if given.
    """
    cmap = 'gray'
    dim_show = 30  # pixel dimension of interferogram crop displayed
    dpi = 350
    bfrac = 0.13  # border width as fraction of single plot
    plot_dim_inches = 1.5
    border_inches = bfrac * plot_dim_inches
    figsize = (plot_dim_inches * (2 + bfrac), plot_dim_inches)  # inches
    hfrac = 0.5  # fractional height of row taken up by the plot
    if type(fpath_config) is str:
        fpath_config = [fpath_config, ]
    elif type(fpath_config) is not list:
        raise ValueError
    fpath_out_schem = []
    fpath_out_plot = []
    for ii, fp_config in enumerate(fpath_config):
        # ----------------
        # RENDER SCHEMATIC
        fp_out_schem = os.path.join(FPATH_TEMP, 'schematic_' + str(ii).zfill(2) + '.png')
        fp_out_plot = os.path.join(FPATH_TEMP, 'plot_' + str(ii).zfill(2) + '.png')
        render_schematic(fp_config, fp_out_schem, show_axes=True, show_cut_angle=True, show_label_details=False,
                         border=0, )
        # ------------------------
        # PLOT INTERFEROGRAM + FFT
        inst = Instrument(config=fp_config)
        igram = inst.capture(get_spectrum_delta(465e-9, 5e3), )
        psd = np.log(np.abs(fft2_im(igram)) ** 2)
        fig = plt.figure(figsize=figsize)
        axes = [fig.add_axes([0, 0, 1 / (2 + bfrac), 1, ]),
                fig.add_axes([((1 + bfrac) / (2 + bfrac)), 0, 1 / (2 + bfrac), 1, ])]
        dim = igram.shape
        igram_show = igram[
                     int(dim[0] / 2) - int(dim_show / 2):int(dim[0] / 2) + int(dim_show / 2),
                     int(dim[1] / 2) - int(dim_show / 2):int(dim[1] / 2) + int(dim_show / 2),
                     ]
        vmax = float(1.05 * igram_show.max())
        igram_show.plot(x='x', y='y', ax=axes[0], add_colorbar=False, cmap=cmap, rasterized=True, vmax=vmax,
                        xincrease=False)
        psd.plot(x='freq_x', y='freq_y', ax=axes[1], add_colorbar=False, cmap=cmap, rasterized=True,
                 xincrease=False)
        for ax in axes:
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            for sp in ax.spines:
                ax.spines[sp].set_visible(False)
        fig.savefig(fp_out_plot, bbox_inches='tight', dpi=dpi, pad_inches=0., )
        plt.cla()
        plt.clf()
        plt.close('all')
        fpath_out_schem.append(fp_out_schem)
        fpath_out_plot.append(fp_out_plot)
    # ---------------------------------
    # PAD + RESIZE IMAGES FOR STITCHING
    ims_schem_og = [Image.open(x) for x in fpath_out_schem]
    ims_plot_og = [Image.open(x) for x in fpath_out_plot]
    w_schem_og, h_schem_og = zip(*(i.size for i in ims_schem_og))
    w_schem_max = max(w_schem_og)
    h_schem_max = max(h_schem_og)
    ims_schem = []
    ims_plot = []
    for im_schem_og, im_plot_og in zip(ims_schem_og, ims_plot_og):
        # pad narrower / shorter schematic images with white space
        w_schem_og, h_schem_og = im_schem_og.size
        if w_schem_og < w_schem_max or h_schem_og < h_schem_max:
            im_schem = Image.new('RGB', (w_schem_max, h_schem_max), (255, 255, 255), )
            im_schem.paste(im_schem_og, (0, h_schem_max - h_schem_og))
            ims_schem.append(im_schem)
        else:
            ims_schem.append(im_schem_og)
        # resize plot images
        w_plot_og, h_plot_og = im_plot_og.size
        w_plot = int(w_plot_og * hfrac * h_schem_max / h_plot_og)
        h_plot = int(hfrac * h_schem_max)
        im_plot = im_plot_og.resize((w_plot, h_plot), Image.ANTIALIAS)
        ims_plot.append(im_plot)
    labels = ['(' + lttr + ')' for lttr in string.ascii_lowercase[:3 * len(fpath_config)]]
    brdr_lab = 3
    fpath_font = [i for i in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') if 'Arial.ttf' in i or 'arial.ttf' in i][0]
    font = ImageFont.truetype(fpath_font, FONTSIZE_LABEL + 2)
    brdr = int(hfrac * bfrac * h_schem_max)
    brdr2 = brdr * 2
    ims_3p = []
    for ii, (im_schem, im_plot) in enumerate(zip(ims_schem, ims_plot)):
        w_tot = im_schem.size[0] + brdr2 + im_plot.size[0] + brdr
        im_3p = Image.new('RGB', (w_tot, im_schem.size[1]), (255, 255, 255), )
        x_offset = 0
        y_offset = 0
        for im in [im_schem, im_plot]:
            im_3p.paste(im, (x_offset, y_offset))
            x_offset += im.size[0] + brdr2
            y_offset += int(h_schem_max * (1 - hfrac) / 2)
        if label_subplots:
            draw = ImageDraw.Draw(im_3p)
            h_lab = int(h_schem_max * (1 - hfrac) / 2) + 4 * brdr_lab
            w_labs = [
                4 * brdr_lab,
                w_schem_max + brdr2 + 4 * brdr_lab,
                w_schem_max + brdr2 + (w_plot - brdr) / 2 + brdr + 4 * brdr_lab
            ]
            for lab, w_lab in zip(labels[3 * ii:3 * ii + 3], w_labs):
                size = font.getsize(lab)
                draw.rectangle(xy=(w_lab - brdr_lab, h_lab - brdr_lab, w_lab + size[0] + brdr_lab, h_lab + size[1] + brdr_lab), fill=(255, 255, 255))
                draw.text((w_lab, h_lab), lab, (0, 0, 0), font=font)
        ims_3p.append(im_3p)
    im_final = imsplice(ims_3p, overlap_frac=0.85)
    im_final = borderfy(im_final, border=IMG_BORDER)
    im_final.save(fpath_out)
    # os.remove(fpath_out_schematic)
    # os.remove(fpath_out_plot)


if __name__ == '__main__':
    make_3panel_figure_2retarder_simple()
    make_3panel_figure_2retarder_pixelated()
    make_3panel_figure_1retarder()