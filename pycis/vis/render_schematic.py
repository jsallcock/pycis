import os
import numpy as np
import glob
import yaml
from PIL import Image, ImageFilter, ImageChops
from vtk import vtkFeatureEdges, vtkRenderLargeImage, vtkLabeledDataMapper, vtkActor2D, vtkAngleWidget
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

# ------------
# USER OPTIONS -- things you might want to change
SYMBOL_ORIENTATION_ANGLE = '\\rho'
SYMBOL_CUT_ANGLE = '\\theta'
SYMBOL_THICKNESS = 'L'
FONTSIZE_LABEL = 38

# ----------------
# DIAGRAM SETTINGS -- things you probably don't want to change
RADIUS = 2
WIDTH_POL = 0.5
WIDTH_RET = 1.5
PIX_HEIGHT = 300
PIX_WIDTH = 300
CYLINDER_RESOLUTION = 100
CYLINDER_OPACITY = 0.88
LINEWIDTH_CYLINDER = 5
line_width_axes = 3
TUBE_RADIUS_DEFAULT = 0.02
WIDTHS = {
    'LinearPolariser': 0.25,
    'UniaxialCrystal': 1.5,
    'QuarterWaveplate': 0.25,
}
COLORS = {
    'LinearPolariser': 'White',
    'UniaxialCrystal': 'AliceBlue',
    'QuarterWaveplate': 'Honeydew',
}
LABELS = {
    'LinearPolariser': 'POL',
    'UniaxialCrystal': 'RET',
    'QuarterWaveplate': 'QWP',
}
COLOR_LINE_DEFAULT = 'Black'
COLOR_AXIS = 'DimGray'
WIDTH_SPACING = 2.6
CAMERA_X_POS = 4.7
SMOL = 0.01 * RADIUS  # small nudges to avoid rendering artefacts
FPATH_ROOT = os.path.dirname(os.path.realpath(__file__))
FPATH_CONFIG = os.path.join(FPATH_ROOT, 'config')

# ---------------------------------
# POLARISED SENSOR DISPLAY SETTINGS
npix = 8  # no. pixels in each dimension x & y
line_width_grid = 1
line_width_grid_bold = 3
line_width_pol = 2


def render_schematic(fpath_config, fpath_out, show_axes=True, show_cut_angle=True, show_label_details=True, title=None):
    """
    Render a schematic diagram of the interferometer with 3-D isometric projection using VTK

    This is still a prototype.

    :param str fpath_config:
    :param str fpath_out:
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
    render_window.SetMultiSamples(1000)
    render_window.SetNumberOfLayers(3)
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
        cylinderActor = vtkActor()
        cylinderActor.SetMapper(cylinderMapper)
        cylinderActor.GetProperty().SetColor(colors.GetColor3d(COLORS[component_type]))
        cylinderActor.GetProperty().SetRepresentationToSurface()
        cylinderActor.GetProperty().BackfaceCullingOn()
        cylinderActor.GetProperty().LightingOff()
        cylinderActor.GetProperty().SetOpacity(CYLINDER_OPACITY)

        def transform_actor(actor):
            actor.SetPosition(0.0, 0.0, 0.0)
            actor.RotateX(90.0)
            actor.SetPosition(0.0, 0.0, width_total + component_width / 2)

        transform_actor(cylinderActor)
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
        renderer_main.AddActor(cylinderActor)
        renderer_main.AddActor(edge_actor)

        # ----------------
        # LINES
        view_angle = 1.14 * np.pi / 4
        nubbin = 0.05
        rad = 1.001 * RADIUS
        add_line(
            [rad * np.cos(view_angle), rad * np.sin(view_angle), width_total],
            [rad * np.cos(view_angle), rad * np.sin(view_angle), width_total + component_width + 3.5 * nubbin],
            renderer=renderer_lines_fg, line_width=0.9 * LINEWIDTH_CYLINDER,
        )
        add_line(
            [-rad * np.cos(view_angle), -rad * np.sin(view_angle), width_total],
            [-rad * np.cos(view_angle), -rad * np.sin(view_angle), width_total + component_width + 2.75 * nubbin],
            renderer=renderer_lines_fg, line_width=0.9 * LINEWIDTH_CYLINDER,
        )

        # --------------------
        # INDICATE COMPONENT ORIENTATION
        add_tube(
            [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation), width_total],
            [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation), width_total],
            color='Red',
            tube_radius=1.75 * TUBE_RADIUS_DEFAULT
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
                    color='Black', opacity=0.75,
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
        renderer_lines_fg.AddActor(top_edge_actor)

        width_total += component_width

    # ----------------
    # PIXELATED SENSOR
    if config['camera']['type'] == 'monochrome_pixelated':
        width_total += 1.2 * WIDTH_SPACING
        sd = 2 * RADIUS
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
        add_text_3d('x', 1.1 * RADIUS, 0, -edge_distance, color=COLOR_AXIS)
        add_text_3d('y', 0,  1.3 * RADIUS, -edge_distance, color=COLOR_AXIS)
        add_text_3d('z', 0,  0.1 * RADIUS, width_total + edge_distance, color=COLOR_AXIS)

    if title is not None:
        add_text_3d(title, 1.2 * RADIUS, 1.2 * RADIUS, 0, )

    # -------------
    # DEFINE CAMERA
    camera = renderer_main.GetActiveCamera()
    camera.ParallelProjectionOn()  # orthographic projection
    camera.SetParallelScale(5.5 + (n_components - 4) * 0.65)  # tweak as needed
    CAMERA_Z_POS = CAMERA_X_POS * np.tan(45 * np.pi / 180)
    CAMERA_Y_POS = np.sqrt(CAMERA_X_POS ** 2 + CAMERA_Z_POS ** 2) * np.tan(30 * np.pi / 180)
    camera.SetPosition(-CAMERA_X_POS, CAMERA_Y_POS, width_total / 2 - CAMERA_Z_POS)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetFocalPoint(0, 0, width_total / 2)
    renderer_lines_fg.SetActiveCamera(camera)
    renderer_lines_bg.SetActiveCamera(camera)

    renderer_main.SetBackground(colors.GetColor3d("BkgColor"))
    render_window.SetSize(n_components * 400, n_components * 250)  # width, height
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


    # remove white-space -- bit of a hack
    im = Image.open(fpath_out)
    im_blurred = im.filter(ImageFilter.GaussianBlur(30))

    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im_blurred, bg)
    # diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im = im.crop(bbox)

    im.save(fpath_out)



    # Start the event loop.
    # iren.Start()  #  <---- UNCOMMENT LINE FOR LIVE RENDER


def make_figure_2retarder_simple_configs():
    """
    splice together images to make a figure showing 3 different 2-crystal interferometer configurations
    """
    FLIST_CONFIG = sorted(glob.glob(os.path.join(FPATH_CONFIG, '2retarder_simple', '*.yaml')))
    N_IM = len(FLIST_CONFIG)
    flist_out = []
    titles = ['(a)', '(b)', '(c)', ]
    for fpath_config, title in zip(FLIST_CONFIG, titles):
        fpath_out = fpath_config.split('.')[0] + '.png'
        render_schematic(fpath_config, fpath_out, show_label_details=False, title=title)
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
    alpha_composite.convert('RGB').save('2retarder_simple_configs.png')

    for f in flist_out:
        os.remove(f)


def make_figure_2retarder_pixelated_configs():
    """
    splice together images to make a figure showing 3 different 2-crystal interferometer configurations
    """
    from PIL import Image
    from PIL import ImageFilter

    FLIST_CONFIG = sorted(glob.glob(os.path.join(FPATH_CONFIG, '2retarder_pixelated', '*.yaml')))
    N_IM = len(FLIST_CONFIG)
    flist_out = []
    titles = ['(a)', '(b)', '(c)', ]
    for fpath_config, title in zip(FLIST_CONFIG, titles):
        fpath_out = fpath_config.split('.')[0] + '.png'
        render_schematic(fpath_config, fpath_out, show_label_details=False, title=title)
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
    alpha_composite.convert('RGB').save('2retarder_pixelated_configs.png')

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


def make_figure_3panel_demo():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image
    from pycis import Instrument, get_spectrum_delta, fft2_im

    CMAP = 'gray'
    dim_show = 40

    # -------------
    # RENDER SCHEMATIC
    fpath_config = os.path.join(FPATH_CONFIG, '1retarder', 'pycis_config_1retarder_simple.yaml')
    fpath_out_schematic = os.path.join(FPATH_ROOT, '3panel_demo_schematic.png')
    fpath_out_plot = os.path.join(FPATH_ROOT, '3panel_demo_plot.png')
    render_schematic(fpath_config, fpath_out_schematic, show_axes=True, show_cut_angle=True, show_label_details=False)

    # ------------------------
    # PLOT INTERFEROGRAM + FFT
    inst = Instrument(config=fpath_config)
    igram = inst.capture(get_spectrum_delta(465e-9, 4e3), )
    psd = np.log(np.abs(fft2_im(igram)) ** 2)

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig, wspace=0.1)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]
    dim = igram.shape
    igram_show = igram[
                 int(dim[0] / 2) - int(dim_show / 2):int(dim[0] / 2) + int(dim_show / 2),
                 int(dim[1] / 2) - int(dim_show / 2):int(dim[1] / 2) + int(dim_show / 2),
                 ]
    igram_show.plot(x='x', y='y', ax=axes[0], add_colorbar=False, cmap=CMAP)
    psd.plot(x='freq_x', y='freq_y', ax=axes[1], add_colorbar=False, cmap=CMAP)
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        for sp in ax.spines:
            ax.spines[sp].set_visible(False)
    fig.savefig(fpath_out_plot, bbox_inches='tight')
    plt.close(fig)

    # -------------
    # STITCH IMAGES
    images = [Image.open(x) for x in [fpath_out_schematic, fpath_out_plot]]

    # basewidth = images[0].size[0]
    # wpercent = (basewidth / float(images[1].size[0]))
    # hsize = int((float(images[1].size[1]) * float(wpercent)))
    # images[1] = images[1].resize((basewidth, hsize), Image.ANTIALIAS)

    height0 = images[0].size[1]
    hpercent = (height0 / float(images[1].size[1]))
    wsize = int((float(images[1].size[0]) * float(hpercent)))
    images[1] = images[1].resize((wsize, height0), Image.ANTIALIAS)

    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save('test.jpg')


if __name__ == '__main__':
    make_figure_3panel_demo()
    # make_figure_2retarder_simple_configs()
    # make_figure_2retarder_pixelated_configs()