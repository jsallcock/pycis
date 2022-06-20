import copy
import os
import numpy as np
import yaml
import string
from PIL import Image, ImageFilter, ImageChops, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager
from vtk import vtkFeatureEdges, vtkRenderLargeImage, vtkLabeledDataMapper, vtkActor2D, vtkAngleWidget, vtkLight, \
    vtkSVGExporter, vtkPDFExporter
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
    vtkEllipseArcSource,
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
from pycis import Instrument, get_spectrum_delta, get_spectrum_delta_pol, fft2_im

# ----------------------------------------------------------------------------------------------------------------------
# Settings you might want to change
SYMBOL_ORIENTATION_ANGLE = '\\rho'
SYMBOL_CUT_ANGLE = '\\theta'
SYMBOL_THICKNESS = 'L'
FONTSIZE_LABEL = 58
COLORS = {
    'LinearPolariser': 'White',
    'UniaxialCrystal': 'AliceBlue',
    'QuarterWaveplate': 'AliceBlue',
    'HalfWaveplate': 'AliceBlue',
    'Lens': 'White',
    'Sensor': 'Gainsboro',
}
COLOR_ORIENTATION = 'Red'
COLOR_LIGHT = 'Red'
COLOR_LINE_DEFAULT = 'Black'
COLOR_AXES = 'DimGray'
colors = vtkNamedColors()
bkg = map(lambda x: x / 255.0, [255, 255, 255, 255])
colors.SetColor("BkgColor", *bkg)
LABELS = {
    'LinearPolariser': 'Pol',
    'UniaxialCrystal': 'Ret',
    'QuarterWaveplate': 'QWP',
    'HalfWaveplate': 'HWP',
    'Lens': 'Lens',
    'Sensor': 'Sensor',
}
# ----------------------------------------------------------------------------------------------------------------------
# Settings you probably don't want to change
# RADIUS = 1.6  # radius of interferometer components
RADIUS = 1.4  # radius of interferometer components
RADIUS_LIGHT = RADIUS  * 0.4
RADIUS_LENS = RADIUS * 0.4
WIDTH_POL = 0.5
WIDTH_RET = 1.5
PIX_HEIGHT = 300
PIX_WIDTH = 300
CYLINDER_RESOLUTION = 100
CYLINDER_OPACITY = 0.88
LINEWIDTH_CYLINDER = 4.
LINEWIDTH_ORIENTATION = 1.2 * LINEWIDTH_CYLINDER
LINEWIDTH_LIGHT = LINEWIDTH_CYLINDER
LINEWIDTH_AXIS = 3
ARROW_BASE_WIDTH_AXIS = 0.2
ARROW_HEIGHT_AXIS = 0.25
ARROW_BASE_WIDTH_LIGHT = 0.15
ARROW_HEIGHT_LIGHT = 0.2
TUBE_RADIUS_DEFAULT = 0.02
IMG_BORDER = 40  # whitespace around image, in pixels
X_LABEL = -0.45 * RADIUS
Y_LABEL = -1.25 * RADIUS
X_LABEL_LIGHT = -0. * RADIUS_LIGHT
Y_LABEL_LIGHT = - RADIUS_LIGHT - 0.3 * RADIUS
WIDTHS = {
    'LinearPolariser': 0.14,
    'UniaxialCrystal': 0.8,
    'QuarterWaveplate': 0.14,
    'HalfWaveplate': 0.14,
    'Lens': 0.3,
    'Sensor': 0.14,
}
WIDTH_SPACING = 1.6
X_CAMERA = -4.7
Z_CAMERA = -X_CAMERA * np.tan(45 * np.pi / 180)
Y_CAMERA = np.sqrt(X_CAMERA ** 2 + Z_CAMERA ** 2) * np.tan(30 * np.pi / 180)
CAMERA_PARALLEL_SCALE = 7
OVERLAP = 120
SMOL = 0.01 * RADIUS  # small nudges to avoid rendering artefacts
FPATH_ROOT = os.path.dirname(os.path.realpath(__file__))
FPATH_CONFIG = os.path.join(FPATH_ROOT, 'config')
FPATH_TEMP = os.path.join(FPATH_ROOT, 'temp')
FPATH_FONT = [i for i in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf') if 'Arial.ttf' in i or 'arial.ttf' in i][0]
FILE_EXT_IMG = '.png'
DIM_IGRAM_PIXELS = 30
CMAP_IGRAM = 'gray'
CMAP_PSD = 'gray'
DPI_IGRAM = 350
BFRAC_IGRAM = 0.13  # border width as fraction of single plot
DIM_IGRAM_INCHES = 1.5
figsize = (DIM_IGRAM_INCHES * (2 + BFRAC_IGRAM), DIM_IGRAM_INCHES)  # inches
HFRAC_IGRAM = 0.6  # fractional height of row taken up by the plot
HFRAC_IGRAM_SCAN_POL = 0.5  # fractional height of row taken up by the plot
# ----------------------------------------------------------------------------------------------------------------------
# POLARISED SENSOR DISPLAY SETTINGS
npix = 8  # no. pixels in each dimension x & y
line_width_grid = 1
line_width_grid_bold = 2
line_width_pol = 3
# ----------------------------------------------------------------------------------------------------------------------
# INCIDENT LIGHT POLARISATION STATE
POL_STATE_UNPOLARISED = {  # see e.g. https://en.wikipedia.org/wiki/Stokes_parameters for full definitions
    'p': 0.,  # degree of polarization, 0 <= p <= 1
    'psi': np.pi / 4,  # angle of polarisation
    'xi': np.pi / 8,  # angle determining degree of ellipticity
}
POL_STATE_LINEAR0 = {
    'p': 1.,
    'psi': 0.,
    'xi': 0.,
}
POL_STATE_LINEAR45 = {
    'p': 1.,
    'psi': np.pi / 4,
    'xi': 0.,
}
POL_STATE_RHC = {
    'p': 1.,
    'psi': 0.,
    'xi': np.pi / 4,
}
POL_STATE_DEFAULT = POL_STATE_UNPOLARISED


# ----------------------------------------------------------------------------------------------------------------------
# FNS FOR MAKING DIAGRAMS

def render_schematic(fpath_config, fpath_out, show_axes=True, show_cut_angle=True, show_label_details=False,
                     pol_state=None, show_label_pol_state=True, index=0, border=IMG_BORDER):
    """
    Render a schematic diagram of the interferometer with 3-D isometric projection using VTK.

    Uses some dirty short-cuts, so won't look good if e.g. the camera position / projection method is changed.

    :param str fpath_config: \
        filepath to pycis instrument configuration .yaml file.

    :param str fpath_out: \
        filepath to use for the output image.

    :param show_axes:
    :param show_cut_angle:
    :param show_label_details:
    :param pol_state:
    :param show_label_pol_state:
    :param border:
    :return:
    """
    with open(fpath_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    n_components = len(config['interferometer'])

    renderer_main, render_window, render_window_interactor = get_renderer_default()
    renderer_fg = vtkRenderer()
    renderer_fg.SetLayer(2)
    renderer_bg = vtkRenderer()
    renderer_bg.SetLayer(1)
    render_window.AddRenderer(renderer_fg)
    render_window.AddRenderer(renderer_bg)

    # ------------------------------------------------------------------------------------------------------------------
    # ADD INTERFEROMETER COMPONENTS ONE BY ONE
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

        # CYLINDER
        color_cyl = COLORS[component_type]
        add_cylinder(width_total, WIDTHS[component_type], RADIUS, renderer=renderer_main, renderer_edges=renderer_fg, color=color_cyl)

        # INDICATE COMPONENT ORIENTATION
        add_line(
            [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation), width_total - SMOL],
            [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation), width_total - SMOL],
            color=COLOR_ORIENTATION, line_width=LINEWIDTH_ORIENTATION, renderer=renderer_main
        )

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
                            '\n' + str(component_orientation_deg) + '°'
                            # '\n$' + SYMBOL_ORIENTATION_ANGLE + '=$' + str(component_orientation_deg) + '°'

        add_text_3d(component_txt, [X_LABEL, Y_LABEL, width_total + component_width / 2], renderer=renderer_main)

        if show_cut_angle:
            if component_type in ['UniaxialCrystal', ]:
                add_rect(
                    [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation), width_total, ],
                    [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation), width_total, ],
                    [-RADIUS * np.cos(component_orientation), -RADIUS * np.sin(component_orientation),
                     width_total + component_width, ],
                    [RADIUS * np.cos(component_orientation), RADIUS * np.sin(component_orientation),
                     width_total + component_width, ],
                    color='Black', opacity=0.6, renderer=renderer_main,
                )
                if component_type == 'UniaxialCrystal':
                    rad_partial = component_width * np.tan(np.pi / 2 - cut_angle)
                    if rad_partial > RADIUS:
                        x_end = RADIUS * np.cos(component_orientation)
                        y_end = RADIUS * np.sin(component_orientation)
                        z_end = width_total + RADIUS * np.tan(cut_angle)
                    else:
                        x_end = rad_partial * np.cos(component_orientation)
                        y_end = rad_partial * np.sin(component_orientation)
                        z_end = width_total + component_width
                    # ------------------------------------------------------------------------------------------------------
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
                    renderer_fg.AddActor(arc_actor)

                    add_tube(
                        [0, 0, width_total],
                        [x_end, y_end, z_end],
                        color='Red',
                        tube_radius=0.4 * TUBE_RADIUS_DEFAULT,
                        renderer=renderer_fg,
                    )
        # --------------------------------------------------------------------------------------------------------------
        # HACK: re-add top edge to foreground to avoid rendering artefacts
        # top_edge = vtkPolyData()
        # npts = feature_edges.GetOutput().GetPoints().GetNumberOfPoints()
        # top_edge_pts = vtkPoints()
        # pids = []
        # for ii_pt_all in range(npts):
        #     pt = feature_edges.GetOutput().GetPoints().GetPoint(ii_pt_all)
        #     if abs(pt[1] + component_width / 2) < 0.01:
        #         if pt[0] + pt[2] < 0.:
        #             pids.append(top_edge_pts.InsertNextPoint(*pt))
        # npts_out = len(pids)
        # lines = vtkCellArray()
        # for ii_pt in range(npts_out - 1):
        #     line = vtkLine()
        #     line.GetPointIds().SetId(0, pids[ii_pt])
        #     line.GetPointIds().SetId(1, pids[ii_pt + 1])
        #     lines.InsertNextCell(line)
        #
        # top_edge.SetPoints(top_edge_pts)
        # top_edge.SetLines(lines)
        # top_edge_actor = vtkActor()
        # top_edge_actor.GetProperty().SetLineWidth(LINEWIDTH_CYLINDER)
        # top_edge_actor.GetProperty().SetColor(colors.GetColor3d('Black'))
        # transform_actor(top_edge_actor)
        # top_edge_mapper = vtkPolyDataMapper()
        # top_edge_mapper.SetInputData(top_edge)
        # top_edge_actor.SetMapper(top_edge_mapper)
        # renderer_lines_fg.AddActor(top_edge_actor)

        width_total += component_width

    # ------------------------------------------------------------------------------------------------------------------
    # LENS
    width_total += WIDTH_SPACING * 1.1
    add_cylinder(width_total, WIDTHS['Lens'], RADIUS_LENS, renderer=renderer_main, renderer_edges=renderer_fg, color=COLORS['Lens'])
    # if index == 0:
    #     add_text_3d(LABELS['Lens'], [X_LABEL, Y_LABEL, width_total + WIDTHS['Lens'] / 2], renderer=renderer_main)
    width_total += WIDTHS['Lens']

    # ------------------------------------------------------------------------------------------------------------------
    # SENSOR
    focal_length = WIDTH_SPACING * 0.3
    width_total += focal_length
    sd = 1. * RADIUS
    sensor_depth = WIDTHS['Sensor']
    add_text_3d('Camera', [X_LABEL, Y_LABEL, width_total - focal_length / 2], renderer=renderer_main)

    if config['camera']['type'] == 'monochrome_polarised':
        gap = 1.4 * sensor_depth
        add_sensor(width_total, WIDTHS['LinearPolariser'], sd, npix, renderer=renderer_main,
                   color=COLORS['LinearPolariser'], opacity=CYLINDER_OPACITY)
        add_sensor(width_total + WIDTHS['LinearPolariser'] + gap, sensor_depth, sd, npix, renderer=renderer_main,
                   color=COLORS['Sensor'])
        assert npix % 2 == 0
        z = width_total - SMOL
        pd = sd / npix
        pd2 = 2 * pd
        for ii_line in range(int(-npix / 2) + 1, int(npix / 2)):
            if ii_line % 2 == 0:
                lw = line_width_grid_bold
                add_line(  # HORIZONTAL
                    [-sd / 2, ii_line * pd, z],
                    [sd / 2, ii_line * pd, z],
                    line_width=lw, renderer=renderer_main,
                )
                add_line(  # VERTICAL
                    [ii_line * pd, -sd / 2, z],
                    [ii_line * pd, sd / 2, z],
                    line_width=lw, renderer=renderer_main,
                )
        nspix = int(npix / 2)  # no. super-pixels in each dimension x & y
        kwargs_sensor_pol = {
            'color': 'Red',
            'line_width': line_width_pol,
            'renderer': renderer_main,
        }
        for ii_x in range(nspix):
            for ii_y in range(nspix):
                add_line(  # HORIZONTAL m=0
                    [-sd / 2 + (ii_x * pd2), -sd / 2 + (ii_y * pd2) + pd / 2, z],
                    [-sd / 2 + (ii_x * pd2) + pd, -sd / 2 + (ii_y * pd2) + pd / 2, z],
                    **kwargs_sensor_pol,
                )
                add_line(  # DIAGONAL m=1
                    [-sd / 2 + (ii_x * pd2) + pd, -sd / 2 + (ii_y * pd2), z],
                    [-sd / 2 + (ii_x * pd2) + pd2, -sd / 2 + (ii_y * pd2) + pd, z],
                    **kwargs_sensor_pol,
                )
                add_line(  # VERTICAL m=2
                    [-sd / 2 + (ii_x * pd2) + 3 / 2 * pd, -sd / 2 + (ii_y * pd2) + pd2, z],
                    [-sd / 2 + (ii_x * pd2) + 3 / 2 * pd, -sd / 2 + (ii_y * pd2) + pd, z],
                    **kwargs_sensor_pol,
                )
                add_line(  # DIAGONAL m=3
                    [-sd / 2 + (ii_x * pd2), -sd / 2 + (ii_y * pd2) + pd2, z],
                    [-sd / 2 + (ii_x * pd2) + pd, -sd / 2 + (ii_y * pd2) + pd, z],
                    **kwargs_sensor_pol,
                )
        width_total += (WIDTHS['LinearPolariser'] + sensor_depth + gap)
    else:
        add_sensor(width_total, sensor_depth, sd, npix, renderer=renderer_main, color=COLORS['Sensor'])
        width_total += sensor_depth
    # if index == 0:


    n_components += 1

    # ------------------------------------------------------------------------------------------------------------------
    # COORDINATE AXES
    edge_distance = 1.35 * RADIUS
    if show_axes:
        end = edge_distance * 0.65
        kwargs = {
            'renderer': renderer_main,
            'color': COLOR_AXES
        }
        add_axis(  # z-axis
            [SMOL, -SMOL, -edge_distance],
            [SMOL, -SMOL, width_total + end],
            axis='z', **kwargs
        )
        add_axis(  # x-axis
            [0, 0, -edge_distance],
            [RADIUS, 0, -edge_distance],
            axis='x', **kwargs
        )
        add_axis(  # y-axis
            [0, 0, -edge_distance],
            [0, RADIUS, -edge_distance],
            axis='y', **kwargs
        )
        add_text_3d('x', [RADIUS, 0.36 * RADIUS, -edge_distance], **kwargs)
        add_text_3d('y', [0, 1.42 * RADIUS, -edge_distance], **kwargs)
        add_text_3d('z', [0, 0.33 * RADIUS, width_total + end], **kwargs)
    # ------------------------------------------------------------------------------------------------------------------
    # INCIDENT LIGHT
    if pol_state is not None:
        add_pol_state(pol_state, -edge_distance, RADIUS_LIGHT, renderer_main, show_label=show_label_pol_state)  # TODO

    p_camera = [X_CAMERA, Y_CAMERA, width_total / 2 - Z_CAMERA]
    p_focal = [0, 0, width_total / 2]
    camera = get_camera(p_camera, p_focal, renderer=[renderer_main, renderer_fg, renderer_bg])
    render_image(render_window, fpath_out)
    borderfy(Image.open(fpath_out), border=border).save(fpath_out)

    # Start the event loop.
    # iren.Start()  # <-- UNCOMMENT LINE FOR LIVE RENDER


def make_3panel_figure(fpath_config, fpath_out, label_subplots=True, pol_state=None, delete_subfigures=True):
    """
    given instrument config file(s), make figure(s) showing schematic diagram + modelled interferogram + FFT.

    :param list or str fpath_config: \
        filepath to pycis instrument configuration .yaml file. Alternatively, a list of filepaths to config files.

    :param str fpath_out: \
        filepath to use for the output image.

    :param bool label_subplots: \
        labels the figure subplots '(a)', '(b)', '(c)' etc.

    :param dict or list of dicts pol_state: \
        Specify polarisation state of the incident light. The default value None does not render any incident light.
    """

    # PARSE INPUTS
    if type(fpath_config) is str:
        fpath_config = [fpath_config, ]
    elif type(fpath_config) is not list:
        raise ValueError

    if type(pol_state) is dict:
        pol_state = [pol_state, ]
    elif type(pol_state) is not list:
        raise ValueError

    n_config = len(fpath_config)
    n_pol_state = len(pol_state)
    if n_config == 1 and n_pol_state > 1:
        mode = 'scan_polarisation'  # no need to render and show the same configuration multiple times
    elif n_config > 1 and n_pol_state == 1:
        mode = 'scan_configuration'
    else:
        mode = 'general'

    print('mode:', mode)
    if mode == 'scan_polarisation':
        fpath_out_pol = []
        fpath_out_igram = []
        fp_config = fpath_config[0]
        for ii, p_state in enumerate(pol_state):
            fp_out_pol = os.path.join(FPATH_TEMP, 'polarisation_state_' + str(ii).zfill(2) + FILE_EXT_IMG)
            fp_out_igram = os.path.join(FPATH_TEMP, 'interferogram_' + str(ii).zfill(2) + FILE_EXT_IMG)
            show_label_pol_state = False
            render_pol_state(p_state, fp_out_pol, show_label_pol_state=show_label_pol_state, border=0)
            render_igram(fp_config, fp_out_igram, pol_state=p_state)

            fpath_out_pol.append(fp_out_pol)
            fpath_out_igram.append(fp_out_igram)
        fp_out_schem = os.path.join(FPATH_TEMP, 'schematic_' + str(ii).zfill(2) + FILE_EXT_IMG)
        render_schematic(fp_config, fp_out_schem, show_axes=True, show_cut_angle=True, show_label_details=False,
                         pol_state=None, border=0)
        # --------------------------------------------------------------------------------------------------------------
        # PAD + RESIZE IMAGES FOR STITCHING
        ims_pol_og = [Image.open(x) for x in fpath_out_pol]
        ims_plot_og = [Image.open(x) for x in fpath_out_igram]
        w_pol_og, h_pol_og = zip(*(i.size for i in ims_pol_og))
        w_pol_max = max(w_pol_og)
        # h_pol_max = max(h_pol_og)
        h_pol_max = 1000
        ims_pol = []
        ims_plot = []
        for im_pol_og, im_plot_og in zip(ims_pol_og, ims_plot_og):
            # pad narrower / shorter schematic images with white space
            w_pol_og, h_pol_og = im_pol_og.size
            if w_pol_og < w_pol_max or h_pol_og < h_pol_max:
                im_pol = Image.new('RGB', (w_pol_max, h_pol_max), (255, 255, 255), )
                im_pol.paste(im_pol_og, (0, int(h_pol_max / 2 - h_pol_og / 2)))
                ims_pol.append(im_pol)
            else:
                ims_pol.append(im_pol_og)
            # resize plot images
            w_plot_og, h_plot_og = im_plot_og.size
            w_plot = int(w_plot_og * HFRAC_IGRAM * h_pol_max / h_plot_og)
            h_plot = int(HFRAC_IGRAM * h_pol_max)
            im_plot = im_plot_og.resize((w_plot, h_plot), Image.ANTIALIAS)
            ims_plot.append(im_plot)
        # --------------------------------------------------------------------------------------------------------------
        # ADD LABELS + COMBINE
        labels = ['(' + lttr + ')' for lttr in string.ascii_lowercase[:3 * len(fpath_config)]]
        brdr_lab = 3
        font = ImageFont.truetype(FPATH_FONT, FONTSIZE_LABEL + 2)
        brdr = int(HFRAC_IGRAM * BFRAC_IGRAM * h_pol_max)
        brdr2 = brdr * 2
        ims_all = []
        for ii, (im_schem, im_plot) in enumerate(zip(ims_pol, ims_plot)):
            w_tot = im_schem.size[0] + brdr2 + im_plot.size[0] + brdr
            im_3p = Image.new('RGB', (w_tot, im_schem.size[1]), (255, 255, 255), )
            x_offset = 0
            y_offset = 0
            for im in [im_schem, im_plot]:
                im_3p.paste(im, (x_offset, y_offset))
                x_offset += im.size[0] + brdr2
                y_offset += int(h_pol_max * (1 - HFRAC_IGRAM) / 2)
            if label_subplots:
                draw = ImageDraw.Draw(im_3p)
                h_lab = int(h_pol_max * (1 - HFRAC_IGRAM) / 2) + 4 * brdr_lab
                w_labs = [
                    4 * brdr_lab,
                    w_pol_max + brdr2 + 4 * brdr_lab,
                    w_pol_max + brdr2 + (w_plot - brdr) / 2 + brdr + 4 * brdr_lab
                ]
                for lab, w_lab in zip(labels[3 * ii:3 * ii + 3], w_labs):
                    size = font.getsize(lab)
                    draw.rectangle(xy=(
                    w_lab - brdr_lab, h_lab - brdr_lab, w_lab + size[0] + brdr_lab, h_lab + size[1] + brdr_lab),
                                   fill=(255, 255, 255))
                    draw.text((w_lab, h_lab), lab, (0, 0, 0), font=font)
            im_3p = borderfy(im_3p, border=int(brdr / 2))
            ims_all.append(im_3p)
        # resize schematic width
        im_schem_og = Image.open(fp_out_schem)
        width_new = im_3p.width
        im_schem = Image.new('RGB', (width_new, im_schem_og.height), (255, 255, 255), )
        im_schem.paste(im_schem_og, (int(width_new / 2 - im_schem_og.width / 2), 0))
        ims_all.insert(0, im_schem)

        im_final = imsplice(ims_all)
        im_final = borderfy(im_final, border=IMG_BORDER)
        im_final.save(fpath_out, dpi=(DPI_IGRAM, DPI_IGRAM))
        for fp_out_schem, fp_out_igram in zip(fpath_out_pol, fpath_out_igram):
            os.remove(fp_out_schem)
            os.remove(fp_out_igram)

    elif mode == 'scan_configuration' or mode == 'general':
        if n_pol_state == 1:
            pol_state *= n_config
        fpath_out_schem = []
        fpath_out_igram = []
        for ii, fp_config in enumerate(fpath_config):
            fp_out_schem = os.path.join(FPATH_TEMP, 'schematic_' + str(ii).zfill(2) + '.png')
            fp_out_igram = os.path.join(FPATH_TEMP, 'interferogram_' + str(ii).zfill(2) + '.png')
            if ii == 0:
                show_label_pol_state = True
            else:
                show_label_pol_state = False
            render_schematic(fp_config, fp_out_schem, show_axes=True, show_cut_angle=True, show_label_details=False,
                             pol_state=pol_state[ii], show_label_pol_state=show_label_pol_state, border=0, index=ii)
            render_igram(fp_config, fp_out_igram, pol_state=pol_state[ii], )

            fpath_out_schem.append(fp_out_schem)
            fpath_out_igram.append(fp_out_igram)

        # --------------------------------------------------------------------------------------------------------------
        # PAD + RESIZE IMAGES FOR STITCHING
        ims_schem_og = [Image.open(x) for x in fpath_out_schem]
        ims_plot_og = [Image.open(x) for x in fpath_out_igram]
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
            w_plot = int(w_plot_og * HFRAC_IGRAM * h_schem_max / h_plot_og)
            h_plot = int(HFRAC_IGRAM * h_schem_max)
            im_plot = im_plot_og.resize((w_plot, h_plot), Image.ANTIALIAS)
            ims_plot.append(im_plot)
        # --------------------------------------------------------------------------------------------------------------
        # ADD LABELS + COMBINE
        labels = ['(' + lttr + ')' for lttr in string.ascii_lowercase[:3 * len(fpath_config)]]
        brdr_lab = 3
        font = ImageFont.truetype(FPATH_FONT, FONTSIZE_LABEL + 2)
        brdr = int(HFRAC_IGRAM * BFRAC_IGRAM * h_schem_max)
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
                y_offset += int(h_schem_max * (1 - HFRAC_IGRAM) / 2)
            if label_subplots:
                draw = ImageDraw.Draw(im_3p)
                h_lab = int(h_schem_max * (1 - HFRAC_IGRAM) / 2) + 4 * brdr_lab
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

        im_final = imsplice(ims_3p, border=4 * IMG_BORDER)
        im_final = borderfy(im_final, border=IMG_BORDER)
        im_final.save(fpath_out)
        if delete_subfigures:
            for fp_out_schem, fp_out_igram in zip(fpath_out_schem, fpath_out_igram):
                os.remove(fp_out_schem)
                os.remove(fp_out_igram)
    else:
        raise NotImplementedError


def render_igram(fpath_config, fpath_out, pol_state, ):
    """
    Render modelled interferogram with corresponding power spectral density.
    """
    wl0 = 465e-9
    inst = Instrument(config=fpath_config)
    inst_roi = copy.deepcopy(inst)
    inst_roi.camera.sensor_format = (DIM_IGRAM_PIXELS, DIM_IGRAM_PIXELS)
    xx, yy = inst_roi.camera.get_pixel_position()
    inst_roi.camera.x = xx
    inst_roi.camera.y = yy

    spec = get_spectrum_delta_pol(wl0, 5e3, pol_state['p'], pol_state['psi'], pol_state['xi'])

    # Gaussian brightness profile to make the DC term stand out in the Fourier domain
    sigma = 15 * inst.camera.pixel_size
    spec_g = spec * np.exp(-1 / 2 * (inst.camera.x / sigma) ** 2)
    spec_g = spec_g * np.exp(-1 / 2 * (inst.camera.y / sigma) ** 2)
    igram_roi = inst_roi.capture(spec, )
    igram_g = inst.capture(spec_g, )

    # from pycis import get_pixelated_phase_mask
    # pm = get_pixelated_phase_mask(igram_g.shape)
    # sp = igram_g * np.exp(-1j * pm)
    # psd_sp = np.log(np.abs(fft2_im(sp)) ** 2)
    # f = plt.figure()
    # a = f.add_subplot()
    # psd_sp.plot(x='freq_x', y='freq_y', ax=a)
    # plt.show(block=True)

    psd = np.log(np.abs(fft2_im(igram_g)) ** 2)
    fig = plt.figure(figsize=figsize)
    axes = [fig.add_axes([0, 0, 1 / (2 + BFRAC_IGRAM), 1, ]),
            fig.add_axes([((1 + BFRAC_IGRAM) / (2 + BFRAC_IGRAM)), 0, 1 / (2 + BFRAC_IGRAM), 1, ])]

    vmax = float(1.05 * igram_roi.max())
    igram_roi.plot(x='x', y='y', ax=axes[0], add_colorbar=False, cmap=CMAP_IGRAM, rasterized=True, vmin=0, vmax=vmax,
                   xincrease=False)
    psd.plot(x='freq_x', y='freq_y', ax=axes[1], add_colorbar=False, cmap=CMAP_PSD, rasterized=True,
             xincrease=False)

    # offset = float(psd.freq_x.max() / 10)
    displacers = [r for r in inst.retarders if hasattr(r, 'cut_angle')]
    displacers = [r for r in displacers if r.cut_angle > 0]
    print('len(displacers) =', len(displacers))

    for displacer in displacers:
        kx, ky = displacer.get_fringe_frequency(wl0, inst.optics[2])
        print('displacer.orientation', displacer.orientation)
        print('kx=', kx)
        print('ky=', ky)
        axes[1].arrow(0, 0, kx, ky, color='r')

    # axes[1].annotate(text='Intensity', xy=(offset, offset))

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        for sp in ax.spines:
            ax.spines[sp].set_visible(False)
    fig.savefig(fpath_out, bbox_inches='tight', dpi=DPI_IGRAM, pad_inches=0., )
    plt.cla()
    plt.clf()
    plt.close('all')


def render_pol_state(pol_state, fpath_out, show_label_pol_state=True, border=IMG_BORDER):
    """
    Render schematic showing coordinate axes and polarisation state of the incident light.

    :param pol_state:
    :param fpath_out:
    :param show_label_pol_state:
    :param border:
    """
    renderer, render_window, render_window_interactor = get_renderer_default()
    renderer_fg = vtkRenderer()
    renderer_fg.SetLayer(2)
    render_window.AddRenderer(renderer_fg)
    p_camera = [X_CAMERA, Y_CAMERA, -Z_CAMERA]
    p_focal = [0, 0, 0]
    camera = get_camera(p_camera, p_focal, renderer=[renderer, renderer_fg])

    # ADD AXES
    kwargs = {
        'renderer': renderer,
        'color': COLOR_AXES
    }
    origin = [0, 0, 0]
    add_axis(origin, [0, 0, RADIUS], axis='z', **kwargs)
    add_axis(origin, [RADIUS, 0, 0], axis='x', **kwargs)
    add_axis(origin, [0, RADIUS, 0], axis='y', **kwargs)
    add_text_3d('x', [RADIUS, 0.33 * RADIUS, 0], **kwargs)
    add_text_3d('y', [0, 1.44 * RADIUS, 0], **kwargs)
    add_text_3d('z', [0, 0.3 * RADIUS, RADIUS], **kwargs)
    add_pol_state(
        pol_state=pol_state,
        z=0,
        radius=RADIUS_LIGHT,
        renderer=renderer_fg,
        show_label=show_label_pol_state,
    )
    render_image(render_window, fpath_out)
    borderfy(Image.open(fpath_out), border=border).save(fpath_out)


# ----------------------------------------------------------------------------------------------------------------------
# LOW-LEVEL VTK FNS


def add_line(p1, p2, renderer, line_width=1., color=COLOR_LINE_DEFAULT):
    """ Add line specified by 2 points
    """
    source = vtkLineSource()
    source.SetPoint1(*p1)
    source.SetPoint2(*p2)
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.GetProperty().SetLineWidth(line_width)
    renderer.AddActor(actor)


def add_tube(p1, p2, renderer, tube_radius=TUBE_RADIUS_DEFAULT, color=COLOR_LINE_DEFAULT):
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


def add_rect(p1, p2, p3, p4, renderer, color='Black', opacity=1.):
    """ Add rectangle specified by 4 points
    """
    points = vtkPoints()
    [points.InsertNextPoint(*p) for p in [p1, p2, p3, p4]]
    rect = vtkPolygon()
    rect.GetPointIds().SetNumberOfIds(4)
    [rect.GetPointIds().SetId(i, i) for i in range(4)]
    rects = vtkCellArray()
    rects.InsertNextCell(rect)
    poly_data = vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(rects)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtkActor()
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)


def add_tri(p1, p2, p3, renderer, color='Black', opacity=1.):
    """ Add triangle specified by 3 points
    """
    tri_points = vtkPoints()
    [tri_points.InsertNextPoint(*p) for p in [p1, p2, p3]]
    tri = vtkTriangle()
    [tri.GetPointIds().SetId(i, i) for i in range(3)]
    tris = vtkCellArray()
    tris.InsertNextCell(tri)
    poly_data = vtkPolyData()
    poly_data.SetPoints(tri_points)
    poly_data.SetPolys(tris)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtkActor()
    actor.GetProperty().SetColor(colors.GetColor3d(color))
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)


def add_text_3d(txt, p, renderer, font_size=FONTSIZE_LABEL, color='Black',):
    """ Add 2-D text at given point
    """
    points = vtkPoints()
    points.InsertNextPoint(*p)
    point = vtkPolyData()
    point.SetPoints(points)
    text_3d_mapper = vtkLabeledDataMapper()
    text_3d_mapper.SetInputData(point)
    text_3d_mapper.SetLabelFormat(txt)
    text_3d_mapper.GetLabelTextProperty().SetColor(colors.GetColor3d(color))
    text_3d_mapper.GetLabelTextProperty().SetJustificationToCentered()
    # text_3d_mapper.GetLabelTextProperty().SetFontFamilyToArial()
    text_3d_mapper.GetLabelTextProperty().SetFontSize(font_size)
    text_3d_mapper.GetLabelTextProperty().BoldOff()
    text_3d_mapper.GetLabelTextProperty().ItalicOff()
    text_3d_mapper.GetLabelTextProperty().ShadowOff()
    text_3d_mapper.GetLabelTextProperty().SetVerticalJustificationToTop()
    text_3d_actor = vtkActor2D()
    text_3d_actor.SetMapper(text_3d_mapper)
    renderer.AddActor(text_3d_actor)


def add_axis(p1, p2, renderer, axis='x', color='Black', ):
    assert axis in ['x', 'y', 'z']
    line_source = vtkLineSource()
    line_source.SetPoint1(p1[0], p1[1], p1[2])
    line_source.SetPoint2(p2[0], p2[1], p2[2])
    line_mapper = vtkPolyDataMapper()
    line_mapper.SetInputConnection(line_source.GetOutputPort())
    line_actor = vtkActor()
    line_actor.SetMapper(line_mapper)
    line_actor.GetProperty().SetColor(colors.GetColor3d(color))
    line_actor.GetProperty().SetLineWidth(LINEWIDTH_AXIS)
    renderer.AddActor(line_actor)
    if axis == 'x':
        tri_p1 = [p2[0] - ARROW_HEIGHT_AXIS / 2, p2[1] - 0.5 * ARROW_BASE_WIDTH_AXIS, p2[2]]
        tri_p2 = [p2[0] - ARROW_HEIGHT_AXIS / 2, p2[1] + 0.5 * ARROW_BASE_WIDTH_AXIS, p2[2]]
        tri_p3 = [p2[0] + ARROW_HEIGHT_AXIS / 2, p2[1], p2[2]]
    elif axis == 'y':
        tri_p1 = [p2[0] - 0.5 * ARROW_BASE_WIDTH_AXIS, p2[1] - ARROW_HEIGHT_AXIS / 2, p2[2]]
        tri_p2 = [p2[0] + 0.5 * ARROW_BASE_WIDTH_AXIS, p2[1] - ARROW_HEIGHT_AXIS / 2, p2[2]]
        tri_p3 = [p2[0], p2[1] + ARROW_HEIGHT_AXIS / 2, p2[2]]
    elif axis == 'z':
        tri_p1 = [p2[0] - 0.5 * ARROW_BASE_WIDTH_AXIS, p2[1], p2[2] - ARROW_HEIGHT_AXIS / 2]
        tri_p2 = [p2[0] + 0.5 * ARROW_BASE_WIDTH_AXIS, p2[1], p2[2] - ARROW_HEIGHT_AXIS / 2]
        tri_p3 = [p2[0], p2[1], p2[2] + ARROW_HEIGHT_AXIS / 2]
    else:
        raise Exception
    add_tri(tri_p1, tri_p2, tri_p3, color=color, renderer=renderer)


def add_pol_state(pol_state, z, radius, renderer, show_label=True, ):
    """
    :param pol_state:
    :param z:
    :param radius:
    :param angle:
    :param renderer:
    :param show_label:
    """
    # UNPOLARISED
    if pol_state['p'] == 0:
        N_POL = 5
        for ii_pol in range(N_POL):
            add_pol_state_linear(z, radius, ii_pol * np.pi / N_POL, renderer)

    # LINEAR
    elif pol_state['xi'] == 0:
        add_pol_state_linear(z, radius, pol_state['psi'], renderer)

    # ELLIPTICAL
    # TODO add arrow showing appropriate direction
    else:
        ell = vtkEllipseArcSource()
        ell.SetResolution(CYLINDER_RESOLUTION)
        ell.SetCenter(0, 0, 0)
        ell.SetNormal(0, 0, 1)
        ell.SetMajorRadiusVector(radius * np.cos(pol_state['psi']), radius * np.sin(pol_state['psi']), 0)
        ell.SetRatio(np.tan(pol_state['xi']))
        ell.SetSegmentAngle(360)
        ell_mapper = vtkPolyDataMapper()
        ell_mapper.SetInputConnection(ell.GetOutputPort())
        ell_actor = vtkActor()
        ell_actor.SetMapper(ell_mapper)
        ell_actor.GetProperty().SetColor(colors.GetColor3d(COLOR_LIGHT))
        ell_actor.GetProperty().ShadingOn()
        ell_actor.GetProperty().LightingOn()
        ell_actor.GetProperty().SetLineWidth(LINEWIDTH_LIGHT)
        renderer.AddActor(ell_actor)

    if show_label:
        add_text_3d('Incident\nlight', [X_LABEL_LIGHT, Y_LABEL_LIGHT, z], renderer=renderer)


def add_pol_state_linear(z, radius, angle, renderer):
    """
    :param z:
    :param radius:
    :param angle: polarisation angle (from x-axis) in radians
    :param renderer:
    """
    p1 = [radius * np.cos(angle), radius * np.sin(angle), z]
    p2 = [-radius * np.cos(angle), -radius * np.sin(angle), z]
    add_line(p1, p2, line_width=LINEWIDTH_LIGHT, color=COLOR_LIGHT, renderer=renderer)
    tri2_p1 = [
        p2[0] + (ARROW_HEIGHT_LIGHT / 2) * np.cos(angle) + (ARROW_BASE_WIDTH_LIGHT / 2) * np.sin(angle),
        p2[1] + (ARROW_HEIGHT_LIGHT / 2) * np.sin(angle) - (ARROW_BASE_WIDTH_LIGHT / 2) * np.cos(angle),
        z,
    ]
    tri2_p2 = [
        p2[0] + (ARROW_HEIGHT_LIGHT / 2) * np.cos(angle) - (ARROW_BASE_WIDTH_LIGHT / 2) * np.sin(angle),
        p2[1] + (ARROW_HEIGHT_LIGHT / 2) * np.sin(angle) + (ARROW_BASE_WIDTH_LIGHT / 2) * np.cos(angle),
        z,
    ]
    tri2_p3 = [
        p2[0] - (ARROW_HEIGHT_LIGHT / 2) * np.cos(angle),
        p2[1] - (ARROW_HEIGHT_LIGHT / 2) * np.sin(angle),
        z,
    ]
    tri2_ps = [tri2_p1, tri2_p2, tri2_p3]
    tri1_ps = [[-p[0], -p[1], p[2]] for p in tri2_ps]
    add_tri(*tri2_ps, color=COLOR_LIGHT, renderer=renderer)
    add_tri(*tri1_ps, color=COLOR_LIGHT, renderer=renderer)


def get_camera(p_camera, p_focal, renderer):
    if type(renderer) != list:
        renderer = [renderer, ]
    camera = renderer[0].GetActiveCamera()
    camera.ParallelProjectionOn()  # orthographic projection
    camera.SetParallelScale(CAMERA_PARALLEL_SCALE)  # tweak as needed
    camera.SetPosition(p_camera)
    camera.SetViewUp(0.0, 1.0, 0.0)
    camera.SetFocalPoint(p_focal)
    for r in renderer:
        r.SetActiveCamera(camera)
    return camera


def get_renderer_default():
    render_window_interactor = vtkRenderWindowInteractor()
    render_window = vtkRenderWindow()
    render_window.SetMultiSamples(2000)
    render_window.SetNumberOfLayers(3)
    render_window.SetPolygonSmoothing(1)
    render_window.SetLineSmoothing(1)
    render_window.SetAlphaBitPlanes(1)
    render_window_interactor.SetRenderWindow(render_window)
    renderer = vtkRenderer()
    renderer.SetLayer(0)
    renderer.SetUseDepthPeeling(1)
    renderer.SetOcclusionRatio(0.05)
    renderer.SetMaximumNumberOfPeels(1000)
    renderer.UseDepthPeelingForVolumesOn()
    render_window.AddRenderer(renderer)
    renderer.SetBackground(colors.GetColor3d("BkgColor"))
    render_window.SetSize(3000, 2000)  # width, height
    render_window.SetWindowName('CylinderExample')
    render_window.LineSmoothingOn()
    render_window.PolygonSmoothingOn()
    render_window_interactor.Initialize()
    return renderer, render_window, render_window_interactor


def render_image(render_window, fpath_out, ):
    render_window.Render()
    w2if = vtkWindowToImageFilter()
    w2if.SetInput(render_window)
    w2if.SetInputBufferTypeToRGB()
    w2if.ReadFrontBufferOff()
    w2if.Update()
    # ext = os.path.basename(fpath_out).split('.')[1].casefold()
    writer = vtkPNGWriter()
    writer.SetFileName(fpath_out)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()


def add_sensor(z, depth, sd, npix, renderer, color=COLORS['Sensor'], opacity=1.):
    """

    :param z: z pos of front surface
    :param depth:
    :param sd:
    :param npix:
    :param renderer:
    :param color:
    :return:
    """
    sensor = vtkCubeSource()
    sensor.SetCenter(0, 0, z + depth / 2)
    sensor.SetXLength(sd)
    sensor.SetYLength(sd)
    sensor.SetZLength(depth)
    sensor_mapper = vtkPolyDataMapper()
    sensor_mapper.SetInputConnection(sensor.GetOutputPort())
    sensor_actor = vtkActor()
    sensor_actor.SetMapper(sensor_mapper)
    sensor_actor.GetProperty().SetColor(colors.GetColor3d(color))
    sensor_actor.GetProperty().SetRepresentationToSurface()
    sensor_actor.GetProperty().BackfaceCullingOn()
    sensor_actor.GetProperty().LightingOff()
    sensor_actor.GetProperty().SetOpacity(opacity)
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
    renderer.AddActor(sensor_actor)
    renderer.AddActor(edge_actor)

    zl = z - SMOL
    pd = sd / npix
    for ii_line in range(int(-npix / 2) + 1, int(npix / 2)):
        lw = line_width_grid
        add_line(  # HORIZONTAL
            [-sd / 2, ii_line * pd, zl],
            [sd / 2, ii_line * pd, zl],
            line_width=lw, renderer=renderer,
        )
        add_line(  # VERTICAL
            [ii_line * pd, -sd / 2, zl],
            [ii_line * pd, sd / 2, zl],
            line_width=lw, renderer=renderer,
        )


def add_cylinder(z, depth, radius, renderer, renderer_edges, color='White', ):
    """

    :param z: z pos of front surface
    :param depth:
    :param radius:
    :param renderer:
    :param renderer_edges:
    :param color:
    :return:
    """
    cyl = vtkCylinderSource()
    cyl.SetResolution(CYLINDER_RESOLUTION)
    cyl.SetRadius(radius)
    cyl.SetHeight(depth)
    cyl_mapper = vtkPolyDataMapper()
    cyl_mapper.SetInputConnection(cyl.GetOutputPort())
    cyl_actor = vtkActor()
    cyl_actor.SetMapper(cyl_mapper)
    cyl_actor.GetProperty().SetColor(colors.GetColor3d(color))
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
        actor.SetPosition(0.0, 0.0, z + depth / 2)

    transform_actor(cyl_actor)
    cyl.Update()
    # --------------------------------------------------------------------------------------------------------------
    # CYLINDER EDGES
    feature_edges = vtkFeatureEdges()
    feature_edges.ColoringOff()
    feature_edges.SetInputConnection(cyl.GetOutputPort())
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
    cyl_mapper.Update()
    edge_mapper.Update()
    renderer.AddActor(cyl_actor)
    renderer.AddActor(edge_actor)
    # --------------------------------------------------------------------------------------------------------------
    # HACK: LINES TO COMPLETE CYLINDER OUTLINE
    view_angle = 1.14 * np.pi / 4
    nubbin = 0.05
    rad = 1.001 * radius
    add_line(
        [rad * np.cos(view_angle), rad * np.sin(view_angle), z],
        [rad * np.cos(view_angle), rad * np.sin(view_angle), z + depth + 2. * nubbin],
        renderer=renderer_edges, line_width=0.9 * LINEWIDTH_CYLINDER,
    )
    add_line(
        [-rad * np.cos(view_angle), -rad * np.sin(view_angle), z],
        [-rad * np.cos(view_angle), -rad * np.sin(view_angle), z + depth + 2. * nubbin],
        renderer=renderer_edges, line_width=0.9 * LINEWIDTH_CYLINDER,
    )

# ----------------------------------------------------------------------------------------------------------------------
# MISC. OTHER FNS


def imsplice(ims, border=50):
    """
    Splice two or more images together vertically, with the minimum vertical whitespace border specified in pixels.
    Assumes images on a white background.

    :param list ims: list of PIL.Images to splice together vertically. Must all have the same width.
    :param int border: minimum thickness of vertical whitespace between images in pixels.
    :return: spliced image as a PIL.Image instance.
    """
    widths, heights = zip(*(i.size for i in ims))
    if len(widths) > 1:
        assert all(x == widths[0] for x in widths)

    n_ims = len(ims)
    sep_min = np.zeros(n_ims - 1)
    for ii_im in range(n_ims - 1):
        im1 = np.asarray(ims[ii_im])
        im2 = np.asarray(ims[ii_im + 1])
        sep = np.zeros(widths[0])
        for ii_w in range(widths[0]):
            col1 = im1[:, ii_w, :]
            col2 = im2[:, ii_w, :]
            for ii_h1 in range(heights[ii_im]):
                if np.any(col1[-(ii_h1 + 1)] != np.array([255, 255, 255])):
                    break
            for ii_h2 in range(heights[ii_im + 1]):
                if np.any(col2[ii_h2] != np.array([255, 255, 255])):
                    break
            sep[ii_w] = ii_h1 + ii_h2
        sep_min[ii_im] = min(sep)

    height_total = sum(heights)
    for sm in sep_min:
        height_total += border - sm

    im_new = Image.new('RGBA', (widths[0], int(height_total)))
    y_offset = 0
    for ii_im, im in enumerate(ims):
        im = im.convert('RGBA')
        im_blurred = im.filter(ImageFilter.GaussianBlur(max(border // 10, 1)))
        data = im.getdata()
        data_blurred = im_blurred.getdata()
        newData = []
        for item, item_blurred in zip(data, data_blurred):
            if item_blurred[0] == 255 and item_blurred[1] == 255 and item_blurred[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        im.putdata(newData)
        im_new.paste(im, (0, y_offset), im)
        if ii_im < n_ims - 1:
            y_offset += int(im.height + border - sep_min[ii_im])
    background = Image.new('RGBA', im_new.size, (255, 255, 255))
    im_out = Image.alpha_composite(background, im_new).convert('RGB')
    return im_out


def borderfy(im, border=IMG_BORDER, ):
    """
    reduce / expand the image bounding box to the specified number of pixels.
    Assumes image has a white background.

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


def pad_to_width(ims, ):
    """
    Given a list of images, use whitespace to pad them to the same width.
    Used to maintain a consistent absolute scale across a document with multiple figures of different widths.
    :param im:
    :param width:
    :return:
    """
    ws, hs = zip(*(i.size for i in ims))
    w_max = max(ws)
    ims_out = []
    for im in ims:
        if im.width < w_max:
            im_new = Image.new('RGB', (w_max, im.height), (255, 255, 255), )
            im_new.paste(im, (int(w_max / 2 - im.width / 2), 0))
            ims_out.append(im_new)
        else:
            ims_out.append(im)
    return ims_out


def str_round(n, sf):
    """
    convert float to string, rounding to the given number of significant figures.
    from Falken's answer at
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    """
    return '{:g}'.format(float('{:.{p}g}'.format(n, p=sf)))


# ----------------------------------------------------------------------------------------------------------------------
# MAKE FIGURES FOR PAPER

def make_figures_multi_delay_paper():
    """
    make all schematic figures for the multi-delay paper
    :return:
    """

    args_1retarder = {
        'fpath_config': [
            os.path.join(FPATH_CONFIG, '1retarder', 'pycis_config_1retarder_simple.yaml'),
            os.path.join(FPATH_CONFIG, '1retarder', 'pycis_config_1retarder_pixelated.yaml'),
        ],
        'fpath_out': '3panel_1retarder.png',
        'pol_state': POL_STATE_UNPOLARISED,
    }

    args_2retarder_linear = {
        'fpath_config': [
            os.path.join(FPATH_CONFIG, '2retarder_linear', 'pycis_config_2retarder_linear_2delay.yaml'),
            os.path.join(FPATH_CONFIG, '2retarder_linear', 'pycis_config_2retarder_linear_3delay.yaml'),
            os.path.join(FPATH_CONFIG, '2retarder_linear', 'pycis_config_2retarder_linear_4delay.yaml'),
        ],
        'fpath_out': '3panel_2retarder_linear.png',
        'pol_state': POL_STATE_UNPOLARISED,
    }

    args_2retarder_pixelated = {
        'fpath_config': [
            os.path.join(FPATH_CONFIG, '2retarder_pixelated', 'pycis_config_2retarder_pixelated_2delay.yaml'),
            os.path.join(FPATH_CONFIG, '2retarder_pixelated', 'pycis_config_2retarder_pixelated_3delay.yaml'),
        ],
        'fpath_out': '3panel_2retarder_pixelated.png',
        'pol_state': POL_STATE_UNPOLARISED,
    }

    args_3retarder_pixelated = {
        'fpath_config': [
            os.path.join(FPATH_CONFIG, '3retarder_pixelated', 'pycis_config_3retarder_pixelated_6delay.yaml'),
        ],
        'fpath_out': '3panel_3retarder_pixelated.png',
        'pol_state': POL_STATE_UNPOLARISED,
    }

    args_2retarder_specpol = {
        'fpath_config': os.path.join(FPATH_CONFIG, 'specpol', 'pycis_config_2retarder_specpol.yaml'),
        'fpath_out': '3panel_2retarder_specpol.png',
        'pol_state': [
            POL_STATE_UNPOLARISED,
            POL_STATE_LINEAR0,
            POL_STATE_LINEAR45,
            POL_STATE_RHC,
        ]
    }

    args_specpol_asdex = {
        'fpath_config': os.path.join(FPATH_CONFIG, 'specpol', 'pycis_config_specpol_asdex.yaml'),
        'fpath_out': '3panel_specpol_asdex.png',
        'pol_state': [
            POL_STATE_UNPOLARISED,
            POL_STATE_LINEAR0,
            POL_STATE_LINEAR45,
            POL_STATE_RHC,
        ]
    }

    args_specpol_asdex_pixelated = {
        'fpath_config': os.path.join(FPATH_CONFIG, 'specpol', 'pycis_config_specpol_asdex_pixelated.yaml'),
        'fpath_out': '3panel_specpol_asdex_pixelated.png',
        'pol_state': [
            POL_STATE_UNPOLARISED,
            POL_STATE_LINEAR0,
            POL_STATE_LINEAR45,
            POL_STATE_RHC,
        ]
    }

    args_2retarder_specpol_pixelated = {
        'fpath_config': os.path.join(FPATH_CONFIG, 'specpol', 'pycis_config_2retarder_specpol_pixelated.yaml'),
        'fpath_out': '3panel_2retarder_specpol_pixelated.png',
        'pol_state': [
            POL_STATE_UNPOLARISED,
            POL_STATE_LINEAR0,
            POL_STATE_LINEAR45,
            POL_STATE_RHC,
        ]
    }

    args_all = [
        args_1retarder,
        # args_2retarder_linear,
        # args_2retarder_pixelated,
        # args_3retarder_pixelated,
        # args_specpol_asdex,
        # args_specpol_asdex_pixelated,
        # args_2retarder_specpol,
        # args_2retarder_specpol_pixelated,
    ]

    for args in args_all:
        print(args['fpath_out'])
        make_3panel_figure(**args, delete_subfigures=True)

    fpath_out_all = [a['fpath_out'] for a in args_all]
    ims_all = [Image.open(fp) for fp in fpath_out_all]
    ims_all = pad_to_width(ims_all)
    [im.save(fp) for im, fp in zip(ims_all, fpath_out_all)]


if __name__ == '__main__':
    make_figures_multi_delay_paper()
