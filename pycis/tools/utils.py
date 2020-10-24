import numpy as np


def get_fwhm(x, y, ):
    """ given a function with a SINGLE PEAK, find the FWHM without fitting. QUICK AND DIRTY. """

    # normalise height
    y_norm = y / np.max(y)

    # split array into two about the max value
    idx_max = np.argmax(y_norm)
    x_l, x_u, y_l, y_u = x[:idx_max], x[idx_max + 1:], y_norm[:idx_max], y_norm[idx_max + 1:]

    # 1D interpolation
    hm_idx_l, hm_idx_u = np.interp(0.5, y_l, x_l), np.interp(0.5, y_u[::-1], x_u[::-1])

    fwhm = hm_idx_u - hm_idx_l

    return fwhm


def get_roi(input_image, centre=None, roi_dim=(250, 250), ):
    """ Given an input image, returns the centred region of interest (ROI) with user-specified dimensions. """

    y_dim, x_dim = np.shape(input_image)

    if centre is None:
        y_dim_h = y_dim / 2
        x_dim_h = x_dim / 2
    else:
        y_dim_h = centre[0]
        x_dim_h = centre[1]

    roi_width_h = roi_dim[1] / 2
    roi_height_h = roi_dim[0] / 2

    y_lo = int(y_dim_h - roi_height_h)
    y_hi = int(y_dim_h + roi_height_h)

    x_lo = int(x_dim_h - roi_width_h)
    x_hi = int(x_dim_h + roi_width_h)

    return input_image[y_lo:y_hi, x_lo:x_hi]
