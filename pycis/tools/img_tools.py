import copy
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.ndimage
# from PIL import Image
import pycis


default_roi_dim = (50, 50)

def get_img_flist(path, fmt='tif'):
    assert os.path.isdir(path)
    return sorted(glob.glob(os.path.join(path, '*.' + fmt)))



def get_img_stack(path, rot90=0, fmt='tif', display=False, overwrite=False, delete=False):
    """
    stack all images in a directory
    
    :param path: 
    :param rot90: number of 90 deg rotations (direction is from first towards the second axis, as per np.rot90
    convention)
    :param fmt: 
    :param overwrite: bool, if False, look for an existing image stack from a previous calculation.
    :param delete: delete original image files after stacking -- use with caution.
    :return: 
    """

    img_stack_path = os.path.join(path, 'img_stack.npy')
    flist = get_img_flist(path, fmt=fmt)
    fnum = len(flist)

    if os.path.isfile(img_stack_path) and overwrite is False:
        img_stack = np.load(img_stack_path)

    else:

        if fnum == 0:
            raise Exception('No files of this type found in the given directory.')

        img_stack = 0

        for f in flist:
            img_stack += get_img(f)

        if rot90 != 0:
            img_stack = np.rot90(img_stack, k=rot90)

        np.save(img_stack_path, img_stack)

    if delete:
        print('You are about to delete the contents of: ')
        print(str(path))
        print('-----')
        choice = input('Proceed? [y/n]: ')
        if choice == 'y':
            for f in flist:
                os.remove(f)
            print('deletion completed')
        else:
            pass

    if display:

        plt.figure()
        plt.imshow(img_stack, 'gray')
        plt.colorbar()
        plt.show(block=True)

    return img_stack


def get_phase_img_stack(path, rot90=0, fmt='tif', overwrite=False, **kwargs):
    """
    Calculate and save the phase of the stacked images of the specified format in the given directory.
    """

    assert os.path.isdir(path)

    phase_img_stack_path = os.path.join(path, 'phase_img_stack.npy')

    if os.path.isfile(phase_img_stack_path) and overwrite is False:
        phase_img_stack = np.load(phase_img_stack_path)

    else:

        img_stack = get_img_stack(path, rot90=rot90, fmt=fmt, overwrite=overwrite)

        intensity, phase, contrast = pycis.fourier_demod_2d(img_stack, mask=True, despeckle=True)
        phase_img_stack = phase
        np.save(phase_img_stack_path, phase_img_stack)

    return phase_img_stack


def get_contrast_img_stack(path, rot90=0, fmt='tif', overwrite=False):

    assert os.path.isdir(path)

    contrast_img_stack_path = os.path.join(path, 'contrast_img_stack.npy')

    if os.path.isfile(contrast_img_stack_path) and overwrite is False:
        contrast_img_stack = np.load(contrast_img_stack_path)

    else:

        img_stack = get_img_stack(path, rot90=rot90, fmt=fmt, overwrite=overwrite)

        intensity, phase, contrast = demod_function(img_stack, display=False)

        contrast_img_stack = contrast

        np.save(contrast_img_stack_path, contrast_img_stack)

    return contrast_img_stack


def get_dc_img_stack(path, rot90=0, fmt='tif', overwrite=False):

    assert os.path.isdir(path)

    dc_img_stack_path = os.path.join(path, 'dc_img_stack.npy')

    if os.path.isfile(dc_img_stack_path) and overwrite is False:
        dc_img_stack = np.load(dc_img_stack_path)

    else:

        img_stack = get_img_stack(path, rot90=rot90, fmt=fmt, overwrite=overwrite)

        intensity, phase, contrast = demod_function(img_stack, display=False)

        dc_img_stack = intensity

        np.save(dc_img_stack_path, dc_img_stack)

    return dc_img_stack


def get_contrast_roi_mean(path, roi_dim=default_roi_dim, overwrite=False):
    """ Calculate the mean roi contrast and estimate the std. roi contrast

        :param path: 
        :param fmt: 
        :param roi_dim:
        :param overwrite: (bool.)
        :return: phase_roi_mean, phase_roi_std (both in radians)
        """

    assert os.path.isdir(path)

    contrast_roi_stack_mean_path = os.path.join(path, 'contrast_roi_stack_mean.npy')

    # contrast ROI mean
    if os.path.isfile(contrast_roi_stack_mean_path) and overwrite is False:
        contrast_roi_stack_mean = np.load(contrast_roi_stack_mean_path)

    else:
        contrast_img_stack = get_contrast_img_stack(path, overwrite=overwrite)

        contrast_roi_stack_mean = np.mean(pycis.tools.get_roi(contrast_img_stack, roi_dim=roi_dim))

        # save
        np.save(contrast_roi_stack_mean_path, contrast_roi_stack_mean)

    return contrast_roi_stack_mean


def get_phase_roi_mean(path, rot90=0, roi_dim=default_roi_dim, overwrite=False):
    """ Calculate the mean roi phase and estimate the std. roi phase

    :param path: 
    :param fmt: 
    :param roi_dim:
    :param overwrite: (bool.)
    :return: phase_roi_mean, phase_roi_std (both in radians)
    """

    assert os.path.isdir(path)

    phase_roi_stack_mean_path = os.path.join(path, 'phase_roi_stack_mean.npy')

    # phase ROI mean (rad)
    if os.path.isfile(phase_roi_stack_mean_path) and overwrite is False:
        phase_roi_stack_mean = np.load(phase_roi_stack_mean_path)

    else:
        phase_img_stack = get_phase_img_stack(path, rot90=rot90, overwrite=overwrite)

        phase_img_stack = pycis.analysis.unwrap(phase_img_stack)

        phase_roi_stack_mean = pycis.analysis.wrap(np.mean(pycis.tools.get_roi(phase_img_stack, roi_dim=roi_dim)), units='rad')

        # save
        np.save(phase_roi_stack_mean_path, phase_roi_stack_mean)

    return phase_roi_stack_mean

#
# def get_phase_pixel_stack_std(path, rot90=0, fmt='tif', img_lim=30, overwrite=False):
#     """ Calculate standard deviation in phase for a single pixel in a series of images (assumes all images in the
#     given directory are the same dimensions).
#
#
#         :param path:
#         :param fmt:
#         :param roi_dim:
#         :param overwrite: (bool.)
#         :return: phase_roi_mean, phase_roi_std (both in radians)
#         """
#
#     assert os.path.isdir(path)
#
#     phase_pixel_stack_std_path = os.path.join(path, 'phase_pixel_stack_std.npy')
#
#
#     if os.path.isfile(phase_pixel_stack_std_path) and overwrite is False:
#         phase_pixel_stack_std = np.load(phase_pixel_stack_std_path)
#
#     else:
#         flist = get_img_flist(path, fmt=fmt)
#         fnum = len(flist)
#
#         # use all the images available, up to a set limit (img_lim)
#         if fnum > img_lim:
#             fnum_trunc = img_lim
#         else:
#             fnum_trunc = fnum
#
#         # get img dimensions and indices for a central pixel
#         img_h, img_w = np.array(Image.open(flist[0]), dtype=np.float64).shape
#         pix_y = round(img_h / 2)
#         pix_x = round(img_w / 2)
#
#         phase_pixel = []
#
#         for imgpath in flist[:fnum_trunc]:
#             print(imgpath)
#             img = np.array(Image.open(imgpath), dtype=np.float64)
#
#             if rot90 != 0:
#                 img = np.rot90(img, k=rot90)
#
#             intensity, phase, contrast = pycis.analysis.fourier_demod_2d(img)
#
#             phase_pixel.append(phase[pix_y, pix_x])
#
#         phase_pixel_std = np.std(np.array(phase_pixel))
#
#         # scale to obtain phase std for the stacked pixel.
#         phase_pixel_stack_std = phase_pixel_std * fnum ** - 0.5
#
#         # save
#         np.save(phase_pixel_stack_std_path, phase_pixel_stack_std)
#
#     return phase_pixel_stack_std

def get_contrast_pixel_stack_std(path, fmt='tif', img_lim=30, overwrite=False):
    """ Calculate standard deviation in contrast for a single pixel in a series of images (assumes all images in the 
    given directory are the same dimensions).


        :param path: 
        :param fmt: 
        :param roi_dim:
        :param overwrite: (bool.)
        :return: phase_roi_mean, phase_roi_std (both in radians)
        """

    assert os.path.isdir(path)

    contrast_pixel_stack_std_path = os.path.join(path, 'contrast_pixel_stack_std.npy')

    if os.path.isfile(contrast_pixel_stack_std_path) and overwrite is False:
        contrast_pixel_stack_std = np.load(contrast_pixel_stack_std_path)

    else:
        flist = get_img_flist(path, fmt=fmt)
        fnum = len(flist)

        if fnum > img_lim:
            fnum_trunc = img_lim
        else:
            fnum_trunc = fnum

        # get img dimensions
        img_h, img_w = np.array(Image.open(flist[0]), dtype=np.float64).shape
        pix_y = round(img_h / 2)
        pix_x = round(img_w / 2)

        contrast_pixel = []

        for imgpath in flist[:fnum_trunc]:
            print(imgpath)
            img = np.array(Image.open(imgpath), dtype=np.float64)

            intensity, phase, contrast = pycis.analysis.fourier_demod_2d(img)

            contrast_pixel.append(contrast[pix_y, pix_x])

        contrast_pixel_std = np.std(np.array(contrast_pixel))

        # scale to obtain phase STD for stacked pixel.
        contrast_pixel_stack_std = contrast_pixel_std * fnum ** - 0.5

        # save
        np.save(contrast_pixel_stack_std_path, contrast_pixel_stack_std)

    return contrast_pixel_stack_std


def get_phase_roi_std_err(path, rot90=0, fmt='tif', roi_dim=default_roi_dim, img_lim=25, overwrite=False):
    """ Calculate the mean roi phase and estimate the std. roi phase

    :param path: 
    :param fmt: 
    :param roi_dim:
    :param overwrite: (bool.)
    :return: phase_roi_mean, phase_roi_std (both in radians)
    """

    assert os.path.isdir(path)

    phase_roi_stack_std_path = os.path.join(path, 'phase_roi_stack_std.npy')
    phase_roi_mean_path = os.path.join(path, 'phase_roi_mean.npy')


    if os.path.isfile(phase_roi_stack_std_path) and overwrite is False:
        phase_roi_stack_std_err = np.load(phase_roi_stack_std_path)
        phase_roi_mean = np.load(phase_roi_mean_path)

    else:
        flist = get_img_flist(path, fmt=fmt)
        fnum = len(flist)

        if fnum > img_lim:
            fnum_trunc = img_lim
        else:
            fnum_trunc = fnum

        phase_roi_mean = []

        for imgpath in flist[:fnum_trunc]:
            print(imgpath)
            img = np.array(Image.open(imgpath), dtype=np.float64)

            if rot90 != 0:
                img = np.rot90(img, k=rot90)

            intensity, phase, contrast = pycis.analysis.fourier_demod_2d(img)
            phase = pycis.analysis.unwrap(phase)

            phase_roi_mean.append(pycis.analysis.wrap(np.mean(pycis.tools.get_roi(phase, roi_dim=roi_dim))))

        phase_roi_mean = np.array(phase_roi_mean)

        # calculate standard error
        phase_roi_stack_std_err = np.std(phase_roi_mean) / fnum ** 0.5

        # save
        np.save(phase_roi_stack_std_path, phase_roi_stack_std_err)
        np.save(phase_roi_mean_path, phase_roi_mean)

    return phase_roi_stack_std_err, phase_roi_mean


def get_contrast_roi_std(path, fmt='tif', roi_dim=default_roi_dim, img_lim=25, overwrite=False):
    """ Calculate the mean roi contrast and estimate the std. roi contrast

    :param path: 
    :param fmt: 
    :param roi_dim:
    :param overwrite: (bool.)
    :return: phase_roi_mean, phase_roi_std (both in radians)
    """

    assert os.path.isdir(path)

    contrast_roi_stack_std_path = os.path.join(path, 'contrast_roi_stack_std.npy')
    contrast_roi_mean_path = os.path.join(path, 'contrast_roi_mean.npy')


    if os.path.isfile(contrast_roi_stack_std_path) and overwrite is False:
        contrast_roi_stack_std = np.load(contrast_roi_stack_std_path)
        contrast_roi_mean = np.load(contrast_roi_mean_path)

    else:
        flist = get_img_flist(path, fmt=fmt)
        fnum = len(flist)

        if fnum > img_lim:
            fnum_trunc = img_lim
        else:
            fnum_trunc = fnum

        contrast_roi_mean = []

        for imgpath in flist[:fnum_trunc]:
            print(imgpath)
            img = np.array(Image.open(imgpath), dtype=np.float64)

            intensity, phase, contrast = pycis.analysis.fourier_demod_2d(img)
            # phase = pycis.uncertainty.unwrap(phase)

            contrast_roi_mean.append(pycis.analysis.wrap(np.mean(pycis.tools.get_roi(phase, roi_dim=roi_dim))))

        contrast_roi_mean = np.array(contrast_roi_mean)

        contrast_roi_stack_std = np.std(contrast_roi_mean) / fnum ** (0.25)  # TODO: double check this scaling.

        # save
        np.save(contrast_roi_stack_std_path, contrast_roi_stack_std)
        np.save(contrast_roi_mean_path, contrast_roi_mean)


    return contrast_roi_stack_std, contrast_roi_mean


def offset_shape(phase, roi_dim=default_roi_dim):
    """ phase must be wrapped and in radians (ie. raw output of fd_image).
    
    :param phase: 
    :param roi_dim: 
    :return: 
    """

    phase = pycis.analysis.unwrap(phase)

    offset = np.mean(pycis.tools.get_roi(phase, roi_dim=roi_dim))
    shape = phase - offset

    return offset, shape


