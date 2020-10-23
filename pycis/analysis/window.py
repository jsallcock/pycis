import numpy as np
import scipy.signal
import xarray as xr


def make_window(fft, fringe_freq):
    """

    :return:
    """

    # make window for isolating carrier frequency
    fringe_freq_abs = np.sqrt(fringe_freq[0] ** 2 + fringe_freq[1] ** 2)
    fringe_freq_angle = np.arctan2(fringe_freq[1], fringe_freq[0])

    coord_proj = np.cos(fringe_freq_angle) * fft.freq_x + np.sin(fringe_freq_angle) * fft.freq_y
    condition_1 = coord_proj < fringe_freq_abs - fringe_freq_abs / 1.5
    condition_2 = coord_proj > fringe_freq_abs + fringe_freq_abs / 1.5
    window = xr.ufuncs.logical_not(condition_1 + condition_2).astype(float)

    window = xr.where(window == 0, np.nan, window)
    coord_proj = window * coord_proj
    coord_proj = (coord_proj - np.nanmin(coord_proj)) / (np.nanmax(coord_proj) - np.nanmin(coord_proj))

    # Blackman window
    alpha = 0.16
    a_0 = (1 - alpha) / 2
    a_1 = 0.5
    a_2 = alpha / 2
    window = a_0 - a_1 * np.cos(2 * np.pi * coord_proj) + a_2 * np.cos(4 * np.pi * coord_proj)
    window = xr.where(xr.ufuncs.isnan(window), 0, window, )
    window = xr.where(window < 0, 0, window, )

    window.values = window.values + np.flip(window.values)  # second window for negative frequency carrier
    window /= float(window.max())  # normalise

    return window


def window_old(rfft_length, nfringes, window_width=None, fn='tukey', width_factor=1., alpha=0.5):
    """
    LEGACY -- to be deleted

    Generate a filters window for isolating the carrier (fringe) frequency.

    :param rfft_length: length of the real FFT to be filtered
    :type rfft_length: int.
    :param nfringes: The carrier (fringe) frequency to be demodulated, in units of cycles per sequence -- approximately the number of fringes present in the image.
    :type nfringes: int.
    :param fn: Specifies the window function to use, see the 'fns' dict at the top of the script for the
    :type fn: str.
    :param width_factor: a multiplicative factor determining the width of the filters, multiplies nfringes, which is found to work reasonably well.
    :type width_factor: float

    :return: generated window as an array.
    """

    fns = {'hanning': scipy.signal.hanning,
           'blackmanharris': scipy.signal.blackmanharris,
           'tukey': scipy.signal.windows.tukey}

    assert fn in fns

    if window_width == None:
        window_width = int(nfringes * width_factor)

    pre_zeros = [0] * int(nfringes - window_width / 2)

    if fn == 'tukey':
        fn = fns[fn]

        window_fn = fn(window_width, alpha=alpha)
    else:
        fn = fns[fn]
        window_fn = fn(window_width)

    post_zeros = [0] * (rfft_length - window_width - int(nfringes - window_width / 2 - 1))

    return np.concatenate((pre_zeros, window_fn, post_zeros))[:rfft_length]


if __name__ == '__main__':

    alphas = [0, 0.25, 0.5, 0.75, 1.0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for a in alphas:
        win = window(1000, 100, fn='tukey', width_factor=1, alpha=a)
        print(win)
        ax.plot(win)

    plt.show(block=True)


