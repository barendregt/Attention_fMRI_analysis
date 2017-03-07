import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def spatial_smooth(in_file, kernel, voxel_size=3):
    """ Smoothes a functional or structural MRI file in the spatial dimension.
    Kernel should be given in desired FWHM in millimeters.
    Assumes isotropic smooth (same kernel in each dimension).

    Neuroimaging, University of Amsterdam, 2016 """

    # Calculate FWHM
    sigma = (kernel / float(voxel_size)) / (2 * np.sqrt(2 * np.log(2)))
    if in_file.ndim > 3:
       smoothed = np.stack([gaussian_filter(in_file[:, :, :, i], sigma)
                            for i in range(in_file.shape[-1])], axis=-1)
    else:
       smoothed = gaussian_filter(in_file, sigma)

    return smoothed

def high_pass_filter(signal, kernel=20):
    """ Simple high pass filter in time-domain. """

    if signal.ndim > 1:
        # Hier nog even mask implementeren voor nonzero voxels
        filt = np.stack([gaussian_filter(signal[:, :, :, i], kernel)
                         for i in range(signal.shape[-1])], axis=-1)
        filtsig = signal - filt
    else:
        filt = gaussian_filter(signal, kernel)
        filtsig = signal - filt
    return filtsig, filt

def double_gamma(x, lag=6, a2=12, b1=0.9, b2=0.9, c=0.35):
    a1 = lag
    d1 = a1 * b1
    d2 = a2 * b2
    return np.array([(t/(d1))**a1 * np.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * np.exp(-(t-d2)/b2) for t in x])

def plot_frequency_domain(data, sampling_rate=2, plot_power=False, plot_log=False,
                          xlim=None, ylim=None, title=None):

    timeres = 1.0/sampling_rate

    ps = np.abs(np.fft.fft(data)/data.size)*2

    if plot_power:
        ps = ps ** 2
        ylabel = 'Power'
    else:
        ylabel = 'Amplitude'

    freqs = np.fft.fftfreq(data.size, timeres)
    idx = np.argsort(freqs)[freqs.size/2:]

    if plot_log:
        plt.loglog(freqs[idx], ps[idx])
        ylabel = 'log(%s)' % xlabel
        xlabel = 'log(frequency)'
    else:
        plt.plot(freqs[idx], ps[idx])
        xlabel = 'frequency (Hz)'

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)
    else:
        plt.title('Frequency domain')

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)


def create_sine_wave(timepoints, frequency=1,amplitude=1, phase=0):
    return amplitude * np.sin(2*np.pi*frequency*timepoints + phase)

