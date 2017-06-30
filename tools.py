import numpy as np
import matplotlib.pyplot as plt


def two_gamma(x, peak1 = 5.4, fwhm1 = 5.2, peak2 = 10.8, fwhm2 = 7.35, dip = 0.35):
    a1 = peak1**2 / fwhm1**2*8*np.log(2)
    b1 = fwhm1**2 / peak1 /8/np.log(2)
    g1 = np.array([(t/peak1)**a1 * np.exp(-(t-peak1)/b1) for t in x])

    a2 = peak2**2 / fwhm2**2*8*np.log(2)
    b2 = fwhm2**2 / peak2 /8/np.log(2)
    g2 = np.array([(t/peak2)**a2 * np.exp(-(t-peak2)/b2) for t in x])
    return np.array(g1-dip*g2)


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax