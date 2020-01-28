import os, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
from cdips.plotting import savefig
import matplotlib as mpl
from astropy import units as u, constants as c
import matplotlib.patheffects as pe
import matplotlib.image as mpimg

def plot_zorro_speckle():

    datapath = (
        '../data/WASP4_zorro_speckle/WASP-4_20190928_832_companionbounds.csv'
    )
    df = pd.read_csv(datapath)

    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(
        df.ang_sep, df.delta_mag, color='k'
    )


    img = mpimg.imread(
        '../data/WASP4_zorro_speckle/WASP-4_20190928_832_img.png'
    )

    # [left, bottom, width, height]
    inset = fig.add_axes([0.55, 0.23, .4, .4])
    inset.imshow(img)
    inset.axis('off')
    plt.setp(inset, xticks=[], yticks=[])

    ax.set_ylabel('$\Delta$m (832 nm)')
    ax.set_xlabel('Angular separation [arcsec]')

    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.tick_params(right=True, which='both', direction='in')
    ax.tick_params(top=True, which='both', direction='in')
    fig.tight_layout(h_pad=0, w_pad=0)

    figpath = (
        '../results/zorro_speckle.png'
    )
    savefig(fig, figpath)


if __name__ == "__main__":

    plot_zorro_speckle()
