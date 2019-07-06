from glob import glob
import datetime, os, pickle, shutil, subprocess
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from numpy import array as nparr

from astropy.io import fits
from astropy.io.votable import from_table, writeto, parse
from astropy.coordinates import SkyCoord
from astropy import units as u

#
# wasp-4 paarameters
#
g_wasp4 = 12.3162
rp_wasp4 = 11.7754
bp_wasp4 = 12.7100
plx_wasp4 = 3.7145

xval_wasp4 = bp_wasp4 - rp_wasp4
yval_wasp4 = g_wasp4 + 5*np.log10(plx_wasp4/1e3) + 5

# took parallax SNR > 20, parallax with 0.1 mas of WASP-4, and galactic
# latitude within 10 degrees. ran query on gaia archive.

#
# select top 10000 g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.phot_g_mean_mag,
# g.parallax
# from gaiadr2.gaia_source import as g WHERE g.parallax_over_error > 20 and
# g.parallax > 3.61 and g.parallax < 3.81 and g.b < -58 and g.b > -78
#
queryvot = '../data/wasp4_phot_binary-result.vot.gz'

tab = parse(queryvot)
t = tab.get_first_table().to_table()
df = t.to_pandas()

xval = nparr(df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'])
yval = nparr(df['phot_g_mean_mag'] + 5*np.log10(df['parallax']/1e3) + 5)

sel = ~pd.isnull(xval) & ~pd.isnull(yval)

xval = xval[sel]
yval = yval[sel]

#
# make plot
#
f, ax = plt.subplots(figsize=(4,3))

h, xedges, yedges, cset0 = ax.hist2d(xval, yval, bins=100, cmap='viridis')

cb0 = f.colorbar(cset0, ax=ax, extend='neither', fraction=0.046, pad=0.04,
                 norm=mcolors.PowerNorm(0.3))
cb0.set_label('counts per bin')

ax.plot(xval_wasp4, yval_wasp4, alpha=1, mew=0.2, zorder=8,
        markerfacecolor='yellow', markersize=4, marker='*', color='black',
        lw=0)

ax.set_xlabel('Bp - Rp')
ax.set_ylabel('G + 5log$_{{10}}\omega$ + 5')

ylim = ax.get_ylim()
ax.set_ylim((max(ylim),min(ylim)))

ax.set_xlim((0.3,3))
ax.set_ylim((11.5, 0))

f.savefig('../results/photometric_binarity_check.png', dpi=450,
          bbox_inches='tight')

ax.set_xlim((0.6,1.4))
ax.set_ylim((7, 3))

f.savefig('../results/photometric_binarity_check_zoom.png', dpi=450,
          bbox_inches='tight')

