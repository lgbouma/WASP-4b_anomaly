"""
Plot significant period derivatives expected from Knutson+14.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy import array as nparr
import matplotlib.patheffects as pe

#
# manually added WASP-4 to the output
# \dot{P}_{\rm RV} &= -5.94 \pm 0.39~{\rm ms}\,{\rm yr}^{-1}.
#

df = pd.read_csv('../results/knutson_all_pdots.csv', sep=';')
sel = (df.K14_significant) & (df.Pdot != 0)
df = df[sel]

savpath = '../results/k14_pdot.png'
fig, ax = plt.subplots(figsize=(4,3))

yval = np.arange(0, len(df), 1) + 0.1
xval = nparr(df.Pdot)
x_perr = nparr(df.Pdot_perr)
x_merr = nparr(df.Pdot_merr)
dotlabels = nparr(df.planet)

ax.errorbar(xval, yval, xerr=np.vstack([x_merr, x_perr]),
            fmt='.k', ecolor='black', zorder=2, alpha=1, mew=1,
            markersize=2.5, capsize=3)

ix = 0
for _x, _y, _l in zip(xval+x_perr+2, yval, dotlabels):

    if 'WASP-4' in _l:
        c = 'C0'
        ax.errorbar(xval[ix], yval[ix],
                    xerr=np.vstack([x_merr[ix], x_perr[ix]]),
                    fmt='.', color=c, ecolor=c, zorder=3, alpha=1,
                    mew=1, markersize=2.5, capsize=3)

    else:
        c = 'k'

    ax.text(
        _x, _y, _l.replace(' ','$\,$'),
        fontsize=7.5, ha='left', va='center',
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        color=c
    )

    ix += 1

# axes etc
xlabel = 'Expected d$P$/d$t$ from RV [millisecond/year]'
ax.set_xlabel(xlabel)

xlim = ax.get_xlim()
xmax = max(np.abs(xlim))
ax.set_xlim([-xmax, xmax])

ax.set_ylim([-1, len(df)])
ylim = ax.get_ylim()
ax.set_ylim((min(ylim), max(ylim)))
ax.vlines(0, min(ylim), max(ylim), color='k', linestyle='--',
          zorder=-2, lw=0.5, alpha=1)
ax.set_ylim((min(ylim), max(ylim)))

ax.get_yaxis().set_tick_params(which='both', direction='in')
ax.get_xaxis().set_tick_params(which='both', direction='in')

ax.set_yticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.tight_layout(h_pad=0, w_pad=0)
fig.savefig(savpath, bbox_inches='tight', dpi=400)
print('saved {:s}'.format(savpath))
savpath = savpath.replace('.png','.pdf')
fig.savefig(savpath, bbox_inches='tight')
print('saved {:s}'.format(savpath))

