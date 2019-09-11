"""
do we need to worry about stellar activity?
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv('../data/20190911_jump_wasp4_rv.csv', comment='#')

f,axs = plt.subplots(nrows=2,ncols=1,figsize=(4,6), sharex=True)

axs[0].scatter(df['bjd'], df['mnvel'])
axs[0].set_ylabel('mnvel')

axs[1].scatter(df['bjd'], df['svalue'])
axs[1].set_ylabel('svalue')
axs[1].set_xlabel('bjd tdb')

f.tight_layout()
outpath = '../results/20190911_rv_vs_CaHK_check_timeseries.png'
f.savefig(outpath, dpi=300, bbox_inches='tight')
print('made {}'.format(outpath))

##########################################

plt.close('all')

f,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,3))

ax.scatter(df['mnvel'], df['svalue'])
ax.set_ylabel('svalue')
ax.set_xlabel('mnvel')

f.tight_layout()
outpath = '../results/20190911_rv_vs_CaHK_check_mnvel_vs_svalue.png'
f.savefig(outpath, dpi=300, bbox_inches='tight')
print('made {}'.format(outpath))


