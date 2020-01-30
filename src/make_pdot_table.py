'''
make table of Pdots
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from numpy import array as nparr

from glob import glob
import os, pickle

def main():

    df = pd.read_csv(
        '../results/knutson_all_pdots.csv', sep=';'
    )

    # \colhead{Planet} &
    # \colhead{$\dot{\gamma}$ [m$\,$s$^{-1}$$\,$yr$^{-1}$]} &
    # \colhead{$+\sigma_{\dot{\gamma}}$ [m$\,$s$^{-1}$yr$^{-1}$]} & 
    # \colhead{$-\sigma_{\dot{\gamma}}$ [m$\,$s$^{-1}$yr$^{-1}$]} & 
    # \colhead{$P$ [days]} &
    # \colhead{$\dot{P}_{\,{\rm RV}}$ [ms$\,$yr$^{-1}$]} &
    # \colhead{$+\sigma_{\dot{P}_{\,{\rm RV}}}$ [ms$\,$yr$^{-1}$]} &
    # \colhead{$-\sigma_{\dot{P}_{\,{\rm RV}}}$ [ms$\,$yr$^{-1}$]} &
    # \colhead{Significant?}

    # 'planet', 'gammadot', 'gammadot_pluserr', 'gammadot_minuserr', 'comment',
    # 'pl_name', 'pl_orbper', 'Pdot', 'Pdot_upper_limit', 'Pdot_lower_limit',
    # 'Pdot_perr', 'Pdot_merr', 'abs_Pdot', 'K14_significant'

    scols = [
        'planet', 'gammadot', 'gammadot_pluserr', 'gammadot_minuserr',
        'pl_orbper', 'Pdot', 'Pdot_perr', 'Pdot_merr', 'K14_significant',
        'comment'
    ]

    outdf = df[scols]

    threeptcols = ['Pdot', 'Pdot_perr', 'Pdot_merr']
    for c in threeptcols:
        outdf[c] = np.round(nparr(outdf[c]),3)

    sixptcols = ['gammadot', 'gammadot_pluserr', 'gammadot_minuserr']
    for c in sixptcols:
        outdf[c] = np.round(nparr(outdf[c]),6)

    outdf['pl_orbper'] = np.round(nparr(outdf['pl_orbper']),6)

    outdf['K14_significant'] = outdf['K14_significant'].astype(int)

    outpath = '../paper/pdot_table_data.csv'
    outdf.to_csv(outpath, index=False)
    print('wrote {}'.format(outpath))


if __name__=="__main__":
    main()
