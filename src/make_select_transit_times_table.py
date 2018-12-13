# -*- coding: utf-8 -*-
'''
make table of selected transit times
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
import os, pickle


def get_data(
    datacsv='../data/WASP-18b_literature_and_TESS_times_O-C_vs_epoch_selected.csv'
    ):
    # need to run make_parameter_vs_epoch_plots.py first; this generates the
    # SELECTED epochs (x values), mid-times (y values), and mid-time errors
    # (sigma_y).

    df = pd.read_csv(datacsv, sep=';')

    return df


def main():

    plname = 'WASP-4b'
    allseldatapath = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/data/'+
        '{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'.
        format(plname)
    )
    df = get_data(datacsv=allseldatapath)

    midpoints = np.array(df['sel_transit_times_BJD_TDB'])
    uncertainty = np.array(df['err_sel_transit_times_BJD_TDB'])
    epochs = np.array(df['sel_epoch']).astype(int)
    is_H13 = (
        (np.array(df['where_I_got_time']) == '2013MNRAS.434...46H').astype(int)
    )

    original_references = np.array(df['original_reference'])
    references = []
    for ref in original_references:
        if ref == '2008ApJ...675L.113W':
            references.append('\citet{wilson_wasp-4b_2008}')
        elif ref == '2009A&A...496..259G':
            references.append('\citet{gillon_improved_2009}')
        elif ref == '2009AJ....137.3826W':
            references.append('\citet{winn_transit_2009}')
        elif ref == '2011AJ....142..115D':
            references.append('\citet{dragomir_terms_2011}')
        elif ref == '2011ApJ...733..127S':
            references.append('\citet{sanchis-ojeda_starspots_2011}')
        elif ref == '2012A&A...539A.159N':
            references.append('\citet{nikolov_wasp-4b_2012}')
        elif ref == '2013MNRAS.434...46H':
            references.append('\citet{hoyer_tramos_2013}')
        elif ref == '2014ApJ...785..148R':
            references.append('\citet{ranjan_atmospheric_2014}')
        elif ref == '2017AJ....154...95H':
            references.append('\citet{huitson_gemini_2017}')
        elif ref == 'me':
            references.append('This work')
        elif ref == 'Baxter et al. (in prep)':
            references.append('Baxter et al.\ (in prep)')
    references = np.array(references)

    outdf = pd.DataFrame(
        {'midpoints': np.round(midpoints,5),
         'uncertainty': np.round(uncertainty,5),
         'epochs': epochs,
         'is_H13': is_H13,
         'original_reference': references
        }
    )
    outdf['midpoints'] = outdf['midpoints'].map('{:.5f}'.format)
    outdf = outdf[
        ['midpoints', 'uncertainty', 'epochs', 'is_H13', 'original_reference']
    ]

    outpath = 'selected_transit_times.tex'
    with open(outpath,'w') as tf:
        print('wrote {:s}'.format(outpath))
        tf.write(outdf.to_latex(index=False, escape=False))

if __name__=="__main__":
    main()
