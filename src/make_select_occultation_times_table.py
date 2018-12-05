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
        '{:s}_occultation_times_selected.csv'.
        format(plname)
    )
    df = get_data(datacsv=allseldatapath)

    midpoints = np.array(df['sel_occ_times_BJD_TDB'])
    uncertainty = np.array(df['err_sel_occ_times_BJD_TDB'])
    epochs = np.array(df['sel_epoch']).astype(int)

    original_references = np.array(df['original_reference'])
    references = []
    for ref in original_references:
        if ref == '2011A&A...530A...5C':
            references.append('\citet{caceres_ground-based_2011}')
        elif ref == '2011ApJ...727...23B':
            references.append('\citet{beerer_secondary_2011}')
        elif ref == '2015MNRAS.454.3002Z':
            references.append('\citet{zhou_secondary_2015}')

    references = np.array(references)

    outdf = pd.DataFrame(
        {'midpoints': np.round(midpoints,5),
         'uncertainty': np.round(uncertainty,5),
         'epochs': epochs,
         'original_reference': references
        }
    )
    outdf['midpoints'] = outdf['midpoints'].map('{:.5f}'.format)
    outdf = outdf[
        ['midpoints', 'uncertainty', 'epochs', 'original_reference']
    ]

    outpath = 'selected_occultation_times.tex'
    with open(outpath,'w') as tf:
        print('wrote {:s}'.format(outpath))
        tf.write(outdf.to_latex(index=False, escape=False))

if __name__=="__main__":
    main()
