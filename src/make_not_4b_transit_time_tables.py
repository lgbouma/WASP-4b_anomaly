# -*- coding: utf-8 -*-
'''
make table of selected transit times for WASP-5b, WASP-6b, WASP-18b,
and WASP-46b.
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from glob import glob
import os, pickle

from shutil import copyfile
from numpy import array as nparr

def get_data(
    datacsv='../data/WASP-18b_literature_and_TESS_times_O-C_vs_epoch_selected.csv'
    ):
    # need to run make_parameter_vs_epoch_plots.py first; this generates the
    # SELECTED epochs (x values), mid-times (y values), and mid-time errors
    # (sigma_y).

    df = pd.read_csv(datacsv, sep=';')

    return df


def make_temp_table(plname):

    allseldatapath = (
        '/home/luke/Dropbox/proj/tessorbitaldecay/data/'+
        '{:s}_literature_and_TESS_times_O-C_vs_epoch_selected.csv'.
        format(plname)
    )
    df = get_data(datacsv=allseldatapath)

    midpoints = np.array(df['sel_transit_times_BJD_TDB'])
    uncertainty = np.array(df['err_sel_transit_times_BJD_TDB'])
    epochs = np.array(df['sel_epoch']).astype(int)
    original_references = np.array(df['original_reference'])
    references = []
    for ref in original_references:
        #WASP-4b
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
        #WASP-5b
        elif ref == '2008MNRAS.387L...4A':
            references.append('\citet{anderson_wasp-5b_2008}')
        elif ref == '2011PASJ...63..287F':
            references.append('\citet{fukui_measurements_2011}')
        elif ref == '2012ApJ...748...22H':
            references.append('\citet{hoyer_transit_2012}')
        elif ref == '2009MNRAS.396.1023S':
            references.append('\citet{southworth_high-precision_2009}')
        elif ref == '2017MNRAS.471..650M':
            references.append('\citet{moyano_multi-band_2017}')
        #WASP-6b
        elif ref == '2009A&A...501..785G':
            references.append('\citet{gillon_discovery_2009}')
        elif ref == '2015MNRAS.450.1760':
            references.append('\citet{tregloan-reed_transits_2015}')
        elif ref == '2013ApJ...778..184J':
            references.append('\citet{jordan_ground-based_2013}')
        elif ref == '2012PASP..124..212S':
            references.append('\citet{sada_extrasolar_2012}')
        elif ref == '2015MNRAS.447..463N':
            references.append('\citet{nikolov_hst_2015}')
        #WASP-18b
        elif ref == '2009Natur.460.1098H':
            references.append('\citet{hellier_orbital_2009}')
        elif ref == '2013MNRAS.428.2645M':
            references.append('\citet{maxted_spitzer_2013}')
        elif ref == '2017ApJ...836L..24W':
            references.append('\citet{wilkins_searching_2017}')
        #WASP-46b
        elif ref == '2012MNRAS.422.1988A':
            references.append('\citet{anderson_wasp-44b_2012}')
        elif ref == '2016MNRAS.456..990C':
            references.append('\citet{ciceri_physical_2016}')
        elif ref == 'ETD,TRESCA':
            references.append('\citet{petrucci_search_2018}')
        elif ref == '2018MNRAS.473.5126P':
            references.append('\citet{petrucci_search_2018}')


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

    outpath = 'temp_{}.tex'.format(plname)
    with open(outpath,'w') as tf:
        print('wrote {:s}'.format(outpath))
        tf.write(outdf.to_latex(index=False, escape=False))

    return outpath

if __name__=="__main__":

    plnames = ['WASP-5b', 'WASP-6b', 'WASP-18b', 'WASP-46b']
    #plnames = ['WASP-18b'] # NOTE for avi's wasp-18 paper, use this

    for plname in plnames:
        temppath = make_temp_table(plname)

        templatetxt = 'template.txt'
        outtable = ( '../paper/template_{:s}_transit_time_table.tex'.
                    format(plname) )

        with open(temppath, mode='r') as f:
            data_lines = f.readlines()

        with open(templatetxt, mode='r') as ft:
            template_lines = [l for l in ft.readlines()
                              if 'XXX' not in l]
            template_lines = (
                [tl.replace('WASP-4b', plname) for tl in template_lines]
            )

            template_startix = int([ix for ix, l in
                                    enumerate(template_lines) if
                                    l.startswith('\startdata')][0])

            template_endix = int([ix for ix, l in
                                  enumerate(template_lines) if
                                  l.startswith('\enddata')][0])

            data_startix = int([ix for ix, l in enumerate(data_lines)
                                if l.startswith('\midrule')][0])

            data_endix = int([ix for ix, l in enumerate(data_lines) if
                              l.startswith(r'\bottomrule')][0])

            sl = template_lines[ : template_startix+1]
            dl = data_lines[data_startix : data_endix]
            el = template_lines[template_endix : ]

            outlines = list ( np.concatenate( (
                nparr(sl),
                nparr(dl[1:]),
                nparr(el),
            ) ) )

        with open(outtable, mode='w') as f:
            f.writelines(outlines)

        print('made {}'.format(outtable))
