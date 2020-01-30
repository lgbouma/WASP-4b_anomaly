'''
make table of RVs
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np

from numpy import array as nparr

from glob import glob
import os, pickle

def main():

    df = pd.read_csv(
        '../data/RVs_all_WASP4b_for_fitting_20190911.csv', sep=','
    )
    jump_df = pd.read_csv(
        '../data/20190911_jump_wasp4_rv.csv', comment='#', sep=','
    )

    svalue = np.hstack((
        np.ones(len(df[df.tel != 'HIRES']))*np.nan,
        nparr(jump_df.svalue)
    ))

    original_references = np.array(df['Source'])
    references = []
    for ref in original_references:
        if ref == 'Triaud+2010_vizier':
            references.append('\citet{triaud_spin-orbit_2010}')
        elif ref == 'Pont+2011_vizierMNRAS':
            references.append('\citet{pont_determining_2011}')
        elif ref == 'jump_20190911':
            references.append('This work')
    references = np.array(references)

    outdf = pd.DataFrame(
        {'time': np.round(nparr(df.time),8),
         'mnvel': np.round(nparr(df.mnvel),5),
         'errvel': np.round(nparr(df.errvel),5),
         'svalue': svalue,
         'tel': nparr(df.tel),
         'provenance': references
        }
    )

    outpath = '../paper/WASP-4b_rv_data.tex'
    with open(outpath,'w') as tf:
        tf.write(outdf.to_latex(index=False, escape=False))
    print('wrote {:s}'.format(outpath))

    outdf.to_csv('../paper/WASP-4b_rv_data.csv', index=False)
    print('also wrote ../paper/WASP-4b_rv_data.csv')


    #
    # trim this tex output to remove the annoying first rows, so you
    # can \input{WASP-4b_rv_data.tex} from the table file.
    #
    with open(outpath,'r') as f:
        lines = f.readlines()
    midrule_ix = [ix for ix,l in enumerate(lines)
                  if r'\midrule' in l][0]
    bottomrule_ix = [ix for ix,l in enumerate(lines)
                     if r'\bottomrule' in l][0]
    sel_lines = lines[midrule_ix+1 : bottomrule_ix]

    with open(outpath, 'w') as f:
        f.writelines(sel_lines)
    print('made {}'.format(outpath))

if __name__=="__main__":
    main()
