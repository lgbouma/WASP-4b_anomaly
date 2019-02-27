import numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
import os, pickle
from numpy import array as nparr
from parse import search

def _make_transit_time_csv(fnames, ticid, outdir='../data/', hasspots=False):

    t0_list, t0_merrs, t0_perrs, t0_bigerrs, picklepaths = (
        [],[],[],[],[]
    )
    seed_list = []

    transit_ix = 0
    for fname in fnames:
        transit_ix += 1

        try:
            d = pickle.load(open(fname, 'rb'))

            if hasspots:
                seed_list.append(int(search('{}_seed{}/{}',fname)[1]))

            fitparams = d['fitinfo']['finalparams']
            fiterrs = d['fitinfo']['finalparamerrs']

            t0_list.append(fitparams['t0'])
            t0_merrs.append(fiterrs['std_merrs']['t0'])
            t0_perrs.append(fiterrs['std_perrs']['t0'])
            t0_bigerrs.append(max(
                fiterrs['std_merrs']['t0'],fiterrs['std_perrs']['t0']))
            picklepaths.append(fname)

        except Exception as e:
            print(e)
            print('transit {:d} failed, continue'.format(transit_ix))
            continue

    t0, t0_merr, t0_perr, t0_bigerr = (
        nparr(t0_list),nparr(t0_merrs),nparr(t0_perrs),nparr(t0_bigerrs)
    )
    seedarr = nparr(seed_list)

    df = pd.DataFrame({
        't0_BTJD':t0, 't0_merr':t0_merr, 't0_perr':t0_perr,
        't0_bigerr':t0_bigerr, 'BJD_TDB':t0+2457000,
        'picklepath':picklepaths
    })
    if hasspots:
        df['seed'] = seedarr
        outname = (
            str(ticid)+'_has_spots_{}_transits.csv'.format(transit_ix)
        )
    else:
        outname = (
            str(ticid)+'_true_times_{:d}_transits.csv'.format(transit_ix)
        )

    outpath = os.path.join(outdir,outname)
    df.to_csv(outpath, index=False)
    print('saved to {:s}'.format(outpath))

    return outpath


def main(ticid=402026209,
         paramdir='/home/luke/Dropbox/proj/tessorbitaldecay/results/tess_lightcurve_fit_parameters/',
         fittype='mandelagol_and_line'):

    truepickledir = os.path.join(paramdir,str(ticid),'sector_*')
    seedpickledirs = os.path.join(paramdir,str(ticid)+"_inject_spot_crossings_seed??",'sector_*')

    # empirical errors -> believable error bars!
    fpattern = (
        '{:s}_{:s}_fit_empiricalerrs_t???.pickle'.
        format(str(ticid), fittype)
    )

    truenames = np.sort(glob(os.path.join(truepickledir,fpattern)))
    seednames = np.sort(glob(os.path.join(seedpickledirs,fpattern)))

    # first, make the transit time csv for the "true times" (no spots injected)
    truepath = _make_transit_time_csv(truenames, ticid)
    spotpath = _make_transit_time_csv(seednames, ticid, hasspots=True)

    df_true = pd.read_csv(truepath)
    df_spot = pd.read_csv(spotpath)

    # require only times that are measured to < 1 minute precision (all that
    # are actually seen!)
    df_true = df_true[df_true['t0_bigerr'] < 1/(60*24)]
    df_spot = df_spot[df_spot['t0_bigerr'] < 1/(60*24)]

    seeds, counts = np.unique(df_spot['seed'], return_counts=True)
    true_minus_spot_times = []
    for seed in np.sort(seeds):
        this_df = df_spot[df_spot['seed']==seed]

        if len(df_true) != len(this_df):
            # in case not done yet
            true_minus_spot_time = (
                nparr(df_true['BJD_TDB'])[:len(this_df)] -
                nparr(this_df['BJD_TDB'])
            )

        else:
            true_minus_spot_time = (
                nparr(df_true['BJD_TDB']) - nparr(this_df['BJD_TDB'])
            )

        true_minus_spot_times.append(true_minus_spot_time)

    # now in seconds
    true_minus_spot_times = np.concatenate(true_minus_spot_times)*24*60*60

    print('mean: {}'.format(np.mean(true_minus_spot_times)))
    print('median: {}'.format(np.median(true_minus_spot_times)))
    print('std: {}'.format(np.std(true_minus_spot_times)))
    print('95% of abs: {}'.format(
        np.percentile(np.abs(true_minus_spot_times), 95)))

    # what fraction of the deviations are larger than 10 seconds? 20
    # seconds? 30? 60?
    devns = [10,20,30,40,50,60]
    for devn in devns:
        devarr = np.abs(true_minus_spot_times)
        print('{:.1f}% have |true - spot| < {} sec'.format(
            len(devarr[devarr<devn])/len(devarr)*100,devn
        ))

        # the distibution is symmetric -> half have the sign that
        # would be needed to mess up ALL OF the TESS transits
        print('{:.1f}% have true - spot > {} sec'.format(
            0.5*len(devarr[~(devarr<devn)])/len(devarr)*100,devn
        ))


    plt.hist(true_minus_spot_times)
    plt.xlabel('true - spot injected time [seconds]')
    plt.ylabel('count')
    outpath = '../results/spot_crossing_time_variation_amplitude.png'
    plt.savefig(outpath, dpi=300)
    print('made {}'.format(outpath))


if __name__=="__main__":
    main()
