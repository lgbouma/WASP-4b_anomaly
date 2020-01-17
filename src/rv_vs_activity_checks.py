"""
(environment): py37_emcee2
"""
import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt

from plot_rvs_2020 import _get_fit_results

from scipy.stats import spearmanr

def check_rv_resid_vs_CaHK():

    df = pd.read_csv('../data/20190911_jump_wasp4_rv.csv', comment='#')

    # initialization script used to make the fix_gammadot fits
    basedir = os.path.join(
        os.path.expanduser('~'),
        "Dropbox/proj/WASP-4b_anomaly/"
    )

    setupfn = os.path.join(basedir,"src/WASP4.py")

    outputdir = os.path.join(
        basedir,
        "results/rv_fitting/LGB_20190911_fix_gammaddot"
    )

    (rvtimes, rvs, rverrs, resid, telvec, dvdt,
     curv, dvdt_merr, dvdt_perr, time_base) = _get_fit_results(
         setupfn, outputdir
    )

    sel = (telvec == 'HIRES')

    offset=2450000
    times  = rvtimes[sel] - offset
    hires_rv_resid = resid[sel]
    hires_rv_err = rverrs[sel]
    # ensure order between df and the rv/time vectors is matched
    np.testing.assert_array_equal(df.bjd-offset, times)
    hires_CaHK = np.array(df.svalue)
    hires_CaHK_err = np.array(df.svalue_err)

    r_value, p_value = spearmanr(hires_rv_resid, hires_CaHK)

    ########################################## 

    savpath='../results/20190911_rv_resid_vs_CaHK_svalue.png'

    plt.close('all')
    f,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,3))

    ax.errorbar(hires_rv_resid, hires_CaHK, xerr=hires_rv_err,
                yerr=hires_CaHK_err, fmt='k.')

    ax.text(
        0.02, 0.98, 'Spearman R: {:.2f} (p={:.3f})'.format(r_value, p_value),
        va='top', ha='left', transform=ax.transAxes, color='black'
    )

    ax.set_ylabel('CaHK Svalue')
    ax.set_xlabel('HIRES RV - orbit model [m/s]')

    f.tight_layout()
    f.savefig(savpath, dpi=300, bbox_inches='tight')
    print('made {}'.format(savpath))


if __name__ == "__main__":
    check_rv_resid_vs_CaHK()
