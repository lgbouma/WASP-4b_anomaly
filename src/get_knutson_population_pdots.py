import os, json, pickle

import numpy as np, pandas as pd
from astropy import units as u, constants as c
from numpy import array as nparr
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

from astropy.coordinates import SkyCoord

from astroquery.mast import Tesscut

from astrobase.services.identifiers import simbad_to_tic
from astrobase.services.mast import tic_objectsearch
from astrobase.services.tesslightcurves import (
    get_tess_visibility_given_ticids,
    is_two_minute_spoc_lightcurve_available
)

def main():

    k14_table = 1
    tess_vis = 0

    if k14_table:
        fdf = calculate_and_format_knutson_pdot_table()
    if tess_vis:
        tdf = check_tess_visibility_and_availability(fdf)


def check_tess_visibility_and_availability(fdf):

    outpath = '../results/knutson_all_pdots_tess_visibility.csv'

    if not os.path.exists(outpath):

        ticids = [simbad_to_tic(n.rstrip(' b')) for n in nparr(fdf['planet'])]
        fdf["TICID"] = ticids

        badnames = ['TrES-2 b','TrES-3 b','TrES-4 b']
        badnameticids = [399860444, 116264089, 159742538]
        for n, ticid in zip(badnames, badnameticids):
            fdf.loc[fdf['planet']==n,'TICID'] = ticid

        sector_strs, full_sector_strs = (
            get_tess_visibility_given_ticids(nparr(fdf.TICID))
        )

        fdf['sector_str'] = sector_strs
        fdf['full_sector_str'] = full_sector_strs

        have_2min = list(
            map(is_two_minute_spoc_lightcurve_available,
                nparr(fdf.TICID).astype(str)
               )
        )
        fdf['2min_spoc_lc_available'] = have_2min

        fdf.to_csv(outpath, sep=';', index=False)
        print('made {}'.format(outpath))

    else:
        fdf = pd.read_csv(outpath, sep=';')

    return fdf


def calculate_and_format_knutson_pdot_table():

    sig_df = calculate_pdots(signifiant_trends=1)
    limit_df = calculate_pdots(signifiant_trends=0)

    #
    # update WASP-4 -- add it to significant table!
    #
    limit_df = limit_df.drop(22, axis=0)

    w4_dict = {
        'planet': 'WASP-4 b',
        'gammadot': -0.0422,
        'gammadot_pluserr': 0.0028,
        'gammadot_minuserr': 0.0027,
        'comment': 'NaN',
        'pl_name': 'WASP-4 b',
        'pl_orbper': 1.338231466,
        'Pdot': -5.94,
        'Pdot_upper_limit': -5.94+0.39,
        'Pdot_lower_limit': -5.94-0.39,
        'Pdot_perr': 0.39,
        'Pdot_merr': 0.39
    }
    w4_df = pd.DataFrame(w4_dict, index=[0])
    sig_df = pd.concat((sig_df, w4_df))

    sig_df['abs_Pdot'] = np.abs(sig_df.Pdot)
    sig_df['K14_significant'] = True
    limit_df['abs_Pdot'] = np.abs(limit_df.Pdot)
    limit_df['K14_significant'] = False

    fdf = pd.concat((sig_df.sort_values(by='abs_Pdot', ascending=False),
                     limit_df.sort_values(by='planet')))
    outpath = '../results/knutson_all_pdots.csv'
    fdf.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))

    return fdf


def calculate_pdots(signifiant_trends=1):

    if signifiant_trends:
        datapath = '../data/Knutson_2014_tables7_and_8_joined.csv'
    else:
        datapath = '../data/Knutson_2014_tables7_and_8_joined_limits.csv'

    df = pd.read_csv(datapath)
    k14_plnames = nparr(df.planet)

    gamma_dot_value = nparr(df.gammadot) * (u.m/u.s)/u.day
    gamma_dot_perr = nparr(df.gammadot_pluserr) * (u.m/u.s)/u.day
    gamma_dot_merr = nparr(df.gammadot_minuserr) * (u.m/u.s)/u.day
    gamma_dot_upper_limit = (gamma_dot_value + 1*gamma_dot_perr)
    gamma_dot_lower_limit = (gamma_dot_value - 1*gamma_dot_merr)

    #
    # get orbital periods for all the planets
    # manual periods:
    #   * XO-2b fails because host star is binary; planets are known around both
    #   components.
    #   * HAT-P-10 = WASP-11
    #
    t = NasaExoplanetArchive.get_confirmed_planets_table(all_columns=False)
    ea_df = t.to_pandas()
    selcols = ['pl_name', 'pl_orbper']
    ea_seldf = ea_df[selcols]

    mdf = df.merge(ea_seldf, left_on='planet', right_on='pl_name', how='left')
    if signifiant_trends:
        mdf.loc[mdf['planet']=='XO-2 b','pl_orbper'] = 2.615862
        mdf.loc[mdf['planet']=='HAT-P-10 b','pl_orbper'] = 3.7224690
    else:
        badnames = ['GJ436 b', 'HD149026 b']
        badnameperiods = [2.643904, '2.87588874']
        for n, per in zip(badnames, badnameperiods):
            mdf.loc[mdf['planet']==n,'pl_orbper'] = per

    #
    # calculate Pdot
    #
    P = nparr(mdf['pl_orbper'])*u.day
    dP_dt_value = gamma_dot_value * P / c.c
    dP_dt_upper_limit = gamma_dot_upper_limit * P / c.c
    dP_dt_lower_limit = gamma_dot_lower_limit * P / c.c

    #
    # insert into data frame and save
    #
    mdf['Pdot'] = dP_dt_value.to(u.millisecond/u.year)
    mdf['Pdot_upper_limit'] = dP_dt_upper_limit.to(u.millisecond/u.year)
    mdf['Pdot_lower_limit'] = dP_dt_lower_limit.to(u.millisecond/u.year)
    mdf['Pdot_perr'] = (
        dP_dt_upper_limit.to(u.millisecond/u.year)
        -
        dP_dt_value.to(u.millisecond/u.year)
    )
    mdf['Pdot_merr'] = (
        dP_dt_value.to(u.millisecond/u.year)
        -
        dP_dt_lower_limit.to(u.millisecond/u.year)
    )

    if signifiant_trends:
        outpath = '../results/knutson_significant_population_pdots.csv'
    else:
        outpath = '../results/knutson_limit_population_pdots.csv'
    mdf.to_csv(outpath, sep=';', index=False)
    print('made {}'.format(outpath))

    return mdf


if __name__ == "__main__":
    main()
