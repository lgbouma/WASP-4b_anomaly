"""
We would like to convert the contrast ratios obtained through the Zorro imaging
to constraints on the masses of putative companions, and their separations from
the host star.

To do this, we followed the methodology of Montet+2014, and opted to employ the
Baraffe+2003 models for substellar mass objects and the MESA isochrones for
stellar mass objects (Choi et al XXXX).

We assumed that the system age was 5 Gyr, so that companions would have fully
contracted.

Due to the narrow-band filters of the Zorro imager, we further assumed that all
sources had blackbody spectra. While this is clearly false, we do not readily
have access to the planetary and stellar atmosphere models needed for the
correct calculation. However, the Baraffe+2003 and MESA models do report
effective temperatures and bolometric luminosities.  We opt to use these
theoretical quantities, combined with the measured Zorro filter wheels, to the
absolute magnitudes in the 562 and 832 nm Zorro bandpasses.  The apparent
magnitudes follow from information about the distance of the star.

Applying the same assumption to WASP-4 itself, we compute its apparent
magnitude in each of the Zorro filters, and finally derive the transformation
from contrast to companion mass.
"""

##########
# config #
##########

import os
import pandas as pd, numpy as np
from numpy import array as nparr

from read_mist_model import ISO

from astropy.modeling.models import BlackBody
from astropy import units as u


datadir = '../data/companion_isochrones/'

def get_merged_companion_isochrone():

    outpath = '../data/companion_isochrones/MIST_plus_Baraffe_merged.csv'

    if not os.path.exists(outpath):

        #
        # Get Teff, L for the stellar models of MIST at 5 Gyr.
        #

        # MIST isochrones, v1.2 from the website
        mistpath = os.path.join(
            datadir, 'MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_basic.iso'
        )
        iso = ISO(mistpath)

        # 10**9.7 = 5.01 Gyr. I'm OK with not interpolating further here.
        mist_age_ind = iso.age_index(9.7)
        mist_logTeff = iso.isos[mist_age_ind]['log_Teff']
        mist_logL = iso.isos[mist_age_ind]['log_L']
        mist_initial_mass = iso.isos[mist_age_ind]['initial_mass']

        mist_df = pd.DataFrame({
            'mass': mist_initial_mass,
            'lum': 10**(mist_logL),
            'teff': 10**(mist_logTeff),
        })

        #
        # 300 Mjup ~= 0.3 Msun (really 0.286), so limit our range of interest.
        # There is one point that overlaps: Mstar = 0.1Msun. For that point,
        # use the Baraffe model. 
        #
        sel = (mist_df.mass < 0.3) & (mist_df.mass > 0.1)
        mist_df = mist_df[sel]

        #
        # Get Teff, L for the Baraffe03 models 5 Gyr.
        #

        # Baraffe+2003 isochrones for substellar mass objects
        bar_df = pd.read_csv(os.path.join(datadir, 'COND03_5gyr.csv'),
                             delim_whitespace=True)

        bar_df = bar_df.drop(
            ['g','R','Mv','Mr','Mi','Mj','Mh','Mk','Mll','Mm'], axis=1
        )

        bar_df['L/Ls'] = 10**nparr(bar_df['L/Ls'])

        bar_df = bar_df.rename(
            columns={'L/Ls':'lum', 'Teff':'teff', 'M/Ms': 'mass'}
        )

        #
        # merge
        #
        mdf = pd.concat((bar_df, mist_df), sort=False).reset_index()

        mdf = mdf.drop(['index'], axis=1)

        mdf.to_csv(outpath, index=False)

    return pd.read_csv(outpath)


def abs_mag_in_zorro_bandpass(lum, teff, bandpass='562'):
    """
    lum: bolometric luminosity in units of Lsun
    teff: effective temperature in units of K
    bandpass: '562' or '832', nanometers.
    """

    if bandpass not in ['562','832']:
        raise ValueError

    bandpassdir = '../data/WASP4_zorro_speckle/filters/'
    bandpasspath = os.path.join(
        bandpassdir, 'filter_EO_{}.csv'.format(bandpass)
    )

    bpdf = pd.read_csv(bandpasspath, delim_whitespace=True)

    # the actual tabulated values here are bogus at the long wavelength end.
    # obvious from like... physics, assuming detectors are silicon. (confirmed
    # by Howell in priv. comm.) 

    width = 100 # nanometers, around the bandpass middle
    sel = np.abs(float(bandpass) - df.nm) < width

    bpdf = bpdf[sel]

    #FIXME: now create the blackbody flux objects, multiply by the transmission
    #filter, and figure out how to convert the counts to ABSOLUTE magntudes.


    import IPython; IPython.embed()


    '''
    Due to the narrow-band filters of the Zorro imager, we further assumed that all
    sources had blackbody spectra. While this is clearly false, we do not readily
    have access to the planetary and stellar atmosphere models needed for the
    correct calculation. However, the Baraffe+2003 and MESA models do report
    effective temperatures and bolometric luminosities.  We opt to use these
    theoretical quantities, combined with the measured Zorro filter wheels, to the
    absolute magnitudes in the 562 and 832 nm Zorro bandpasses.  The apparent
    magnitudes follow from information about the distance of the star.

    Applying the same assumption to WASP-4 itself, we compute its apparent
    magnitude in each of the Zorro filters, and finally derive the transformation
    from contrast to companion mass.
    '''



if __name__ == "__main__":

    df = get_merged_companion_isochrone()

    df['M_562'] = abs_mag_in_zorro_bandpass(nparr(df.lum), nparr(df.teff), '562')
    import IPython; IPython.embed()
    assert 0
    df['M_832'] = abs_mag_in_zorro_bandpass(nparr(df.lum), nparr(df.teff), '832')
