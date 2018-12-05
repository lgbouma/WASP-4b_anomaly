import numpy as np
from astropy import units as u, constants as c
from astropy.time import Time
from astropy import coordinates as coord

def get_epoch():
    # work out the approximate epoch of observation
    # answer: first observation was E = 1555.5, next was 1560.5.

    # obs_time = 2014-09-04 15:54–17:02
    # 2014-09-11 09:23–11:38
    sso = coord.EarthLocation.of_site('siding spring observatory')

    wasp4 = coord.SkyCoord("23 34 15.0858223039", "-42 03 41.049468209",
                           unit=(u.hourangle, u.deg), frame='icrs')

    obstime_0 = ['2014-09-04T15:54:00', '2014-09-04T17:02:00']
    time_0 = Time(obstime_0, format='isot', scale='utc', location=sso)
    ltt_bary_0 = time_0.light_travel_time(wasp4)
    time0_bjd_tdb = (time_0.tdb + ltt_bary_0).jd

    epoch0 = (time0_bjd_tdb - t0) / period

    obstime_1 = ['2014-09-11T09:23:00', '2014-09-11T11:38:00']
    time_1 = Time(obstime_1, format='isot', scale='utc', location=sso)
    ltt_bary_1 = time_1.light_travel_time(wasp4)
    time1_bjd_tdb = (time_1.tdb + ltt_bary_1).jd

    epoch1 = (time1_bjd_tdb - t0) / period


t0 = 2454823.59192
t0_err = 3e-5

ecosomega = -0.001
ecosomega_err = 3e-3

period = 1.3382320
period_err = 2e-7

epoch0 = 1555.5
epoch1 = 1560.5
meanepoch = 1557.

tocc = t0 + period*meanepoch + period/2 * (1 + 4/(np.pi)*ecosomega)
tocc_upper = (
    t0+t0_err + (period+period_err)*meanepoch +
    (period+period_err)/2 * (1 + 4/(np.pi)*(ecosomega+ecosomega_err))
)
tocc_lower = (
    t0-t0_err + (period-period_err)*meanepoch +
    (period-period_err)/2 * (1 + 4/(np.pi)*(ecosomega-ecosomega_err))
)

print('converting Zhou+ 2015\'s ecosw constraint to an occultation time:')
print('tocc %s' % tocc)
print('tocc_upper %s' % tocc_upper)
print('tocc_lower %s' % tocc_lower)
print('(tocc_upper-tocc) in minutes {:.2f}'.format((tocc_upper-tocc)*24*60))
