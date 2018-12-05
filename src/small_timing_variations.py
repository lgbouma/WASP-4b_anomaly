import numpy as np
from astropy import units as u, constants as c

# WASP-12b sanity check.
P = 1.09142*u.day
d = (1/(2.3122e-3))*u.pc # Gaia DR2
dec = 29.672266247*u.degree
pm_RA = (-1.570e-3*u.arcsec/u.yr).to(u.rad/u.yr)
pm_dec = (-6.958e-3*u.arcsec/u.yr).to(u.rad/u.yr)

mu = np.sqrt( pm_dec**2 + np.cos(dec)**2 * pm_RA**2 )

dP_dt_Shk = ((P*mu**2 * d)/(c.c)).cgs
dP_dt_rafikov_apsidal = ((P*mu)**2/(2*np.pi)).cgs
print('for WASP-12b')
print('mu: {:.3e} = {:.3e}'.format(mu, mu.to(u.arcsec/u.yr)))
print('dP_dt_Shk: {:.3e}'.format(dP_dt_Shk))
print('dP_dt_rafikov_apsidal: {:.3e}'.format(dP_dt_rafikov_apsidal))

# WASP-4b
P = 1.3382324*u.day
d = (1/(3.7145e-3))*u.pc # Gaia DR2
dec = (-42.061779443*u.degree).to(u.rad)
pm_RA = (9.874e-3*u.arcsec/u.yr).to(u.rad/u.yr)
pm_dec = (-87.518e-3*u.arcsec/u.yr).to(u.rad/u.yr)

mu = np.sqrt( pm_dec**2 + np.cos(dec)**2 * pm_RA**2 )

dP_dt_Shk = ((P*mu**2 * d)/(c.c)).cgs

print('for WASP-4b')
print('mu: {:.3e} = {:.3e}'.format(mu, mu.to(u.arcsec/u.yr)))
print('dP_dt_Shk: {:.3e}'.format(dP_dt_Shk))
print('dP_dt_rafikov_apsidal: {:.3e}'.format(dP_dt_rafikov_apsidal))
