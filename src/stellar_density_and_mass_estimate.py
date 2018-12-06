# -*- coding: utf-8 -*-
"""
estimate the stellar density, and mass.
"""
from __future__ import division, print_function
import numpy as np

from astropy import units as u, constants as const

# use parameters from Table 4 of Hoyer+2013
i = 88.52*u.deg
Rp_by_Rs = 0.15445
a_by_Rs = 5.463
a_by_Rs_perr = 0.025
a_by_Rs_merr = 0.020
P = 1.33823204*u.day

# eq 30 of Winn (2010).
# note k^3 = 3e-3, and the density of jupiter is about equal to the density of
# the sun, so neglect the k^3*rho_planet term.

rho_star = 3*np.pi/(const.G*P**2) * a_by_Rs**3
print(rho_star.cgs)

rho_star_upper = 3*np.pi/(const.G*P**2) * (a_by_Rs+a_by_Rs_perr)**3
print(rho_star_upper.cgs)

rho_star_lower = 3*np.pi/(const.G*P**2) * (a_by_Rs-a_by_Rs_merr)**3
print(rho_star_lower.cgs)

print(
    'rhostar = {:.3f} +({:.3f}) -({:.3f}) g/cm^3'.format(
        rho_star.cgs.value,
        (rho_star_upper-rho_star).cgs.value,
        (rho_star-rho_star_lower).cgs.value
    )
)

# Keivan value
Rstar = 0.86*u.Rsun
Rstar_err = 0.02*u.Rsun

Mstar = (4*np.pi)/3*rho_star*Rstar**3
Mstar_lower = (4*np.pi)/3*rho_star_lower*(Rstar-Rstar_err)**3
Mstar_upper = (4*np.pi)/3*rho_star_upper*(Rstar+Rstar_err)**3

print(
    'Mstar = {:.3f} +({:.3f}) -({:.3f}) Msun'.format(
        Mstar.to(u.Msun).value,
        (Mstar_upper-Mstar).to(u.Msun).value,
        (Mstar-Mstar_lower).to(u.Msun).value,
    )
)
