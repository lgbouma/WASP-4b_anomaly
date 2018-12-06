# -*- coding: utf-8 -*-
"""
calculate the required mass for a companion if Kozai-Lidov oscillations
are pumping WASP-4b's eccentricity
"""
from __future__ import division, print_function
import numpy as np

from astropy import units as u, constants as const

# Petrucci+ 2013, table 3, for all parameters
a_b = 0.0228*u.au
M_star, R_star, M_b, R_b = (
    0.89*u.Msun, 0.92*u.Rsun, 1.216*u.Mjup, 1.33*u.Rjup
)

# This number is made up, but similar to Jupiter's from Juno (Wahl+ 2016).
k_2b = 0.6

#####################
# begin calculation #
#####################

# from 2018/12/05.0, equation "*".
prefactor = 10/3 * k_2b * M_star**2 * R_b**5 * a_b**(-5) * M_b**(-1)

print('M_c > {:.2f} * (a_c/a_b)^(3/2)'.format(prefactor.to(u.Mearth)))
