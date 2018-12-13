# -*- coding: utf-8 -*-
"""
calculate the required mass for a companion if Kozai-Lidov oscillations
are pumping WASP-4b's eccentricity
"""
from __future__ import division, print_function
import numpy as np

from astropy import units as u, constants as const

# parameters from my table
a_b = 0.0227*u.au
M_star, R_star, M_b, R_b = (
    0.867*u.Msun, 0.893*u.Rsun, 1.189*u.Mjup, 1.314*u.Rjup
)

# from equation (52) of the latex'd equation in Avery's email

O_minus_C_RV = 15.2 * u.m/u.s       # table 3 of Triaud+ 2010.

prefactor = O_minus_C_RV * (M_star*a_b/const.G)**(1/2)

print('M_c < {:.2f} * (a_c/a_b)^(1/2) * f^(-1/2)'.
      format(prefactor.to(u.Mearth)))
