"""
calculate the eccentricity damping timescale for WASP-4b

also, sanity-check the approximations used to simplify from the
Metzger+ 2012 appendix B equations to the quoted tau_a and tau_e
timescales.
"""
import numpy as np
from astropy import units as u, constants as const

# Petrucci+ 2013, table 3, for all parameters
P = 1.33823251*u.day

#Mstar, Rstar, Mplanet, Rplanet = 0.867, 0.893, 1.189, 1.314 # USED: my table 1
Mstar, Rstar, Mplanet, Rplanet = (
    0.864*u.Msun, 0.893*u.Rsun, 1.186*u.Mjup, 1.321*u.Rjup
)
sma = 0.0226*u.au

Qplanetprime = 1e5

tau_e = ( 2 * Qplanetprime / (63*np.pi) *
         (sma/Rplanet)**5 *
         (Mplanet/Mstar) *
         P
).to(u.Myr)

print('-'*42)
print('2018/12/04.3 eq (1) timescale')
print(tau_e)
print('tau_e = {:.2f} (Qp\'/10^5) Myr'.format(tau_e.value))
print('-'*42)

# from Metzger+ 2012 appendix B
K_1 = 63/4 * const.G**(1/2) * Mstar**(3/2) * Mplanet**(-1)
K_2 = 225/16 * const.G**(1/2) * Mplanet * Mstar**(-1/2)

print('K_1: {:.3e}'.format(K_1.cgs))
print('K_2: {:.3e}'.format(K_2.cgs))

print('K_1 R_p^5: {:.3e}'.format((K_1*Rplanet**5).cgs))
print('K_2 R_star^5: {:.3e}'.format((K_2*Rstar**5).cgs))


