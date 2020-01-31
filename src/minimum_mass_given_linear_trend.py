"""
given a linear RV trend, what's the minimum mass implied?

Wright's blog says:
"the lowest mass object that could be contributing a linear trend turns out to
be one in an eccentric orbit with e~0.5"
https://sites.psu.edu/astrowright/2015/01/06/loooooooong-period-exoplanets/

This calculation was published in https://arxiv.org/pdf/1501.00633.pdf,
Katherina Feng et al (2015).
"""

import numpy as np
from astropy import units as u, constants as const

baseline = 10*u.year
gammadot = -0.0422 *u.m/u.s/u.day
mstar = 0.864*u.Msun

M_min = (
    (0.0164 * u.Mjup) * (baseline/(1*u.yr))**(4/3)
    * np.abs(gammadot/(1*u.m/u.s/u.year))
    * (mstar/u.Msun)**(2/3)
)

print(M_min.to(u.Mjup))
print(M_min.to(u.Mearth))

####################

#
# now double check the prefactor
#
ecc = 0.5
prefactor = (
    (1.25/(2*np.pi*const.G))**(1/3)
    * (1-ecc**2)**(1/2)
    * ( (1*u.Msun)**(2/3)*(1*u.m/u.s/u.year)*(1*u.year)**(4/3) )
)
print(prefactor.to(u.Mjup))
print((prefactor/2).to(u.Mjup))


#
# what if we used better units?
#
ecc = 0.5
prefactor = (
    (1.25/(2*np.pi*const.G))**(1/3)
    * (1-ecc**2)**(1/2)
    * ( (1*u.Msun)**(2/3)*(1*u.m/u.s/u.day)*(1*u.year)**(4/3) )
)
print(prefactor.to(u.Mjup))
print((prefactor/2).to(u.Mjup))

M_min = (
    (5.992106 * u.Mjup) * (baseline/(1*u.yr))**(4/3)
    * np.abs(gammadot/(1*u.m/u.s/u.day))
    * (mstar/u.Msun)**(2/3)
)
print(M_min)
