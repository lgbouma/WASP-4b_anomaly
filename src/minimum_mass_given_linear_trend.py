"""
given a linear RV trend, what's the minimum mass implied?

Wright's blog says:
"the lowest mass object that could be contributing a linear trend turns out to
be one in an eccentric orbit with e~0.5"
https://sites.psu.edu/astrowright/2015/01/06/loooooooong-period-exoplanets/
"""

import numpy as np
from astropy import units as u, constants as c

baseline = 10*u.year
gammadot = -0.04 *u.m/u.s/u.year
mstar = 0.864*u.Msun

M_min = (
    (0.0164 * u.Mjup) * (baseline/(1*u.yr))**(4/3)
    * np.abs(gammadot/(1*u.m/u.s/u.year))
    * (mstar/u.Msun)**(2/3)
)

print(M_min.to(u.Mjup))
print(M_min.to(u.Mearth))
