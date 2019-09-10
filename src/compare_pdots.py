import numpy as np
from astropy import units as u, constants as c

period = 1.3382324*u.day

# Baluev Figure 4.  His T_d = -P/Pdot.
T_d_inv_min = 0.075*(1/u.Myr)
T_d_inv_max = 0.100*(1/u.Myr)

T_d_max = 1/T_d_inv_min
T_d_min = 1/T_d_inv_max

Pdot_max = (-period/T_d_min).to(u.millisecond/u.yr)
Pdot_min = (-period/T_d_max).to(u.millisecond/u.yr)

print(Pdot_max)
print(Pdot_min)
