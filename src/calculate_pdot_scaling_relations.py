from astropy import units as u, constants as const
import numpy as np

const = (1*u.day)/(const.c) * (1*u.m/u.s/u.day)

print(const.to(u.millisecond/u.year))
