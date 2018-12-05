# -*- coding: utf-8 -*-
'''
parse Bonomo et al (2017) table 8. ask: are there any hot Jupiters with small
(e<0.1) but significant eccentricties?
'''
from __future__ import division, print_function

from astropy.table import Table

import matplotlib.pyplot as plt, pandas as pd, numpy as np

from numpy import array as nparr

from astropy.coordinates import SkyCoord
from astropy import units, constants

t = Table.read("../data/Bonomo_2017_table8.vot", format="votable")

sel_eccentric = (t['Forbit']=='E')
sel_weakly_ecc = (t['ecc']<=0.1)

print('Bonomo+2017 has {:d} homogeneous transiting giant planets'.
      format(len(t)))

print('Bonomo+2017 has {:d} significant eccentric planets'.
      format(len(t[sel_eccentric])))

tsel = t[sel_eccentric & sel_weakly_ecc]
print('Bonomo+2017 has {:d} e<0.1, but significant eccentricity planets'.
      format(len(tsel)))

print(tsel)

df = t.to_pandas()
print('Bonomo+2017 has {:d} with ecc2s < {:.2f}'.format(
    len(df[df['ecc2s']<0.01]), 0.01))

print('Bonomo+2017 has {:d} with ecc2s < {:.2f}'.format(
    len(df[df['ecc2s']<0.02]), 0.02))

import IPython; IPython.embed()
