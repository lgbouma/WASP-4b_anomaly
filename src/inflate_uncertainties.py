"""
One approach to analyzing such data would be to inflate the transit
measurement uncertainties, and lower the reduced $\chi^2$.  We do not
think that such an approach is warranted, because it would not change
the result that the quadratic model is strongly preferred.  Instead,
we opt to simply inflate the uncertainties reported for each model by
a factor of $\sqrt{\chi^2_{\rm red}}$ ($\approx$1.73$\times$ for the
linear model, and $\approx$1.41$\times$ for the quadtratic).
"""

from numpy import sqrt
from astropy import units as u, constants as c

# error provenance:
# /Users/luke/Dropbox/proj/tessorbitaldecay/results/model_comparison/WASP-4b_20200127/model_comparison_output.txt

linear_chisq_red = 3.0

t0_err = 14
P_err = 15
print('t0_err: {} -> {}'.format(t0_err, sqrt(linear_chisq_red)*t0_err ))
print('P_err: {} -> {}'.format(P_err, sqrt(linear_chisq_red)*P_err ))


quad_chisq_red = 2.0
t0_err = 22
P_err = 17
Pdot_err = 2.815
print('t0_err: {} -> {}'.format(t0_err, sqrt(quad_chisq_red)*t0_err ))
print('P_err: {} -> {}'.format(P_err, sqrt(quad_chisq_red)*P_err ))
print('Pdot_err: {} -> {}'.format(Pdot_err, sqrt(quad_chisq_red)*Pdot_err ))

Pdot_err = sqrt(quad_chisq_red)*Pdot_err * 1e-11* (u.m/u.m)
print('Pdot_err: {} -> {}'.format(Pdot_err,
                                  Pdot_err.to(u.millisecond/u.year) ))

