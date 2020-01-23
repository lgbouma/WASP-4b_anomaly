import numpy as np
from astropy import units as u, constants as c

# WASP-12b sanity check.
P = 1.09142*u.day
gamma_dot_value, gamma_dot_sig = -0.0009, 0.0093
N_CPS, N_data = 30, 3

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.day
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.day

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_limit = gamma_dot_limit * P / c.c
dP_dt_measured = -9.3e-10
dP_dt_measured_err = 1.1e-10

print('for WASP-12b')
print('Knutson+ 2014 reported {:d} data points, from {:d} data sets'.
     format(N_CPS, N_data))
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))

# WASP-4b. Knutson+ 2014 values.
P = 1.3382324*u.day
gamma_dot_value, gamma_dot_sig = -0.0099, 0.0054
N_CPS, N_data = 5, 4

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.day
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.day

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_plus1sigma = (
    (gamma_dot_value + gamma_dot_sig)* (u.m/u.s)/u.day * P / c.c - dP_dt_preferred
)
dP_dt_minus1sigma = (
    (gamma_dot_value - gamma_dot_sig)* (u.m/u.s)/u.day * P / c.c - dP_dt_preferred
)
dP_dt_limit = gamma_dot_limit * P / c.c
dP_dt_measured = -4.00e-10
dP_dt_measured_err = 3.8e-11

print('\nfor WASP-4b')
print('Knutson+ 2014 reported {:d} CPS data points, and used {:d} data sets'.
     format(N_CPS, N_data))
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('dP/dt_Dopplerquoted = {:.6g} +{:.6g}, -{:.6g}'.
      format(dP_dt_preferred, dP_dt_plus1sigma, np.abs(dP_dt_minus1sigma)))
print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))

# WASP-4b. Husnoo+ 2012 values.
P = 1.3382324*u.day
gamma_dot_value, gamma_dot_sig = 1023, 490
N_CPS, N_data = 14, 1

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.year
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.year

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_limit = gamma_dot_limit * P / c.c

print('\nfor WASP-4b')
print('Husnoo+ 2012 reported {:d} data points, from {:d} data sets (HARPS)'.
     format(N_CPS, N_data))
print('gamma_dot_value: {:.6g}'.format(gamma_dot_preferred.to((u.m/u.s)/u.day)))
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))


# WASP-4b. Michelle Hill values, in WASP4b4_results.pdf
P = 1.3382324*u.day
gamma_dot_value, gamma_dot_sig = -0.0253, 0.0084

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.day
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.day

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_limit = gamma_dot_limit * P / c.c

print('\nfor WASP-4b')
print('Hill reports...')
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))


# WASP-4b. 2019/01/26 Michelle Hill analysis, in WASP4b4_new_results.pdf
P = 1.3382324*u.day
gamma_dot_value, gamma_dot_sig = -0.0123, 0.0081

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.day
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.day

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_limit = gamma_dot_limit * P / c.c

print('\nfor WASP-4b')
print('Hill "new" (WASP4b4_new, full data) reports...')
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))

# WASP-4b. 2019/01/26 Fei Dai analysis, in email.
P = 1.3382324*u.day
gamma_dot_value, gamma_dot_sig = -0.00337, 0.01

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.day
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.day

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_limit = gamma_dot_limit * P / c.c

print('\nfor WASP-4b')
print('Fei Dai\'s analysis gives...')
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))


# WASP-4b. 2019/09/11 LGB analysis, for paper #2
P = 1.3382324*u.day
gamma_dot_value, gamma_dot_sig = -0.0422, 0.0028

gamma_dot_preferred = gamma_dot_value * (u.m/u.s)/u.day
gamma_dot_limit = (gamma_dot_value - 2*gamma_dot_sig) * (u.m/u.s)/u.day

gamma_dot_upper = (gamma_dot_value + gamma_dot_sig) * (u.m/u.s)/u.day
gamma_dot_lower = (gamma_dot_value - gamma_dot_sig) * (u.m/u.s)/u.day

dP_dt_preferred = gamma_dot_preferred * P / c.c
dP_dt_upper = gamma_dot_upper * P / c.c
dP_dt_lower = gamma_dot_lower * P / c.c
dP_dt_limit = gamma_dot_limit * P / c.c

print('\nfor WASP-4b')
print('LGB\'s RNAAS analysis gives...')
print('gamma_dot_value: {:.6g} +/-({:.6g}) --> gamma_dot_limit: {:.6g}'.
      format(gamma_dot_value, gamma_dot_sig, gamma_dot_limit))
print('dP/dt_RV = {:.3g} +{:.3g} - {:.3g}'.format(
    dP_dt_preferred,
    dP_dt_upper-dP_dt_preferred,
    dP_dt_preferred-dP_dt_lower
))
print('dP/dt_RV = {:.3g} +{:.3g} - {:.3g}'.format(
    dP_dt_preferred.to(u.millisecond/u.yr),
    (dP_dt_upper-dP_dt_preferred).to(u.millisecond/u.yr),
    (dP_dt_preferred-dP_dt_lower).to(u.millisecond/u.yr)
))

print('2σ upper limit on dP/dt change from Doppler shift: {:.6g}'.format(dP_dt_limit.cgs))
print('dP/dt measured: {:.6g}, +/-({:.6g})'.
      format(dP_dt_measured, dP_dt_measured_err))
print('(dP/dt_Dopplerlimit)/(dP/dt_measured): {:.6g}'.format(dP_dt_limit.cgs.value/dP_dt_measured))
print('(dP/dt_Dopplerquoted)/(dP/dt_measured): {:.6g}'.format(dP_dt_preferred.cgs.value/dP_dt_measured))

