import numpy as np
from astropy import units as u, constants as c

"""
In binary systems with a single convective component, you get (BG2019)

<e^2>^(1/2) ~= 2e-5 * ( L_star/L_sun * R_env/R_sun * (1.4Msun)/M_star )^(1/3)
                    * (M_env / (0.0004M_sun) )^(1/6)

One caveat might be that the planet also has convection, and we are ignoring
that. Is this OK? Instead, check

<e^2>^(1/2) ~= 2e-5 * ( L_HJ/L_sun * R_envHJ/R_sun * (1.4Msun)/M_HJ )^(1/3)
                    * (M_envHJ / (0.0004M_sun) )^(1/6).

Also, rederive it by starting with Phinney (1992) equation (7.33). The mass
ratio is the important term, not the stellar mass as in the BG2019
approximation.

For WASP-4...

"""

def mean_ecc(L, M_planet, M_star, R_conv, M_conv, period, sma):
    # from 2018/12/02.0

    μ = (M_star*M_planet)/(M_star + M_planet)
    Ω_b = 2*np.pi/period

    return (
        8.3e-3 * (μ * Ω_b**2 * sma**2)**(-1/2) *
        ( L**2 * R_conv**2 * M_conv )**(1/6)
    ).cgs

# Me, table 1, for the star.
Rstar = 0.893*u.Rsun
Mstar = 0.864*u.Msun
Teff = 5400*u.K
age = 8*u.Gyr   # used in matching MESA
Fe_by_H = -0.05 # used in matching MESA

Lstar = ( ( c.sigma_sb * Teff**4 ) * 4*np.pi*Rstar**2 ).to(u.Lsun)

# from my R_tachocline_vs_t_varM_merged.pdf plot, fit with WebPlotDigitizer.
# origin is MESA / MIST tracks. I could simply cite the MIST tracks here too.
log10_R_tacholine = -0.25
R_tachocline = 10**(log10_R_tacholine)*u.Rsun
assert Rstar > R_tachocline
R_env_star = Rstar - R_tachocline

# note I_conv ~= (constant) * M_conv * R_conv^2
log10_I_conv = -5
I_conv = 10**log10_I_conv * u.Msun * u.Rsun**2
M_env_star = ( I_conv / (R_env_star**2) ).to(u.Msun)
M_env_star = 1e-4 * u.Msun

##########
period = 1.3382351*u.day
sma = 0.0228*u.au

# What if the planet were the driver?
Rplanet = 1.32*u.Rjup
Mplanet = 1.186*u.Mjup
Teqplanet = 1664*u.K

Lplanet = ( ( c.sigma_sb * Teqplanet**4 ) * 4*np.pi*Rplanet**2 ).to(u.Lsun)

# imagine the HJ were radiative in the inner 1%, convective for the rest.
R_tachocline_planet = 0.01*Rplanet
R_conv_planet = Rplanet - R_tachocline_planet
M_conv_planet = Mplanet

##########
print('='*42)
print('for WASP-4')
print('Rstar: {}'.format(Rstar))
print('Mstar: {}'.format(Mstar))
print('Lstar: {}'.format(Lstar))
print('R_tachocline/Rstar: {}'.format(R_tachocline/Rstar))
print('R_env_star: {}'.format(R_env_star))
print('(guess) M_env_star: {}'.format(M_env_star))
print('<e^2>^(1/2): {}'.format(
    mean_ecc(Lstar, Mplanet, Mstar, R_env_star, M_env_star, period, sma))
)

print('='*42)
print('for WASP-4b')
print('Rplanet: {}'.format(Rplanet))
print('Mplanet: {}'.format(Mplanet))
print('Lplanet: {}'.format(Lplanet))
print('R_tachocline_planet/Rplanet: {}'.
      format(R_tachocline_planet/Rplanet))
print('R_conv_planet: {}'.format(R_conv_planet))
print('(guess) M_conv_planet: {}'.format(M_conv_planet))
print('<e^2>^(1/2): {}'.format(
    mean_ecc(Lplanet, Mplanet, Mstar, R_conv_planet, M_conv_planet, period, sma))
)
