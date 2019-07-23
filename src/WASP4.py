# Example Keplerian fit configuration file

# Required packages for setup
import os
import pandas as pd
import numpy as np
import radvel

# Define global planetary system and dataset parameters
starname = 'WASP4'
nplanets = 1    # number of planets in the system
instnames = ['CORALIE', 'HARPS', 'HIRES']    # list of instrument names. Can be whatever you like (no spaces) but should match 'tel' column in the input file.
ntels = len(instnames)       # number of instruments with unique velocity zero-points
fitting_basis = 'per tc secosw sesinw logk'    # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
bjd0 = 0   # reference epoch for RV timestamps (i.e. this number has been subtracted off your timestamps)
planet_letters = {1: 'b'}   # map the numbers in the Parameters keys to planet letters (for plotting and tables)


# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)    # initialize Parameters object

anybasis_params['per1'] = radvel.Parameter(value=1.338231466)       # period of 1st planet
anybasis_params['tc1'] = radvel.Parameter(value=2455804.515752)     # time of inferior conjunction (transit) of 1st planet
anybasis_params['e1'] = radvel.Parameter(value=0.)          # eccentricity of 1st planet
anybasis_params['w1'] = radvel.Parameter(value=np.pi/2.)      # argument of periastron of the star's orbit for 1st planet
anybasis_params['k1'] = radvel.Parameter(value=241.1)          # velocity semi-amplitude for 1st planet

time_base = 2455470          # abscissa for slope and curvature terms (should be near mid-point of time baseline)
anybasis_params['dvdt'] = radvel.Parameter(value=0.0)        # slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
anybasis_params['curv'] = radvel.Parameter(value=0.0)        # curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)

anybasis_params['gamma_CORALIE'] = radvel.Parameter(value=0.0)     # velocity zero-point for hires_rk
anybasis_params['gamma_HARPS'] = radvel.Parameter(value=0.0)       # "                   "   hires_rj
anybasis_params['gamma_HIRES'] = radvel.Parameter(value=0.0)       # "                   "   hires_HIRESpf

anybasis_params['jit_CORALIE'] = radvel.Parameter(value=10)        # jitter for hires_rk
anybasis_params['jit_HARPS'] = radvel.Parameter(value=3)           # "      "   hires_rj
anybasis_params['jit_HIRES'] = radvel.Parameter(value=3)           # "      "   hires_HIRESpf

# Convert input orbital parameters into the fitting basis
params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

# Set the 'vary' attributes of each of the parameters in the fitting basis. A parameter's 'vary' attribute should
# be set to False if you wish to hold it fixed during the fitting process. By default, all 'vary' parameters
# are set to True.
params['secosw1'].vary = False
params['sesinw1'].vary = False

params['curv'].vary = False

params['per1'].vary = True  # if false, struggles more w/ convergence.
params['tc1'].vary = True
params['dvdt'].vary = True

# Load radial velocity data, in this example the data is contained in
# an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
datadir = "/home/luke/Dropbox/proj/WASP-4b_anomaly/data"
#datapath = os.path.join(datadir,'RVs_all_WASP4b_for_fitting_USEDINPAPER.csv')
datapath = os.path.join(datadir,'RVs_all_minustwo_WASP4b_for_fitting_20190716.csv')
data = pd.read_csv(datapath, sep=',')

# Define prior shapes and widths here.
priors = [
    radvel.prior.EccentricityPrior( nplanets ),           # Keeps eccentricity < 1
    radvel.prior.Gaussian('per1', params['per1'].value, 0.000000023), # Gaussian prior from Table 4
    radvel.prior.Gaussian('tc1', params['tc1'].value, 0.000024),      # Gaussian prior from Table 4
    radvel.prior.Gaussian('dvdt', params['dvdt'].value, 0.5),  # Gaussian prior, width 0.5 m/s/day
    radvel.prior.HardBounds('jit_CORALIE', 0.0, 50.0),
    radvel.prior.HardBounds('jit_HARPS', 0.0, 30.0),
    radvel.prior.HardBounds('jit_HIRES', 0.0, 20.0)
]

# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=0.864, mstar_err=0.0087)
planet = dict(rp1=1.321, rperr1=0.039)
