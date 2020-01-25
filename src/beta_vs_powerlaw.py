import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import beta, powerlaw

n_draws = int(1e4)
a, b = 1.12, 3.09
ecc_pl = beta.rvs(a, b, size=n_draws)

eta = 0.5 # roughly

# f(x,a) = ax^{a-1} for scipy's built-in powerlaw distribution.
ecc_stellar = powerlaw.rvs(eta+1, size=n_draws)

plt.hist(ecc_pl, color='C0', label='beta (1.12,3.09)', alpha=0.5)

plt.hist(ecc_stellar, color='C1', label='powerlaw (e^{0.5})',
         alpha=0.5)
plt.xlabel('ecc')
plt.legend()
outpath = '../results/beta_vs_powerlaw.png'
plt.savefig(outpath, dpi=350)
print('made {}'.format(outpath))
