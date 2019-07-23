"""
a figure for talks, showing tidal inspiral
"""

import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 1, 1e-5)
#r = np.logspace(-5,0,int(1e5))
theta = 30 * np.pi * r**5

#fig = plt.Figure(figsize=(4,4))
ax = plt.subplot(111, projection='polar')

ax.plot(theta, r, c='k', lw=1.5)
ax.set_rmax(1)
#ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
#ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
#ax.grid(True)

plt.axis('off')

plt.savefig('../results/tidal_inspiral.png', dpi=350, bbox_inches='tight')
plt.savefig('../results/tidal_inspiral.pdf', bbox_inches='tight')
