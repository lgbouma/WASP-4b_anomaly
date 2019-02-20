from astropy.io.votable import parse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os

k14 = parse('../data/RVs_Knutson_2014_HIRES_WASP-4b.vot')
k14 = k14.get_first_table().to_table().to_pandas()

# Note: this table has the wrong unit for e_RV. It is meters per second. I
# checked against the published version of the table from MNRAS, rather than
# Vizier.
p11 = parse('../data/RVs_Pont_2011_HARPS_WASP-4b.vot')
p11 = p11.get_first_table().to_table().to_pandas()

w08 = pd.read_csv('../data/RVs_Wilson_2008_CORALIE_WASP-4b.txt', comment='#')

t10 = parse('../data/RVs_Triaud_2010_HARPS_WASP-4b.vot')
t10 = t10.get_first_table().to_table().to_pandas()
t10['Name'] = list(map(lambda x: x.decode('utf-8'), t10['Name']))
t10['Inst'] = list(map(lambda x: x.decode('utf-8'), t10['Inst']))
t10 = t10[t10['Name']=='WASP-4b']

# Triaud writes:
#
# "The spectrograph CORALIE continued monitoring WASP-4 and we add ten radial
# velocity measurements to the ones published in Wilson et al. (2008)."
#
# To select the triaud HARPS in-transit points, plot RV vs time, and compare to
# his model in fig 2 of the paper. Done in ../data/Triaud_2010_in_transit.png
#
# First 8 points after the 2454748 day mark are clearly out of transit...
# Claims in paper "13 points are in transit".  Then last 9 points before the
# 2454750.5 mark are out.  So BJD = 2454748.605332 to 2454748.695374 are in
# transit by his mark.
#
# Note also that Triaud's CORALIE re-reduced values are slightly different than
# original Wilson et al (2008) ones. But to use the extra Triaud 2010 CORALIE
# values, it makes sense to use his instead of Wilson's.

intra = (
    (t10['BJD'] >= 2454748.605) &
    (t10['BJD'] <= 2454748.6954) &
    (t10['Inst']=='H2')
)

sel_t10_harps = (t10['Inst']=='H2') & ~intra
sel_t10 = sel_t10_harps | (t10['Inst']=='C1')

##########
# given the confusing details above, I am choosing to concatenate:
# * the Knutson+2014 HIRES points
# * the Triaud+2010 CORALIE and out of transit HARPS points
# * the Pont+2011 HARPS points.
# ... and this is it. I am omitting the Wilson '08 CORALIE points, since they
# are redunant with Triaud+2010's.
# I will give everything (RV, e_RV) in km/s.
##########
t10 = t10.drop(['Exp'], axis=1)
t10 = t10[sel_t10]
t10['Source'] = 'Triaud+2010_vizier'

k14["Name"] = "WASP-4b"
k14["Inst"] = "HIRES"
k14["RV"] /= 1e3
k14["e_RV"] /= 1e3
k14['Source'] = 'Knutson+2014_vizier'

p11 = p11.drop(['WASP'], axis=1)
p11["Name"] = "WASP-4b"
p11["Inst"] = 'H2'
p11["e_RV"] /= 1e3
p11['Source'] = 'Pont+2011_vizierMNRAS'

outdf = pd.concat((t10, k14, p11))

outpath = '../data/RVs_all_WASP4b.csv'
outdf.to_csv(outpath, index=False)
print('made {}'.format(outpath))

# there are overlapping points between w08 and t10. how different are they?
w08['BJD'] = w08['BJD_minus_2450000'] + 2450000

t10_coralie = t10[t10['Inst'] == 'C1']

f,ax = plt.subplots()
meanrv = np.round(np.mean(t10_coralie['RV']), decimals=1)
ax.scatter(w08['BJD'], w08['RV_kmpersec']-meanrv, label='Wilson+08 CORALIE')
ax.scatter(t10_coralie['BJD'], t10_coralie['RV']-meanrv, label='Triaud+10 CORALIE')
ax.set_xlabel('bjd')
ax.set_ylabel('RV - {:.1f} [km/s]'.format(meanrv))
ax.legend(loc='best',fontsize='x-small')
outname = '../results/wilson08_vs_triaud10_rvs.png'
f.savefig(outname, dpi=350, bbox_inches='tight')
print('made {}'.format(outname))

import IPython; IPython.embed()
