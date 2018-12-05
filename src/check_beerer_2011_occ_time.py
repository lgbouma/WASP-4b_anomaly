"""
Beerer+ 2011 reported eclipse times in "BJD", and offsets relative to the Winn+
2009 ephemeris.

They probably meant BJD_UTC, because of the description in the text. We should
be able to check by recalculating the offsets. From their Table 1, the expected
offsets are 0.5 and 0.1 minutes.

The calculations below produce:

    occ time from table - predicted occ time from Winn ephem [min]
    1.0690627992153168
    1.4352265745401382
    (occ time from table - ltt correction) - predicted occ time from Winn ephem
    0.6790629029273987
    1.0452266782522202
    (UTC occ time from table - ltt correction) - predicted occ time from Winn ephem
    -0.40427058935165405
    -0.03810681402683258
    (UTC occ time from table) - predicted occ time from Winn ephem
    -0.014270693063735962
    0.3518930822610855

so none of the permutations of offsets produce the expected offset. The closest is
the negative of
    (UTC occ time from table - ltt correction) - predicted occ time from Winn ephem
which is pretty close.

So this would imply we assume the standard thing: the reported times are in
BJD_UTC, WITHOUT the light-travel time correction applied.
"""
import numpy as np

# Beerer+ 2011 give times in BJD_UTC.
t0_bjd_utc = 2455174.87731
t1_bjd_utc = 2455172.2011

# correct them by 65 seconds, per 2009 obsn (Fig 3 Eastman+ 2010)
t0_bjd_tdb = t0_bjd_utc + (65/(60*60*24))
t1_bjd_tdb = t1_bjd_utc + (65/(60*60*24))

# Winn+ 2009 give an epoch, that Beerer+2011 say they compute their offset
# from.
tc_0 = 2454697.797562
period = 1.33823214
epoch = np.arange(0, 2000, 1)
tocc_mids = tc_0 + period*epoch + period/2

tmids_first = tocc_mids[np.argmin(np.abs(tocc_mids - t0_bjd_tdb))]
tmids_second = tocc_mids[np.argmin(np.abs(tocc_mids - t1_bjd_tdb))]

print('occ time from table - predicted occ time from Winn ephem [min]')
print((t0_bjd_tdb - tmids_first)*24*60)
print((t1_bjd_tdb - tmids_second)*24*60)

# maybe beerer included the ltt correction?
print('(occ time from table - ltt correction) - predicted occ time from Winn ephem')
ltt_corr = 23.4/(60*60*24) # quoted in text
print(( (t0_bjd_tdb-ltt_corr) - tmids_first)*24*60)
print(( (t1_bjd_tdb-ltt_corr) - tmids_second)*24*60)

# maybe beerer included the ltt correction, but incorrectly were in BJD_UTC?
print('(UTC occ time from table - ltt correction) - predicted occ time from Winn ephem')
ltt_corr = 23.4/(60*60*24) # quoted in text
print(( (t0_bjd_utc-ltt_corr) - tmids_first)*24*60)
print(( (t1_bjd_utc-ltt_corr) - tmids_second)*24*60)

# maybe beerer did not include the ltt correction , but incorrectly were in BJD_UTC?
print('(UTC occ time from table) - predicted occ time from Winn ephem')
ltt_corr = 23.4/(60*60*24) # quoted in text
print(( t0_bjd_utc - tmids_first)*24*60)
print(( t1_bjd_utc - tmids_second)*24*60)

