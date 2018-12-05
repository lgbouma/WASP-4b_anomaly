analysis scripts for understanding / writing up the timing variations in
WASP-4b.

Assorted Tools
====================

`small_timing_variations.py`: compute expected timing variations for
Shklovskii (1970) effect, and apsidal precession rate described by Rafikov
(2009).

`make_select_transit_times_table.py`: make `selected_transit_times.tex`, which
is copied into the latex table file with appropriate formatting under
`/paper/`.

`check_rpbyrstar_fit_vs_literature.py`: compare your measured Rp/Rstar values
against those reported by W009, G09, W09, S09.

`is_it_doppler_shift.py`: could the timing variations happen because the star
is accelerating towards us?

`check_times_vs_feis.py`: independent verification that my measured transit
times agree with Fei's


Plotting Scripts
====================

* `plot_stacked_lightcurves.py`
* `plot_O_minus_C.py`
* `plot_future.py`

maybe to be included:

* `plot_HJs_HR_diagram.py`
