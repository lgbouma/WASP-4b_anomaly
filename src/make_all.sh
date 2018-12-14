#!/usr/bin/env bash

##########################################
# USAGE: ./make_all.sh
##########################################

echo "making the transit time table"

python make_select_transit_times_table.py

echo "making plots"

python plot_phasefold.py

python plot_stacked_lightcurves.py

python plot_arrived_early.py

python plot_O_minus_C.py

python plot_population_vs_wasp4.py

python plot_future.py

python plot_hjs.py
