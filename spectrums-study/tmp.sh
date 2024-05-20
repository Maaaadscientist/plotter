#!/bin/bash
python=$(which python)
$python /Users/allen/codes/plotter/spectrums-study/fit_spectrum.py ~/codes/sipm-massive/main_run_0458/main_run_0458_ov_6.00_sipmgr_10_tile.root ../../sipm-massive/high_k.csv 14 14-458-10-6
