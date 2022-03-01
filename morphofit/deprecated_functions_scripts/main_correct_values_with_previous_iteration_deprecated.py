#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
from astropy.table import Table
import numpy as np

# morphofit imports


def substitute_values(path_table_prev_iter, path_table_current_iter, waveband_list, properties_list):

    table_prev_iter = Table.read(path_table_prev_iter, format='fits')
    table_current_iter = Table.read(path_table_current_iter, format='fits')

    for band in waveband_list:
        w = np.where((table_current_iter['mag_err_{}'.format(band)] == 0.) &
                     (table_current_iter['re_err_{}'.format(band)] == 0.) &
                     (table_current_iter['n_err_{}'.format(band)] == 0.))
        for colname in properties_list:
            table_current_iter['{}_{}'.format(colname, band)][w] = table_prev_iter['{}_{}'.format(colname, band)][w]
            table_current_iter['{}_{}'.format(colname, band)] = np.nan_to_num(table_current_iter['{}_{}'.format(colname, band)])

    table_current_iter.write(path_table_current_iter, format='fits', overwrite=True)


root = '/cluster/scratch/torluca/gal_evo/'
properties_list = ['x','x_err','y','y_err','mag','mag_err','re','re_err','n','n_err','ar','ar_err','pa','pa_err']
cluster_names = ['abells1063']

for cluster_name in cluster_names:
    if cluster_name == 'macs1206':
        prefix = 'hlsp_clash_hst_30mas'
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
    else:
        prefix = 'hlsp_frontier_hst_30mas'
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']

    path_table_prev_iter = root + '{}/stamps/cats/{}_{}_stamps_mediangalfit_multiband.forced.sexcat'.format(
        cluster_name, prefix, cluster_name)
    path_table_current_iter = root + '{}/regions/cats/{}_{}_regions_mediangalfit_multiband.forced.sexcat'.format(
        cluster_name, prefix, cluster_name)

    substitute_values(path_table_prev_iter, path_table_current_iter, waveband_list, properties_list)
