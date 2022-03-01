#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import numpy as np
from astropy.table import Table, vstack, unique
import glob
import pickle

# morphofit imports
from morphofit.catalogue_managing import get_median_properties_region, create_fixed_cluster_stamp_table, \
    delete_repeating_sources


root = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/'
# cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
psf_types = ['psf_pca', 'direct']
background_estimate_methods = ['sky_free_fit', 'sky_fixed_value']
sigma_image_types = ['sigma_custom', 'sigma_int_gen']
N = 4
n_regions = N**2

for cluster_name in cluster_names:

    if cluster_name == 'macs1206':
        prefix = 'hlsp_clash_hst_30mas'
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
    else:
        prefix = 'hlsp_frontier_hst_30mas'
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']

    if cluster_name == 'macs1206':
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
        acs_waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']
        wfc3_waveband_list = ['f105w']
        pixel_scale = {'f435w': 0.030, 'f475w': 0.030, 'f606w': 0.030, 'f625w': 0.030, 'f775w': 0.030,
                       'f814w': 0.030, 'f850lp': 0.030, 'f105w': 0.030}
    else:
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
        acs_waveband_list = ['f435w', 'f606w', 'f814w']
        wfc3_waveband_list = ['f105w', 'f125w', 'f140w', 'f160w']
        pixel_scale = {'f435w': 0.030, 'f606w': 0.030, 'f814w': 0.030, 'f105w': 0.030, 'f125w': 0.030,
                       'f140w': 0.030, 'f160w': 0.030}

    root_input = root + '{}/'.format(cluster_name)
    cluster_sexcat = Table.read(root + '{}/{}_{}_multiband.forced.sexcat'.format(cluster_name, prefix, cluster_name),
                                format='fits')
    cluster_zcat = Table.read(root + '{}/{}_{}_multiband_zcatmatched.forced.sexcat'.format(cluster_name, prefix,
                                                                                           cluster_name),
                              format='fits')
    cluster_musecat = Table.read(root + '{}/{}_{}_multiband_musecatmatched.forced.sexcat'.format(cluster_name, prefix,
                                                                                                 cluster_name),
                                 format='fits')


    f = open(root + '{}/regions/x_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    x_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/y_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    y_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/ra_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    ra_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/dec_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    dec_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/mag_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    mag_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/re_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    re_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/n_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    n_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/ar_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    ar_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/pa_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    pa_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/sky_value_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    sky_value_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/sky_x_grad_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    sky_x_grad_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/sky_y_grad_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    sky_y_grad_dict = pickle.load(f)
    f.close()
    f = open(root + '{}/regions/red_chisquare_dict_{}.pkl'.format(cluster_name, cluster_name), 'rb')
    red_chisquare_dict = pickle.load(f)
    f.close()

    for i in range(n_regions):
        try:
            n_galaxies = len(x_dict['{}_f814w_psf_pca_sky_free_fit_sigma_custom_reg{}'.format(cluster_name, i)][0])
            x, x_err, y, y_err, ra, dec, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err, sky_x_grad, \
            sky_x_grad_err, sky_y_grad, sky_y_grad_err = get_median_properties_region(cluster_name, waveband_list,
                                                                                      psf_types,
                                                                                      background_estimate_methods,
                                                                                      sigma_image_types,
                                                                                      n_galaxies, i,
                                                                                      x_dict, y_dict, ra_dict, dec_dict,
                                                                                      mag_dict, re_dict, n_dict,
                                                                                      ar_dict,
                                                                                      pa_dict, sky_value_dict,
                                                                                      sky_x_grad_dict, sky_y_grad_dict)
        except Exception as e:
            print(e)
            continue
        try:
            print(table)
            newtable = create_fixed_cluster_stamp_table(cluster_name, waveband_list, cluster_sexcat, cluster_zcat,
                                                        x, x_err, y, y_err, ra, dec, mag,
                                                        mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err,
                                                        sky_value,
                                                        sky_value_err, sky_x_grad,
                                                        sky_x_grad_err, sky_y_grad, sky_y_grad_err)
            newtable.write(
                root + '{}/regions/cats/{}_{}_reg{}_mediangalfit_multiband.forced.sexcat'.format(cluster_name,
                                                                                                 prefix,
                                                                                                 cluster_name,
                                                                                                 i),
                format='fits', overwrite=True)
            table = vstack([table, newtable], join_type='exact')
        except Exception as e:
            print(e)
            table = create_fixed_cluster_stamp_table(cluster_name, waveband_list, cluster_sexcat, cluster_zcat,
                                                     x, x_err, y, y_err, ra, dec, mag,
                                                     mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value,
                                                     sky_value_err, sky_x_grad,
                                                     sky_x_grad_err, sky_y_grad, sky_y_grad_err)
            table.write(root + '{}/regions/cats/{}_{}_reg{}_mediangalfit_multiband.forced.sexcat'.format(cluster_name,
                                                                                                         prefix,
                                                                                                         cluster_name,
                                                                                                         i),
                        format='fits', overwrite=True)
        # n_galaxies = len(x_dict['{}_f814w_psf_pca_sky_free_fit_sigma_custom_reg{}'.format(cluster_name, i)][0])
        # x, x_err, y, y_err, ra, dec, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err, sky_x_grad, \
        # sky_x_grad_err, sky_y_grad, sky_y_grad_err = get_median_properties_region(cluster_name, waveband_list,
        #                                                                           psf_types,
        #                                                                           background_estimate_methods,
        #                                                                           sigma_image_types,
        #                                                                           n_galaxies, i,
        #                                                                           x_dict, y_dict, ra_dict, dec_dict,
        #                                                                           mag_dict, re_dict, n_dict, ar_dict,
        #                                                                           pa_dict, sky_value_dict,
        #                                                                           sky_x_grad_dict, sky_y_grad_dict)
        # if i == 0:
        #     table = create_fixed_cluster_stamp_table(cluster_name, waveband_list, cluster_sexcat, cluster_zcat,
        #                                              x, x_err, y, y_err, ra, dec, mag,
        #                                              mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value,
        #                                              sky_value_err, sky_x_grad,
        #                                              sky_x_grad_err, sky_y_grad, sky_y_grad_err)
        #     table.write(root + '{}/regions/cats/{}_{}_reg{}_mediangalfit_multiband.forced.sexcat'.format(cluster_name,
        #                                                                                                   prefix,
        #                                                                                                   cluster_name,
        #                                                                                                   i),
        #     format='fits', overwrite=True)
        # else:
        #     newtable = create_fixed_cluster_stamp_table(cluster_name, waveband_list, cluster_sexcat, cluster_zcat,
        #                                                 x, x_err, y, y_err, ra, dec, mag,
        #                                                 mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value,
        #                                                 sky_value_err, sky_x_grad,
        #                                                 sky_x_grad_err, sky_y_grad, sky_y_grad_err)
        #     newtable.write(root + '{}/regions/cats/{}_{}_reg{}_mediangalfit_multiband.forced.sexcat'.format(cluster_name,
        #                                                                                                      prefix,
        #                                                                                                      cluster_name,
        #                                                                                                      i),
        #                 format='fits', overwrite=True)
        #     table = vstack([table, newtable], join_type='exact')

    table = delete_repeating_sources(table, waveband_list)
    table.write(
            root + '{}/regions/cats/{}_{}_regions_mediangalfit_multiband.forced.sexcat'.format(cluster_name, prefix,
                                                                                             cluster_name),
            format='fits', overwrite=True)
