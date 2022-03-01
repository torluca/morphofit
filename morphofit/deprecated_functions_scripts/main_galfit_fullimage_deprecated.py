#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
from astropy.table import Table
import numpy as np
from astropy.io import fits
import os

# morphofit imports
from morphofit.image_utils import cut_muse_fov, create_bad_pixel_region_mask, create_sigma_image
from morphofit.catalogue_managing import check_parameters_for_next_fitting, get_best_fitting_params
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_properties_for_regions_galfit, create_galfit_inputfile, run_galfit
from morphofit.utils import get_sky_background_region, save_fullimage_dict_properties, get_psf_image


def main_galfit_fullimage(root, cluster_name, muse_fov_image, muse_fov_seg_image, muse_fov_rms_image, regions_mastercat,
                          band, psf_type, background_estimate_method,
                          sigma_image_type, conv_box_size, param_table, binary_file, pixel_scale,
                          constraints_file = 'none'):
    x_dict = {}
    y_dict = {}
    ra_dict = {}
    dec_dict = {}
    mag_dict = {}
    re_dict = {}
    n_dict = {}
    ar_dict = {}
    pa_dict = {}
    sky_value_dict = {}
    sky_x_grad_dict = {}
    sky_y_grad_dict = {}
    red_chisquare_dict = {}

    profile, position, ra, dec, tot_magnitude, eff_radius, \
    sersic_index, axis_ratio, pa_angle, subtract = format_properties_for_regions_galfit(muse_fov_image,
                                                                                       regions_mastercat, band)
    image_size = fits.getheader(muse_fov_image)['NAXIS1']
    bad_pixel_mask = create_bad_pixel_region_mask(regions_mastercat, muse_fov_seg_image)
    filename = root + '{}/full_image/{}_{}_{}_{}.INPUT'.format(cluster_name, band, psf_type,
                                                                  background_estimate_method, sigma_image_type)
    out_image = muse_fov_image[:-5] + '_{}_{}_{}_imgblock.fits'.format(psf_type,
                                                                       background_estimate_method,
                                                                       sigma_image_type)
    psf_image = get_psf_image(root, psf_type, cluster_name, band)
    psf_sampling = 1
    back_sky, sky_x_grad, sky_y_grad, sky_subtract = get_sky_background_region(
        background_estimate_method,
        param_table,
        regions_mastercat,
        band)
    sigma_image, muse_fov_image, magnitude_zeropoint, back_sky[0] = create_sigma_image(muse_fov_image,
                                                                                       muse_fov_rms_image,
                                                                                       param_table,
                                                                                       sigma_image_type,
                                                                                       back_sky[0],
                                                                                       band)
    create_galfit_inputfile(filename, os.path.basename(muse_fov_image),
                            os.path.basename(out_image),
                            os.path.basename(sigma_image), os.path.basename(psf_image), psf_sampling,
                            os.path.basename(bad_pixel_mask), constraints_file, image_size,
                            conv_box_size,
                            magnitude_zeropoint, pixel_scale[band],
                            profile, position, tot_magnitude, eff_radius, sersic_index,
                            axis_ratio, pa_angle, subtract, back_sky, sky_x_grad, sky_y_grad,
                            sky_subtract,
                            display_type='regular', options='0')
    cwd = os.getcwd()
    run_galfit(binary_file, cwd, filename, muse_fov_image, out_image, sigma_image, psf_image,
               bad_pixel_mask)
    x, y, mag, re, n, ar, pa, sky_value, sky_x_grad, sky_y_grad, red_chisquare = get_best_fitting_params(
        out_image, len(regions_mastercat))
    x_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                   sigma_image_type)] = x
    y_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                   sigma_image_type)] = y
    ra_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                    sigma_image_type)] = ra
    dec_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                     sigma_image_type)] = dec
    mag_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                     sigma_image_type)] = mag
    re_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                    sigma_image_type)] = re
    n_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                   sigma_image_type)] = n
    ar_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                    sigma_image_type)] = ar
    pa_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                    sigma_image_type)] = pa
    sky_value_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                           sigma_image_type)] = sky_value
    sky_x_grad_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                            sigma_image_type)] = sky_x_grad
    sky_y_grad_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                            sigma_image_type)] = sky_y_grad
    red_chisquare_dict['{}_{}_{}_{}_{}'.format(cluster_name, band, psf_type, background_estimate_method,
                                               sigma_image_type)] = red_chisquare

    save_fullimage_dict_properties(root, x_dict, 'x', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, y_dict, 'y', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, ra_dict, 'ra', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, dec_dict, 'dec', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, mag_dict, 'mag', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, re_dict, 're', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, n_dict, 'n', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, ar_dict, 'ar', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, pa_dict, 'pa', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, sky_value_dict, 'sky_value', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, sky_x_grad_dict, 'sky_x_grad', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, sky_y_grad_dict, 'sky_y_grad', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type)
    save_fullimage_dict_properties(root, red_chisquare_dict, 'red_chisquare', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type)



root = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/'
root_res = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/res/'
binary_file = '/Users/torluca/galfit'
#cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
epochs_acs = {'abell370':'v1.0-epoch1', 'abell2744':'v1.0-epoch2', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0',
              'macs0717':'v1.0-epoch1', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
epochs_wfc3 = {'abell370':'v1.0-epoch2', 'abell2744':'v1.0', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0-epoch2',
               'macs0717':'v1.0-epoch2', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
psf_types = ['psf_pca','direct']
background_estimate_methods = ['sky_free_fit','sky_fixed_value']
sigma_image_types = ['sigma_custom','sigma_int_gen']
conv_box_size = 256

for cluster_name in cluster_names:
    print('cluster name: {}'.format(cluster_name))

    if cluster_name == 'macs1206':
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
        acs_waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']
        wfc3_waveband_list = ['f105w']
        pixel_scale = {'f435w': 0.030, 'f475w': 0.030, 'f606w': 0.030, 'f625w': 0.030, 'f775w': 0.030,
                       'f814w': 0.030, 'f850lp': 0.030, 'f105w': 0.030}
        prefix = 'hlsp_clash_hst_30mas'
    else:
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
        acs_waveband_list = ['f435w', 'f606w', 'f814w']
        wfc3_waveband_list = ['f105w', 'f125w', 'f140w', 'f160w']
        pixel_scale = {'f435w': 0.030, 'f606w': 0.030, 'f814w': 0.030, 'f105w': 0.030, 'f125w': 0.030,
                       'f140w': 0.030, 'f160w': 0.030}
        prefix = 'hlsp_frontier_hst_30mas'

    root_input = root + '{}/'.format(cluster_name)
    root_output = root_input + 'full_image/'

    cluster_sexcat = Table.read(root + '{}/{}_{}_multiband.forced.sexcat'.format(cluster_name, prefix, cluster_name),
                                format='fits')
    param_table = Table.read(root + '{}/{}_param_table.fits'.format(cluster_name, cluster_name), format='fits')
    stamps_mastercat = Table.read(root + '{}/stamps/cats/{}_{}_stamps_mediangalfit_multiband.forced.sexcat'.format(
        cluster_name, prefix, cluster_name), format='fits')
    regions_mastercat = Table.read(root + '{}/regions/cats/{}_{}_regions_mediangalfit_multiband.forced.sexcat'.format(
            cluster_name, prefix, cluster_name), format='fits')

    for band in waveband_list:
        print('waveband: {}'.format(band))

        w = np.where(param_table['wavebands'] == '{}'.format(band))
        magnitude_zeropoint = param_table['zeropoints'][w][0]

        if band in acs_waveband_list:
            if cluster_name == 'macs1206':
                additional_text = 'hlsp_clash_hst_acs-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_acs-30mas-selfcal'
            image_name = '{}_{}_{}_{}_drz.fits'.format(additional_text,
                                                                              cluster_name, band,
                                                                              epochs_acs[cluster_name])
            seg_image_name = '{}_{}_{}_{}_drz_forced_seg.fits'.format(additional_text,
                                                                                             cluster_name, band,
                                                                                             epochs_acs[cluster_name])
            rms_image_name = '{}_{}_{}_{}_rms.fits'.format(additional_text,
                                                                                  cluster_name, band,
                                                                                  epochs_acs[cluster_name])
        else:
            if cluster_name == 'macs1206':
                additional_text = 'hlsp_clash_hst_wfc3ir-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_wfc3-30mas-bkgdcor'
            image_name = '{}_{}_{}_{}_drz.fits'.format(additional_text,
                                                                              cluster_name, band,
                                                                              epochs_wfc3[cluster_name])
            seg_image_name = '{}_{}_{}_{}_drz_forced_seg.fits'.format(additional_text,
                                                                                             cluster_name, band,
                                                                                             epochs_wfc3[cluster_name])
            rms_image_name = '{}_{}_{}_{}_rms.fits'.format(additional_text,
                                                                                  cluster_name, band,
                                                                                  epochs_wfc3[cluster_name])

        muse_fov_image, muse_fov_seg_image, muse_fov_rms_image = cut_muse_fov(root_input, image_name,
                                                                              seg_image_name, rms_image_name,
                                                                              root_input, stamps_mastercat)

        regions_mod_mastercat = check_parameters_for_next_fitting(regions_mastercat, waveband_list)

        for psf_type in psf_types:
            print('psf_type: {} {} {}'.format(psf_type, cluster_name, band))
            for background_estimate_method in background_estimate_methods:
                print('background method: {} {} {}'.format(background_estimate_method, cluster_name, band))
                for sigma_image_type in sigma_image_types:
                    print('sigma image: {} {} {}'.format(sigma_image_type, cluster_name, band))

                    main_galfit_fullimage(root, cluster_name, muse_fov_image, muse_fov_seg_image, muse_fov_rms_image,
                                          regions_mod_mastercat, band, psf_type, background_estimate_method,
                                          sigma_image_type, conv_box_size, param_table, binary_file, pixel_scale,
                                          constraints_file = 'none')
