#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import os
import numpy as np
from astropy.table import Table
from astropy.io import fits

# morphofit imports
from morphofit.image_utils import cut_stamp, create_bad_pixel_mask, create_sigma_image
from morphofit.catalogue_managing import find_neighbouring_galaxies, get_best_fitting_params
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_properties_for_galfit, create_galfit_inputfile, run_galfit
from morphofit.utils import get_psf_image, get_sky_background, save_stamps_dict_properties


def main_galfit_stamp(root, i, cluster_name, band, root_name, muse_id, N, x_memb, y_memb, effective_radii, data, seg,
                      rms_image, head, seg_head, rms_image_head, crpix1, crpix2, acs_waveband_list, wfc3_waveband_list,
                      epoch_acs, epoch_wfc3, ra_memb, dec_memb, pixel_scale, cluster_sexcat, muse_members, psf_type,
                      background_estimate_method, sigma_image_type, number_memb, conv_box_size, param_table,
                      binary_file, constraints_file = 'none'):

    print('stamp number: {} {} {}'.format(i, cluster_name, band))

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

    image_path, seg_image_path, rms_image_path = cut_stamp(root, cluster_name, band, root_name[i],
                                                           muse_id[i], N, x_memb[i], y_memb[i],
                                                           effective_radii[i], data, seg,
                                                           rms_image, head, seg_head,
                                                           rms_image_head,
                                                           crpix1, crpix2, acs_waveband_list, wfc3_waveband_list,
                                                           epoch_acs, epoch_wfc3)
    neigh_gal = find_neighbouring_galaxies(ra_memb[i], dec_memb[i], N, effective_radii[i],
                                           pixel_scale[band], cluster_sexcat)
    profile, position, ra, dec, tot_magnitude, eff_radius, \
    sersic_index, axis_ratio, pa_angle, subtract = format_properties_for_galfit(muse_members[i],
                                                                                neigh_gal, band,
                                                                                image_path)

    filename = root + '{}/stamps/{}_{}_{}.INPUT'.format(cluster_name, root_name[i], muse_id[i], band)
    out_image = image_path[:-5] + '_{}_{}_{}_imgblock.fits'.format(psf_type, background_estimate_method,
                                                                   sigma_image_type)
    psf_image = get_psf_image(root, psf_type, cluster_name, band)
    psf_sampling = 1
    bad_pixel_mask = create_bad_pixel_mask(number_memb[i], neigh_gal['NUMBER'], seg_image_path)
    image_size = int(round(effective_radii[i] * N))
    w = np.where(param_table['wavebands'] == '{}'.format(band))
    magnitude_zeropoint = param_table['zeropoints'][w][0]

    back_sky, sky_x_grad, sky_y_grad, sky_subtract = get_sky_background(background_estimate_method,
                                                                        param_table, band)
    sigma_image, image_path, magnitude_zeropoint, back_sky[0] = create_sigma_image(image_path, rms_image_path,
                                                                                   param_table, sigma_image_type,
                                                                                   back_sky[0], band)
    create_galfit_inputfile(filename, os.path.basename(image_path),
                            os.path.basename(out_image),
                            os.path.basename(sigma_image), os.path.basename(psf_image), psf_sampling,
                            os.path.basename(bad_pixel_mask), constraints_file, image_size, conv_box_size,
                            magnitude_zeropoint, pixel_scale[band],
                            profile, position, tot_magnitude, eff_radius, sersic_index,
                            axis_ratio, pa_angle, subtract, back_sky, sky_x_grad, sky_y_grad, sky_subtract,
                            display_type='regular', options='0')
    cwd = os.getcwd()
    run_galfit(binary_file, cwd, filename, image_path, out_image, sigma_image, psf_image, bad_pixel_mask)
    x, y, mag, re, n, ar, pa, sky_value, sky_x_grad, sky_y_grad, red_chisquare = get_best_fitting_params(out_image, len(
        neigh_gal) + 1)
    x_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = x
    y_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = y
    ra_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = ra
    dec_dict[
        '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = dec
    mag_dict[
        '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = mag
    re_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = re
    n_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = n
    ar_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = ar
    pa_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)] = pa
    sky_value_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                                   i)] = sky_value
    sky_x_grad_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                                    i)] = sky_x_grad
    sky_y_grad_dict['{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                                    i)] = sky_y_grad
    red_chisquare_dict[
        '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                        i)] = red_chisquare

    save_stamps_dict_properties(root, x_dict, 'x', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, y_dict, 'y', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, ra_dict, 'ra', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, dec_dict, 'dec', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, mag_dict, 'mag', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, re_dict, 're', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, n_dict, 'n', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, ar_dict, 'ar', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, pa_dict, 'pa', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, sky_value_dict, 'sky_value', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, sky_x_grad_dict, 'sky_x_grad', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, sky_y_grad_dict, 'sky_y_grad', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)
    save_stamps_dict_properties(root, red_chisquare_dict, 'red_chisquare', cluster_name, band, psf_type, background_estimate_method,
                                sigma_image_type, i)


root = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/'
root_res = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/res/'
binary_file = '/Users/torluca/galfit'
#cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
epochs_acs = {'abell370':'v1.0-epoch1', 'abell2744':'v1.0-epoch2', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0',
              'macs0717':'v1.0-epoch1', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
epochs_wfc3 = {'abell370':'v1.0-epoch2', 'abell2744':'v1.0', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0-epoch2',
               'macs0717':'v1.0-epoch2', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
psf_types = ['psf_pca','direct'] # 'direct' from stars in the field, 'indirect' from MultiKing
background_estimate_methods = ['sky_free_fit','sky_fixed_value'] # 'sky_fixed_value' from param_table, 'sky_fixed_value' from galapagos skymap
sigma_image_types = ['sigma_custom','sigma_int_gen'] # from formula or manually created, 'sigma_int_gen'internally generated by galfit
N = 10 # from visual inspection
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

    cluster_sexcat = Table.read(root + '{}/{}_{}_multiband.forced.sexcat'.format(cluster_name, prefix, cluster_name),
                                format='fits')
    muse_members = Table.read(
        root + '{}/{}_{}_multiband_musematched.forced.sexcat'.format(cluster_name, prefix, cluster_name), format='fits')
    param_table = Table.read(root + '{}/{}_param_table.fits'.format(cluster_name, cluster_name), format='fits')

    x_memb = muse_members['XWIN_IMAGE_f814w']
    y_memb = muse_members['YWIN_IMAGE_f814w']
    effective_radii = muse_members['FLUX_RADIUS_f814w']
    ra_memb = muse_members['ALPHAWIN_J2000_f814w']
    dec_memb = muse_members['DELTAWIN_J2000_f814w']
    number_memb = muse_members['NUMBER']
    root_name = muse_members['root_name']
    muse_id = muse_members['ID']

    for band in waveband_list:
        print('waveband: {}'.format(band))

        if band in acs_waveband_list:

            if cluster_name == 'macs1206':
                additional_text = 'hlsp_clash_hst_acs-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_acs-30mas-selfcal'

            data = fits.getdata(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                        cluster_name, band,
                                                                        epochs_acs[cluster_name]))
            seg = fits.getdata(root + '{}/{}_{}_{}_{}_drz.forced_seg.fits'.format(cluster_name,
                                                                                  additional_text,
                                                                                  cluster_name, band,
                                                                                  epochs_acs[cluster_name]))
            head = fits.getheader(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                          cluster_name, band,
                                                                          epochs_acs[cluster_name]))
            crpix1 = head['CRPIX1']
            crpix2 = head['CRPIX2']
            seg_head = fits.getheader(root + '{}/{}_{}_{}_{}_drz.forced_seg.fits'.format(cluster_name,
                                                                                         additional_text,
                                                                                         cluster_name, band,
                                                                                         epochs_acs[cluster_name]))
            rms_image = fits.getdata(root + '{}/{}_{}_{}_{}_rms.fits'.format(cluster_name, additional_text,
                                                                             cluster_name, band,
                                                                             epochs_acs[cluster_name]))
            rms_image_head = fits.getheader(root + '{}/{}_{}_{}_{}_rms.fits'.format(cluster_name, additional_text,
                                                                                    cluster_name, band,
                                                                                    epochs_acs[cluster_name]))
        else:

            if cluster_name == 'macs1206':
                additional_text = 'hlsp_clash_hst_wfc3ir-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_wfc3-30mas-bkgdcor'

            data = fits.getdata(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                        cluster_name, band, epochs_wfc3[cluster_name]))
            seg = fits.getdata(root + '{}/{}_{}_{}_{}_drz.forced_seg.fits'.format(cluster_name, additional_text,
                                                                                  cluster_name, band,
                                                                                  epochs_wfc3[cluster_name]))
            head = fits.getheader(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                          cluster_name, band,
                                                                          epochs_wfc3[cluster_name]))
            crpix1 = head['CRPIX1']
            crpix2 = head['CRPIX2']
            seg_head = fits.getheader(root + '{}/{}_{}_{}_{}_drz.forced_seg.fits'.format(cluster_name, additional_text,
                                                                                         cluster_name, band,
                                                                                         epochs_wfc3[cluster_name]))
            rms_image = fits.getdata(root + '{}/{}_{}_{}_{}_rms.fits'.format(cluster_name, additional_text,
                                                                             cluster_name, band,
                                                                             epochs_wfc3[cluster_name]))
            rms_image_head = fits.getheader(root + '{}/{}_{}_{}_{}_rms.fits'.format(cluster_name, additional_text,
                                                                                    cluster_name, band,
                                                                                    epochs_wfc3[cluster_name]))

        for psf_type in psf_types:
            print('psf_type: {} {} {}'.format(psf_type, cluster_name, band))
            for background_estimate_method in background_estimate_methods:
                print('background method: {} {} {}'.format(background_estimate_method, cluster_name, band))
                for sigma_image_type in sigma_image_types:
                    print('sigma image: {} {} {}'.format(sigma_image_type, cluster_name, band))
                    for i in range(len(x_memb)):
                        main_galfit_stamp(root, i, cluster_name, band, root_name, muse_id, N, x_memb, y_memb,
                                          effective_radii, data, seg,
                                          rms_image, head, seg_head, rms_image_head, crpix1, crpix2, acs_waveband_list,
                                          wfc3_waveband_list,
                                          epochs_acs[cluster_name], epochs_wfc3[cluster_name], ra_memb, dec_memb,
                                          pixel_scale, cluster_sexcat,
                                          muse_members, psf_type,
                                          background_estimate_method, sigma_image_type, number_memb, conv_box_size,
                                          param_table, binary_file,
                                          constraints_file='none')
