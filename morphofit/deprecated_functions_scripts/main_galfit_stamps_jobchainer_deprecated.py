#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import os
import logging
import warnings
import sys
import numpy as np
import argparse
import h5py
from astropy.table import Table
from astropy.io import fits
import pickle

# morphofit imports
from morphofit.image_utils import cut_stamp, create_bad_pixel_mask, create_sigma_image
from morphofit.catalogue_managing import find_neighbouring_galaxies, get_best_fitting_params
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_properties_for_galfit, create_galfit_inputfile, run_galfit
from morphofit.utils import get_psf_image, get_sky_background, save_stamps_dict_properties
from morphofit import jobchainer


logger = logging.getLogger(os.path.basename(__file__)[:10])

if len(logger.handlers) == 0:
    log_formatter = logging.Formatter("%(asctime)s %(name)0.10s %(levelname)0.3s   %(message)s ", "%y-%m-%d %H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ========================================================================= STAMP FIT FUNCTION


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

    filename = root + '{}/stamps/{}_{}_{}_{}_{}_{}.INPUT'.format(cluster_name, i, muse_id[i], band, psf_type,
                                                                 background_estimate_method, sigma_image_type)
    out_image = image_path[:-5] + '_{}_{}_{}_imgblock.fits'.format(psf_type, background_estimate_method,
                                                                   sigma_image_type)
    psf_image = get_psf_image(root, psf_type, cluster_name, band)
    psf_sampling = 1
    bad_pixel_mask = create_bad_pixel_mask(number_memb[i], neigh_gal['NUMBER'], seg_image_path)
    image_size = int(round(effective_radii[i] * N))
    w = np.where(param_table['wavebands'] == '{}'.format(band))
    magnitude_zeropoint = param_table['zeropoints'][w][0]
    # n_galaxies = len(neigh_gal) + 1
    # create_constraints_file(root, constraints_file, n_galaxies) # aggiungi a run_galfit

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


# ========================================================================= JOBCHAINER FUNCTIONS

def run_range(indices):

    h5table = h5py.File(args.filename_h5pytable, 'r')

    for index in indices:

        logger.info('=============================== running on index=%d' % index)

        cluster_name = h5table['cluster_name'][index].decode('utf8')
        root = h5table['root'][index].decode('utf8')
        band = h5table['waveband'][index].decode('utf8')
        psf_type = h5table['psf_types'][index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'][index].decode('utf8')
        sigma_image_type = h5table['sigma_image_types'][index].decode('utf8')
        i = int(h5table['idxs'][index].decode('utf8'))

        N = 10
        conv_box_size = 256
        binary_file = '/cluster/scratch/torluca/gal_evo/galfit'

        epochs_acs = {'abell370': 'v1.0-epoch1', 'abell2744': 'v1.0-epoch2', 'abells1063': 'v1.0-epoch1',
                      'macs0416': 'v1.0',
                      'macs0717': 'v1.0-epoch1', 'macs1149': 'v1.0-epoch2', 'macs1206': 'v1'}
        epochs_wfc3 = {'abell370': 'v1.0-epoch2', 'abell2744': 'v1.0', 'abells1063': 'v1.0-epoch1',
                       'macs0416': 'v1.0-epoch2',
                       'macs0717': 'v1.0-epoch2', 'macs1149': 'v1.0-epoch2', 'macs1206': 'v1'}

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

        cluster_sexcat = Table.read(root + '{}/{}_{}_multiband.forced.sexcat'.format(cluster_name,
                                                                                     prefix, cluster_name),
                                    format='fits')
        muse_members = Table.read(root + '{}/{}_{}_multiband_musecatmatched.forced.sexcat'.format(cluster_name,
                                                                                               prefix, cluster_name),
                                  format='fits')
        param_table = Table.read(root + '{}/{}_param_table.fits'.format(cluster_name, cluster_name), format='fits')

        x_memb = muse_members['XWIN_IMAGE_f814w']
        y_memb = muse_members['YWIN_IMAGE_f814w']
        effective_radii = muse_members['FLUX_RADIUS_f814w']
        ra_memb = muse_members['ALPHAWIN_J2000_f814w']
        dec_memb = muse_members['DELTAWIN_J2000_f814w']
        number_memb = muse_members['NUMBER']
        try:
            root_name = muse_members['root_name']
        except Exception as e:
            print(e)
            root_name = muse_members['ID']
        muse_id = muse_members['ID']

        if band in acs_waveband_list:

            if cluster_name == 'macs1206':
                additional_text = 'hlsp_clash_hst_acs-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_acs-30mas-selfcal'

            data = fits.getdata(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                        cluster_name, band,
                                                                        epochs_acs[cluster_name]))
            seg = fits.getdata(root + '{}/{}_{}_{}_{}_drz_forced_seg.fits'.format(cluster_name,
                                                                                  additional_text,
                                                                                  cluster_name, band,
                                                                                  epochs_acs[cluster_name]))
            head = fits.getheader(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                          cluster_name, band,
                                                                          epochs_acs[cluster_name]))
            crpix1 = head['CRPIX1']
            crpix2 = head['CRPIX2']
            seg_head = fits.getheader(root + '{}/{}_{}_{}_{}_drz_forced_seg.fits'.format(cluster_name,
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
            seg = fits.getdata(root + '{}/{}_{}_{}_{}_drz_forced_seg.fits'.format(cluster_name, additional_text,
                                                                                  cluster_name, band,
                                                                                  epochs_wfc3[cluster_name]))
            head = fits.getheader(root + '{}/{}_{}_{}_{}_drz.fits'.format(cluster_name, additional_text,
                                                                          cluster_name, band,
                                                                          epochs_wfc3[cluster_name]))
            crpix1 = head['CRPIX1']
            crpix2 = head['CRPIX2']
            seg_head = fits.getheader(root + '{}/{}_{}_{}_{}_drz_forced_seg.fits'.format(cluster_name, additional_text,
                                                                                         cluster_name, band,
                                                                                         epochs_wfc3[cluster_name]))
            rms_image = fits.getdata(root + '{}/{}_{}_{}_{}_rms.fits'.format(cluster_name, additional_text,
                                                                             cluster_name, band,
                                                                             epochs_wfc3[cluster_name]))
            rms_image_head = fits.getheader(root + '{}/{}_{}_{}_{}_rms.fits'.format(cluster_name, additional_text,
                                                                                    cluster_name, band,
                                                                                    epochs_wfc3[cluster_name]))

        main_galfit_stamp(root, i, cluster_name, band, root_name, muse_id, N, x_memb, y_memb, effective_radii, data,
                          seg,
                          rms_image, head, seg_head, rms_image_head, crpix1, crpix2, acs_waveband_list,
                          wfc3_waveband_list,
                          epochs_acs[cluster_name], epochs_wfc3[cluster_name], ra_memb, dec_memb, pixel_scale,
                          cluster_sexcat, muse_members, psf_type,
                          background_estimate_method, sigma_image_type, number_memb, conv_box_size, param_table,
                          binary_file, constraints_file='none')


def get_missing(indices):

    h5table = h5py.File(args.filename_h5pytable, 'r')

    list_missing = []

    for index in indices:

        current_is_missing = False

        cluster_name = h5table['cluster_name'][index].decode('utf8')
        root = h5table['root'][index].decode('utf8')
        band = h5table['waveband'][index].decode('utf8')
        psf_type = h5table['psf_types'][index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'][index].decode('utf8')
        sigma_image_type = h5table['sigma_image_types'][index].decode('utf8')
        i = int(h5table['idxs'][index].decode('utf8'))

        try:
            file = open(root + "{}/stamps/pkl_files/x_dict_{}_{}_{}_{}_{}_stamp{}.pkl".format(cluster_name, cluster_name,
                                                                                    band, psf_type,
                                                                                    background_estimate_method,
                                                                                    sigma_image_type, i), 'rb')
            object_file = pickle.load(file)
            file.close()
            file = open(root + "{}/stamps/pkl_files/re_dict_{}_{}_{}_{}_{}_stamp{}.pkl".format(cluster_name, cluster_name,
                                                                                     band, psf_type,
                                                                                     background_estimate_method,
                                                                                     sigma_image_type, i), 'rb')
            object_file = pickle.load(file)
            file.close()
            file = open(root + "{}/stamps/pkl_files/mag_dict_{}_{}_{}_{}_{}_stamp{}.pkl".format(cluster_name, cluster_name,
                                                                                      band, psf_type,
                                                                                      background_estimate_method,
                                                                                      sigma_image_type, i), 'rb')
            object_file = pickle.load(file)
            file.close()

        except Exception as errmsg:
            logger.error('error opening catalogue: errmsg: %s' % errmsg)
            current_is_missing = True

        if current_is_missing:
            list_missing.append(index)
            logger.info('%d catalogue missing' % (index))
        else:
            logger.debug('%d tile all OK' % (index))

    n_missing = len(list_missing)
    logger.info('found missing %d' % n_missing)
    logger.info(str(list_missing))

    jobchainer.write_list_missing(list_missing)


if __name__ == '__main__':

    description = "Run galfit on stamps"

    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, action='store', default='debug',
                        choices=('critical', 'error', 'warning', 'info', 'debug'), help='logging level')
    parser.add_argument('--filename_h5pytable', type=str, action='store', default='table.h5',
                        help='Fits table of the file to run on')
    parser.add_argument('--dirpath_out', type=str, action='store', help='name of the output folder')
    logging_levels = {'critical': logging.CRITICAL, 'error': logging.ERROR, 'warning': logging.WARNING,
                      'info': logging.INFO, 'debug': logging.DEBUG}

    global args
    args = jobchainer.parse_args(parser)
    logger.setLevel(logging_levels[args.verbosity])
    jobchainer.jobchainer_init(use_exe=os.path.join(os.getcwd(), __file__), use_args=args, mem=8192, hrs=24)

    if not os.path.isabs(args.filename_h5pytable): args.filename_h5pytable = os.path.join(os.getcwd(),
                                                                                          args.filename_h5pytable)
    if not os.path.isabs(args.dirpath_out): args.dirpath_out = os.path.join(os.getcwd(), args.dirpath_out)

    jobchainer.jobchainer_main(use_args=args, run_range=run_range, get_missing=get_missing)

# python main_galfit_stamps_jobchainer_deprecated.py --mode=run_chain --n_tasks=5320 --filename_indices=/cluster/scratch/torluca/gal_evo/abells1063/indices_5320.yaml --dirpath_out=/cluster/scratch/torluca/gal_evo/abells1063 --filename_h5pytable=/cluster/scratch/torluca/gal_evo/abells1063/galfit_stamps_table.h5 --n_rerun_missing=1
