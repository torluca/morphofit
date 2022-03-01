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
import h5py
from astropy.table import Table
from astropy.io import fits
import pickle
import argparse

# morphofit imports
from morphofit.image_utils import create_bad_pixel_region_mask, create_sigma_image
from morphofit.catalogue_managing import get_best_fitting_params
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_properties_for_regions_galfit, create_galfit_inputfile, run_galfit
from morphofit.utils import get_sky_background_region, save_regions_dict_properties, get_psf_image
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


# ========================================================================= REGION FIT FUNCTION


def main_galfit_region(root, i, cluster_name, band, region_catalogue, region_image_filename, region_seg_image_filename,
                       region_rms_image_filename, pixel_scale, binary_file,
                       psf_type, background_estimate_method, sigma_image_type, conv_box_size, param_table,
                       constraints_file='none'):
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

    table_reg_cat = Table.read(region_catalogue, format='fits')
    if len(table_reg_cat) == 0:
        raise os.error
    profile, position, ra, dec, tot_magnitude, eff_radius, \
    sersic_index, axis_ratio, pa_angle, subtract = format_properties_for_regions_galfit(region_image_filename,
                                                                                        region_catalogue, band)
    filename = root + '{}/regions/reg{}_{}_{}_{}_{}.INPUT'.format(cluster_name, i, band, psf_type,
                                                                  background_estimate_method, sigma_image_type)
    image_size = fits.getheader(region_image_filename)['NAXIS1']
    bad_pixel_mask = create_bad_pixel_region_mask(region_catalogue, region_seg_image_filename)
    out_image = region_image_filename[:-5] + '_{}_{}_{}_imgblock.fits'.format(psf_type,
                                                                              background_estimate_method,
                                                                              sigma_image_type)
    psf_image = get_psf_image(root, psf_type, cluster_name, band)
    psf_sampling = 1
    back_sky, sky_x_grad, sky_y_grad, sky_subtract = get_sky_background_region(background_estimate_method,
                                                                               param_table,
                                                                               region_catalogue,
                                                                               band)
    sigma_image, region_image_filename, magnitude_zeropoint, back_sky[0] = create_sigma_image(region_image_filename,
                                                                                              region_rms_image_filename,
                                                                                              param_table,
                                                                                              sigma_image_type,
                                                                                              back_sky[0], band)
    create_galfit_inputfile(filename, os.path.basename(region_image_filename),
                            os.path.basename(out_image),
                            os.path.basename(sigma_image), os.path.basename(psf_image), psf_sampling,
                            os.path.basename(bad_pixel_mask), constraints_file, image_size, conv_box_size,
                            magnitude_zeropoint, pixel_scale[band],
                            profile, position, tot_magnitude, eff_radius, sersic_index,
                            axis_ratio, pa_angle, subtract, back_sky, sky_x_grad, sky_y_grad, sky_subtract,
                            display_type='regular', options='0')
    cwd = os.getcwd()
    run_galfit(binary_file, cwd, filename, region_image_filename, out_image, sigma_image, psf_image, bad_pixel_mask)
    x, y, mag, re, n, ar, pa, sky_value, sky_x_grad, sky_y_grad, red_chisquare = get_best_fitting_params(out_image, len(
        table_reg_cat))
    x_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                         i)] = x
    y_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                         i)] = y
    ra_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                          i)] = ra
    dec_dict[
        '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                      i)] = dec
    mag_dict[
        '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                      i)] = mag
    re_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                          i)] = re
    n_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                         i)] = n
    ar_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                          i)] = ar
    pa_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                          i)] = pa
    sky_value_dict[
        '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                      i)] = sky_value
    sky_x_grad_dict[
        '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                      i)] = sky_x_grad
    sky_y_grad_dict[
        '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                      i)] = sky_y_grad
    red_chisquare_dict[
        '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, band, psf_type, background_estimate_method, sigma_image_type,
                                      i)] = red_chisquare

    save_regions_dict_properties(root, x_dict, 'x', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, y_dict, 'y', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, ra_dict, 'ra', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, dec_dict, 'dec', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, mag_dict, 'mag', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, re_dict, 're', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, n_dict, 'n', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, ar_dict, 'ar', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, pa_dict, 'pa', cluster_name, band, psf_type, background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, sky_value_dict, 'sky_value', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, sky_x_grad_dict, 'sky_x_grad', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, sky_y_grad_dict, 'sky_y_grad', cluster_name, band, psf_type,
                                 background_estimate_method,
                                 sigma_image_type, i)
    save_regions_dict_properties(root, red_chisquare_dict, 'red_chisquare', cluster_name, band, psf_type,
                                 background_estimate_method,
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

        print(root, cluster_name, band, psf_type, background_estimate_method, sigma_image_type, i)

        N = 4  # number of subregions per side
        n_regions = N ** 2
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

        param_table = Table.read(root + '{}/{}_param_table.fits'.format(cluster_name, cluster_name), format='fits')
        stamps_mastercat = Table.read(root + '{}/stamps/cats/{}_{}_stamps_mediangalfit_multiband.forced.sexcat'.format(
            cluster_name, prefix, cluster_name), format='fits')
        root_input = root + '{}/'.format(cluster_name)
        root_output = root_input + 'regions/'
        reg_image_filenames = []
        reg_seg_image_filenames = []
        reg_rms_image_filenames = []
        region_cats = []
        for m in range(0, N):
            for n in range(0, N):
                reg_image_filenames.append('{}{}_muse_reg{}_drz.fits'.format(root_output,
                                                                               image_name.split('_drz.fits')[0], (m*N) + n))
                reg_seg_image_filenames.append('{}{}_muse_reg{}_drz_forced_seg.fits'.format(root_output,
                                                                                              seg_image_name.split(
                                                                                                  '_drz_forced_seg.fits')[
                                                                                                  0],
                                                                                              (m*N) + n))
                reg_rms_image_filenames.append(
                    '{}{}_muse_reg{}_rms.fits'.format(root_output, rms_image_name.split('_rms.fits')[0],
                                                      (m*N) + n))
        for name in reg_image_filenames:
            region_cats.append(name[:-4] + 'sexcat')

        main_galfit_region(root, i, cluster_name, band, region_cats[i], reg_image_filenames[i],
                           reg_seg_image_filenames[i],
                           reg_rms_image_filenames[i], pixel_scale, binary_file,
                           psf_type, background_estimate_method, sigma_image_type, conv_box_size, param_table,
                           constraints_file='none')


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
            file = open(
                root + "{}/regions/pkl_files/x_dict_{}_{}_{}_{}_{}_reg{}.pkl".format(cluster_name, cluster_name,
                                                                                       band, psf_type,
                                                                                       background_estimate_method,
                                                                                       sigma_image_type, i), 'rb')
            object_file = pickle.load(file)
            file.close()
            file = open(
                root + "{}/regions/pkl_files/re_dict_{}_{}_{}_{}_{}_reg{}.pkl".format(cluster_name, cluster_name,
                                                                                        band, psf_type,
                                                                                        background_estimate_method,
                                                                                        sigma_image_type, i), 'rb')
            object_file = pickle.load(file)
            file.close()
            file = open(
                root + "{}/regions/pkl_files/mag_dict_{}_{}_{}_{}_{}_reg{}.pkl".format(cluster_name, cluster_name,
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

    description = "Run galfit on regions"

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

# python main_galfit_regions_jobchainer_deprecated.py --mode=run_chain --n_tasks=896 --filename_indices=/cluster/scratch/torluca/gal_evo/abells1063/indices_896.yaml --dirpath_out=/cluster/scratch/torluca/gal_evo/abells1063 --filename_h5pytable=/cluster/scratch/torluca/gal_evo/abells1063/galfit_regions_table.h5 --n_rerun_missing=1
