#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
import argparse
import h5py
from astropy.table import Table
import subprocess

# morphofit imports
from morphofit.utils import get_saturations, get_gains, get_exposure_times, get_zeropoints, create_image_params_table
from morphofit.background_estimation import get_hst_background_parameters
from morphofit.psf_estimation import get_seeings
from morphofit.image_utils import create_detection_image, create_rms_detection_image
from morphofit.run_sextractor import get_sextractor_forced_cmd, run_sex_dual_mode
from morphofit.catalogue_managing import get_multiband_catalogue
from morphofit.utils import get_logger

logger = get_logger(__file__)


def run_sextractor(telescope_name, target_name, root_target, sci_images, rms_images, wavebands, pixel_scale, photo_cmd,
                   seeing_initial_guesses, sextractor_binary,  sextractor_config, sextractor_params, sextractor_filter,
                   sextractor_nnw, sextractor_checkimages, sextractor_checkimages_endings, ext_star_cat):
    """

    :param telescope_name:
    :param target_name:
    :param root_target:
    :param sci_images:
    :param rms_images:
    :param wavebands:
    :param pixel_scale:
    :param photo_cmd:
    :param seeing_initial_guesses:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :param ext_star_cat:
    :return:
    """

    se_catalogs = [root_target + os.path.basename(img).split('.fits')[0] + '.sexcat' for img in sci_images]
    se_forced_catalogs = [root_target + os.path.basename(img).split('.fits')[0] + '_forced.sexcat' for img in
                          sci_images]

    detect_image_name = '{}_detection.fits'.format(sci_images[0][:-5])
    detect_catalog_name = detect_image_name.split('.fits')[0] + '.forced.sexcat'
    if rms_images:
        detect_rms_image_name = '{}_detection.fits'.format(rms_images[0][:-5])
    else:
        detect_rms_image_name = ''
    multiband_catalog_name = root_target + '{}_{}_multiband.forced.sexcat'.format(telescope_name, target_name)

    logger.info('=============================== get saturations')
    saturations = get_saturations(telescope_name, sci_images, wavebands)

    logger.info('=============================== get gains')
    effective_gains, instrumental_gains = get_gains(telescope_name, sci_images, wavebands)

    logger.info('=============================== get exptimes')
    exptimes = get_exposure_times(telescope_name, sci_images, wavebands)

    logger.info('=============================== get zeropoints')
    zeropoints = get_zeropoints(telescope_name, target_name, sci_images, wavebands)

    logger.info('=============================== get background')
    bkg_amps, bkg_sigmas = get_hst_background_parameters(sci_images, wavebands, se_catalogs,
                                                         saturations, zeropoints, effective_gains, pixel_scale,
                                                         photo_cmd,
                                                         sextractor_binary, sextractor_config,
                                                         sextractor_params, sextractor_filter,
                                                         sextractor_nnw, sextractor_checkimages,
                                                         sextractor_checkimages_endings, rms_images=rms_images)

    logger.info('=============================== get seeing')
    fwhms, betas = get_seeings(telescope_name, sci_images, wavebands, se_catalogs, ext_star_cat, pixel_scale,
                               bkg_amps, seeing_initial_guesses, match_type='external_star_catalogue')

    logger.info('=============================== get param table')
    param_table = create_image_params_table(wavebands, saturations, effective_gains, instrumental_gains, exptimes,
                                            zeropoints, bkg_amps, bkg_sigmas, fwhms, betas)
    param_table.write(root_target + '{}_param_table.fits'.format(target_name), format='fits', overwrite=True)

    logger.info('=============================== create detection image')
    create_detection_image(sci_images, detect_image_name, wavebands, fwhms, zeropoints,
                           saturations, effective_gains, pixel_scale,
                           bkg_sigmas)

    if rms_images:
        logger.info('=============================== create rms detection image')
        create_rms_detection_image(rms_images, detect_rms_image_name, wavebands, bkg_sigmas)

    logger.info('=============================== run SExtractor')
    forced_cmd = get_sextractor_forced_cmd(sci_images, detect_image_name, se_forced_catalogs,
                                           wavebands, fwhms, saturations,
                                           zeropoints, effective_gains, pixel_scale,
                                           photo_cmd, sextractor_binary, sextractor_config,
                                           sextractor_params, sextractor_filter,
                                           sextractor_nnw, sextractor_checkimages,
                                           sextractor_checkimages_endings, rms_images=rms_images,
                                           detection_rms_image=detect_rms_image_name)
    run_sex_dual_mode(forced_cmd)

    logger.info('=============================== get multiband catalog')
    multiband_catalogue, multiband_catalogue_removed_stars, multiband_clean_catalogue = \
        get_multiband_catalogue(se_forced_catalogs, detect_catalog_name, wavebands)
    multiband_catalogue.write(multiband_catalog_name, format='fits', overwrite=True)
    multiband_catalogue_removed_stars.write(multiband_catalog_name.split('forced.sexcat')[0] + 'nostars.forced.sexcat',
                                            format='fits', overwrite=True)
    multiband_clean_catalogue.write(multiband_catalog_name.split('forced.sexcat')[0] + 'final.forced.sexcat',
                                    format='fits', overwrite=True)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)

    for index in indices:

        logger.info('=============================== running on index={}'.format(index))

        if args.local_or_cluster == 'cluster':
            temp_dir = os.environ['TMPDIR']
        elif args.local_or_cluster == 'local':
            temp_dir = os.path.join(args.temp_dir_path, index)
            os.makedirs(temp_dir, exist_ok=False)
        else:
            raise KeyError

        subprocess.run(['cp', args.filename_h5pytable, temp_dir])

        h5pytable_filename = os.path.join(temp_dir, os.path.basename(args.filename_h5pytable))

        h5table = h5py.File(h5pytable_filename, 'r')

        # subprocess.run(['tar', '-C', cwd, '-xvf', filename_res])

        logger.info('=============================== running SE on {}'
                    .format(h5table['target_names'][index].decode('utf8')))

        telescope_name = h5table['telescope_names'].value[index].decode('utf8')
        target_name = h5table['target_names'].value[index].decode('utf8')
        root_target = h5table['root_targets'].value[index].decode('utf8')
        sci_images = [name.decode('utf8') for name in h5table['sci_images'].value[index]]
        rms_images = [name.decode('utf8') for name in h5table['rms_images'].value[index]]
        wavebands = [band.decode('utf8') for band in h5table['wavebands'].value[index]]
        pixel_scales = h5table['pixel_scales'].value[index]
        photo_cmd = [key.decode('utf8') for key in h5table['photo_cmd'].value]
        seeing_initial_guesses = h5table['seeing_initial_guesses'].value[index]
        sextractor_binary = h5table['sextractor_binary'].value.decode('utf8')
        sextractor_config = h5table['sextractor_config'].value.decode('utf8')
        sextractor_params = h5table['sextractor_params'].value.decode('utf8')
        sextractor_filter = h5table['sextractor_filter'].value.decode('utf8')
        sextractor_nnw = h5table['sextractor_nnw'].value.decode('utf8')
        sextractor_checkimages = [key.decode('utf8') for key in h5table['sextractor_checkimages'].value]
        sextractor_checkimages_endings = [key.decode('utf8') for key in h5table['sextractor_checkimages_endings'].value]
        ext_star_cat = h5table['ext_star_cat'].value[index].decode('utf8')

        run_sextractor(telescope_name, target_name, root_target,
                       sci_images, rms_images, wavebands, pixel_scales,
                       photo_cmd, seeing_initial_guesses, sextractor_binary,
                       sextractor_config, sextractor_params, sextractor_filter,
                       sextractor_nnw, sextractor_checkimages,
                       sextractor_checkimages_endings, ext_star_cat)

        h5table.close()

        yield index


def check_missing(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    list_missing = []

    filename_h5pytable = setup(args)
    h5table = h5py.File(filename_h5pytable, 'r')

    for index in indices:

        current_is_missing = False

        telescope_name = h5table['telescope_names'].value[index].decode('utf8')
        target_name = h5table['target_names'].value[index].decode('utf8')
        root_target = h5table['root_targets'].value[index].decode('utf8')

        try:
            multiband_catalog_name = root_target + '{}_{}_multiband.forced.sexcat'.format(telescope_name, target_name)
            table = Table.read(multiband_catalog_name, format='fits')
            print(len(table))
        except Exception as errmsg:
            logger.error('error opening catalogue: errmsg: %s' % errmsg)
            current_is_missing = True

        if current_is_missing:
            list_missing.append(index)
            logger.info('%d catalogue missing' % index)
        else:
            logger.debug('%d tile all OK' % index)

    n_missing = len(list_missing)
    logger.info('found missing %d' % n_missing)
    logger.info(str(list_missing))

    return list_missing


def setup(args):
    """

    :param args:
    :return:
    """

    description = "Run SExtractor on Targets"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--h5pytable_filepath', type=str, action='store', default='table.h5',
                        help='h5py table of the file to run on')
    parser.add_argument('--input_data_path', type=str, action='store', default='table.h5',
                        help='input data folder')
    parser.add_argument('--output_data_path', type=str, action='store', default='table.h5',
                        help='output data folder')
    parser.add_argument('--temp_dir_path', type=str, action='store', default='table.h5',
                        help='temporary folder where to make calculations')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')
    args = parser.parse_args(args)

    return args
