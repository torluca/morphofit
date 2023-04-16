#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
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
from morphofit.background_estimation import get_background_parameters
from morphofit.psf_estimation import get_seeings
from morphofit.image_utils import create_detection_image, create_rms_detection_image
from morphofit.run_sextractor import get_sextractor_forced_cmd, run_sex_dual_mode
from morphofit.catalogue_managing import get_multiband_catalogue
from morphofit.utils import get_logger, save_sextractor_output_files, uncompress_files

logger = get_logger(__file__)


def run_sextractor(args, telescope_name, target_field_name, root_target_field, sci_images, rms_images, wavebands,
                   pixel_scale, photo_cmd, psf_fwhm_init_guesses, sextractor_binary,  sextractor_config,
                   sextractor_params, sextractor_filter, sextractor_nnw, sextractor_checkimages,
                   sextractor_checkimages_endings, ext_star_cat, temp_dir):
    """

    :param args:
    :param telescope_name:
    :param target_field_name:
    :param root_target_field:
    :param sci_images:
    :param rms_images:
    :param wavebands:
    :param pixel_scale:
    :param photo_cmd:
    :param psf_fwhm_init_guesses:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :param ext_star_cat:
    :param temp_dir:
    :return:
    """

    se_catalogues = [name.split('fits')[0] + 'cat' for name in sci_images]
    se_forced_catalogues = [name.split('fits')[0] + args.sextractor_forced_catalogue_suffix for name in sci_images]

    detect_image_name = '{}_{}'.format(sci_images[0].split('.fits')[0], args.detection_image_suffix)
    detect_catalogue_name = detect_image_name.split('fits')[0] + args.sextractor_forced_catalogue_suffix

    if rms_images:
        detect_rms_image_name = '{}_{}'.format(rms_images[0].split('.fits')[0], args.detection_image_suffix)
    else:
        detect_rms_image_name = ''

    logger.info('=============================== get saturations')
    saturations = get_saturations(telescope_name, sci_images, wavebands)

    logger.info('=============================== get gains')
    effective_gains, instrumental_gains = get_gains(telescope_name, sci_images, wavebands)

    logger.info('=============================== get exptimes')
    exptimes = get_exposure_times(telescope_name, sci_images, wavebands)

    logger.info('=============================== get zeropoints')
    zeropoints = get_zeropoints(telescope_name, target_field_name, sci_images, wavebands)

    logger.info('=============================== get background')
    bkg_amps, bkg_sigmas = get_background_parameters(sci_images, wavebands, se_catalogues,
                                                     saturations, zeropoints, effective_gains, pixel_scale,
                                                     psf_fwhm_init_guesses, photo_cmd,
                                                     sextractor_binary, sextractor_config,
                                                     sextractor_params, sextractor_filter,
                                                     sextractor_nnw, sextractor_checkimages,
                                                     sextractor_checkimages_endings, rms_images=rms_images)

    logger.info('=============================== get seeing')
    fwhms, betas = get_seeings(telescope_name, sci_images, wavebands, se_catalogues, ext_star_cat, pixel_scale,
                               bkg_amps, psf_fwhm_init_guesses, args.sextractor_ra_keyword,
                               args.sextractor_dec_keyword, args.star_catalogue_ra_keyword,
                               args.star_catalogue_dec_keyword)

    logger.info('=============================== get param table')
    param_table = create_image_params_table(wavebands, saturations, effective_gains, instrumental_gains, exptimes,
                                            zeropoints, bkg_amps, bkg_sigmas, fwhms, betas)
    param_table_filename = os.path.join(temp_dir, '{}_{}'.format(target_field_name, args.parameters_table_suffix))
    param_table.write(param_table_filename, format='fits', overwrite=True)

    logger.info('=============================== create detection image')
    create_detection_image(sci_images, detect_image_name, wavebands, fwhms, zeropoints,
                           saturations, effective_gains, pixel_scale,
                           bkg_sigmas)

    if 'None' not in [os.path.basename(name) for name in rms_images]:
        logger.info('=============================== create rms detection image')
        create_rms_detection_image(rms_images, detect_rms_image_name, wavebands, bkg_sigmas)

    logger.info('=============================== run SExtractor')
    forced_cmd = get_sextractor_forced_cmd(sci_images, detect_image_name, detect_catalogue_name,
                                           se_forced_catalogues,
                                           wavebands, fwhms, saturations,
                                           zeropoints, effective_gains, pixel_scale,
                                           photo_cmd, sextractor_binary, sextractor_config,
                                           sextractor_params, sextractor_filter,
                                           sextractor_nnw, sextractor_checkimages,
                                           sextractor_checkimages_endings, rms_images=rms_images,
                                           detection_rms_image=detect_rms_image_name)
    run_sex_dual_mode(forced_cmd)

    logger.info('=============================== get multiband catalogue')

    multiband_catalogue_name = os.path.join(temp_dir, '{}_{}_{}'.format(telescope_name, target_field_name,
                                                                        args.multiband_catalogue_suffix))

    multiband_catalogue, multiband_catalogue_removed_stars, multiband_clean_catalogue = \
        get_multiband_catalogue(se_forced_catalogues, detect_catalogue_name, wavebands)

    multiband_catalogue.write(multiband_catalogue_name, format='fits', overwrite=True)

    multiband_catalogue_removed_stars.write(multiband_catalogue_name.split(args.multiband_catalogue_suffix)[0] +
                                            'nostars.{}'.format(args.multiband_catalogue_suffix), format='fits',
                                            overwrite=True)

    multiband_clean_catalogue.write(multiband_catalogue_name.split(args.multiband_catalogue_suffix)[0] +
                                    'final.{}'.format(args.multiband_catalogue_suffix), format='fits', overwrite=True)

    save_sextractor_output_files(temp_dir, root_target_field, param_table_filename,
                                 se_catalogues, se_forced_catalogues, multiband_catalogue_name,
                                 args.detection_image_suffix, args.multiband_catalogue_suffix,
                                 sextractor_checkimages_endings)


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
            temp_dir = os.path.join(args.temp_dir_path, 'tmp_index{:06d}'.format(index))
            os.makedirs(temp_dir, exist_ok=False)
        else:
            raise KeyError

        h5pytable_filename = os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'.format(args.h5pytable_prefix,
                                                                                            index))

        subprocess.run(['cp', h5pytable_filename, temp_dir])

        h5pytable_filename = os.path.join(temp_dir, '{}_index{:06d}.h5'.format(args.h5pytable_prefix, index))

        h5table = h5py.File(h5pytable_filename, 'r')

        target_field_name = h5table['target_field_name'][()].decode('utf8')
        root_target_field = os.path.join(args.root_path, target_field_name)

        subprocess.run(['cp', os.path.join(root_target_field, '{}_index{:06d}.tar'.format(args.image_archive_prefix,
                                                                                          index)), temp_dir])
        subprocess.run(['cp', os.path.join(root_target_field, '{}_index{:06d}.tar'
                                           .format(args.resources_archive_prefix, index)),
                        temp_dir])
        subprocess.run(['cp', h5table['sextractor_binary'][()].decode('utf8'), temp_dir])
        uncompress_files(temp_dir, temp_dir, '{}_index{:06d}.tar'.format(args.image_archive_prefix, index))
        uncompress_files(temp_dir, temp_dir, '{}_index{:06d}.tar'.format(args.resources_archive_prefix, index))

        logger.info('=============================== running SE on {}'.format(target_field_name))

        telescope_name = h5table['telescope_name'][()].decode('utf8')

        sci_images = [os.path.join(temp_dir, name.decode('utf8')) for name in h5table['sci_images'][()]]
        rms_images = [os.path.join(temp_dir, name.decode('utf8')) for name in h5table['rms_images'][()]]
        wavebands = [band.decode('utf8') for band in h5table['wavebands'][()]]
        pixel_scale = h5table['pixel_scale'][()]
        photo_cmd = [key.decode('utf8') for key in h5table['photo_cmd'][()]]
        psf_fwhm_init_guesses = h5table['psf_fwhm_init_guesses'][()]

        sextractor_binary = os.path.join(temp_dir, os.path.basename(h5table['sextractor_binary'][()].decode('utf8')))
        sextractor_config = os.path.join(temp_dir, h5table['sextractor_config'][()].decode('utf8'))
        sextractor_params = os.path.join(temp_dir, h5table['sextractor_params'][()].decode('utf8'))
        sextractor_filter = os.path.join(temp_dir, h5table['sextractor_filter'][()].decode('utf8'))
        sextractor_nnw = os.path.join(temp_dir, h5table['sextractor_nnw'][()].decode('utf8'))
        sextractor_checkimages = [key.decode('utf8') for key in h5table['sextractor_checkimages'][()]]
        sextractor_checkimages_endings = [key.decode('utf8') for key in h5table['sextractor_checkimages_endings'][()]]

        ext_star_cat = os.path.join(temp_dir, h5table['external_star_catalogue'][()].decode('utf8'))

        run_sextractor(args, telescope_name, target_field_name, root_target_field,
                       sci_images, rms_images, wavebands, pixel_scale,
                       photo_cmd, psf_fwhm_init_guesses, sextractor_binary,
                       sextractor_config, sextractor_params, sextractor_filter,
                       sextractor_nnw, sextractor_checkimages,
                       sextractor_checkimages_endings, ext_star_cat, temp_dir)

        h5table.close()

        if args.local_or_cluster == 'local':
            subprocess.run(['rm', '-rf', temp_dir])

        yield index


def check_missing(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    list_missing = []

    args = setup(args)

    for index in indices:

        h5pytable_filename = os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'.format(args.h5pytable_prefix,
                                                                                            index))
        h5table = h5py.File(h5pytable_filename, 'r')

        telescope_name = h5table['telescope_name'][()].decode('utf8')
        target_field_name = h5table['target_field_name'][()].decode('utf8')
        root_target_field = os.path.join(args.root_path, target_field_name)

        current_is_missing = False

        try:
            multiband_catalogue_name = os.path.join(root_target_field, '{}_{}_{}'
                                                    .format(telescope_name, target_field_name,
                                                            args.multiband_catalogue_suffix))
            table = Table.read(multiband_catalogue_name, format='fits')
            print(len(table))
        except Exception as errmsg:
            logger.error('error opening catalogue: errmsg: %s' % errmsg)
            current_is_missing = True

        if current_is_missing:
            list_missing.append(index)
            logger.info('%d catalogue missing' % index)
        else:
            logger.debug('%d tile all OK' % index)

        h5table.close()

    n_missing = len(list_missing)
    logger.info('found missing %d' % n_missing)
    logger.info(str(list_missing))

    return list_missing


def setup(args):
    """

    :param args:
    :return:
    """

    cwd = os.getcwd()

    description = "Run SExtractor on Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--root_path', type=str, action='store', default=cwd,
                        help='root files path')
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_prefix', type=str, action='store', default='sextractor_run_table',
                        help='h5py table prefix')
    parser.add_argument('--image_archive_prefix', type=str, action='store', default='images',
                        help='filename prefix of tar file containing images')
    parser.add_argument('--resources_archive_prefix', type=str, action='store', default='res_sextractor_files',
                        help='filename prefix of tar file containing res files')
    parser.add_argument('--temp_dir_path', type=str, action='store', default=cwd,
                        help='temporary folder where to make calculations locally, used only if --local_or_cluster'
                             ' is set to local')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')
    parser.add_argument('--sextractor_forced_catalogue_suffix', type=str, action='store', default='forced.cat',
                        help='SExtractor forced photometry suffix')
    parser.add_argument('--sextractor_checkimages_endings', type=str, action='store', default='seg',
                        help='SExtractor checkimages suffix list, only values allowed are seg,ap. seg must'
                             'be always present as first')
    parser.add_argument('--sextractor_ra_keyword', type=str, action='store', default='ALPHAWIN_J2000',
                        help='SExtractor catalogue right ascension keyword')
    parser.add_argument('--sextractor_dec_keyword', type=str, action='store', default='DELTAWIN_J2000',
                        help='SExtractor catalogue declination keyword')
    parser.add_argument('--star_catalogue_ra_keyword', type=str, action='store', default='ra',
                        help='Star catalogue right ascension keyword')
    parser.add_argument('--star_catalogue_dec_keyword', type=str, action='store', default='dec',
                        help='Star catalogue declination keyword')
    parser.add_argument('--detection_image_suffix', type=str, action='store', default='detection.fits',
                        help='Detection image suffix')
    parser.add_argument('--parameters_table_suffix', type=str, action='store', default='param_table.fits',
                        help='Parameters table suffix')
    parser.add_argument('--multiband_catalogue_suffix', type=str, action='store', default='multiband.forced.cat',
                        help='Multiband catalogue suffix')
    args = parser.parse_args(args)

    return args
