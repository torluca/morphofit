#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
import h5py
import glob
import numpy as np
import morphofit
from pkg_resources import resource_filename
import argparse

# morphofit imports
from morphofit.utils import get_logger, compress_list_of_files, add_to_compressed_list_of_files

logger = get_logger(__file__)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)

    for index in indices:

        logger.info('=============================== running on index={}'.format(index))

        target_field_names = args.target_field_names.split(',')

        root_target_field = os.path.join(args.root_path, target_field_names[index])

        sci_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field,
                                                              '*_{}'.format(args.sci_images_suffix)))]

        wavebands_list = args.wavebands_list.split(',')

        ordered_wavebands = {b: i for i, b in enumerate(wavebands_list)}

        sci_images_list = sorted(sci_images_list, key=lambda x: ordered_wavebands[x.split('_')[-2]])

        sci_images_encoded = [name.encode('utf8') for name in sci_images_list]

        rms_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field,
                                                              '*_{}'.format(args.rms_images_suffix)))]

        if rms_images_list:
            rms_images_list = sorted(rms_images_list, key=lambda x: ordered_wavebands[x.split('_')[-2]])
            rms_images_encoded = [name.encode('utf8') for name in rms_images_list]
        else:
            rms_images_encoded = list(np.full(len(sci_images_list), 'None'.encode('utf8')))

        exp_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field,
                                                              '*_{}'.format(args.exp_images_suffix)))]

        if exp_images_list:
            exp_images_list = sorted(exp_images_list, key=lambda x: ordered_wavebands[x.split('_')[-2]])
            exp_images_encoded = [name.encode('utf8') for name in exp_images_list]
        else:
            exp_images_encoded = list(np.full(len(sci_images_list), 'None'.encode('utf8')))

        images_to_compress = sci_images_list + rms_images_list + exp_images_list

        compress_list_of_files(root_target_field, '{}_index{:06d}.tar'.format(args.image_archive_prefix, index),
                               root_target_field, images_to_compress)

        target_field_name_encoded = target_field_names[index].encode('utf8')
        wavebands_encoded = [name.encode('utf8') for name in wavebands_list]
        telescope_name_encoded = args.telescope_name.encode('utf8')

        photo_cmd = ['-DETECT_MINAREA'.encode('utf8'), str(args.detect_minarea).encode('utf8'),
                     '-DETECT_THRESH'.encode('utf8'), str(args.detect_thresh).encode('utf8'),
                     '-ANALYSIS_THRESH'.encode('utf8'), str(args.analysis_thresh).encode('utf8'),
                     '-DEBLEND_NTHRESH'.encode('utf8'), str(args.deblend_nthresh).encode('utf8'),
                     '-DEBLEND_MINCONT'.encode('utf8'), str(args.deblend_mincont).encode('utf8'),
                     '-PHOT_APERTURES'.encode('utf8'), args.phot_apertures.encode('utf8'),
                     '-PHOT_AUTOPARAMS'.encode('utf8'), args.phot_autoparams.encode('utf8'),
                     '-PHOT_PETROPARAMS'.encode('utf8'), args.phot_petroparams.encode('utf8'),
                     '-PHOT_AUTOAPERS'.encode('utf8'), args.phot_autoapers.encode('utf8'),
                     '-PHOT_FLUXFRAC'.encode('utf8'), str(args.phot_fluxfrac).encode('utf8'),
                     '-BACK_SIZE'.encode('utf8'), str(args.back_size).encode('utf8'),
                     '-BACK_FILTERSIZE'.encode('utf8'), str(args.back_filtersize).encode('utf8'),
                     '-BACKPHOTO_THICK'.encode('utf8'), str(args.backphoto_thick).encode('utf8')]

        initial_guesses = args.psf_fwhm_init_guesses.split(',')
        psf_fwhm_init_guesses = np.full(len(wavebands_list), [float(name) for name in initial_guesses])
        e_b_v = float(args.e_b_v.split(',')[index])

        sextractor_binary = args.sextractor_binary_filename.encode('utf8')
        sextractor_config = args.sextractor_config_filename.encode('utf8')
        sextractor_params = args.sextractor_params_filename.encode('utf8')
        sextractor_filter = args.sextractor_filter_filename.encode('utf8')
        sextractor_nnw = args.sextractor_nnw_filename.encode('utf8')
        checkimages = args.sextractor_checkimages.split(',')
        sextractor_checkimages = [name.encode('utf8') for name in checkimages]
        checkimages_endings = args.sextractor_checkimages_endings.split(',')
        sextractor_checkimages_endings = [name.encode('utf8') for name in checkimages_endings]

        external_star_catalogue = '{}_{}'.format(target_field_names[index], args.ext_star_cat_suffix).encode('utf8')

        res_files = []
        res_files.extend([sextractor_config.decode('utf8'), sextractor_params.decode('utf8'),
                          sextractor_filter.decode('utf8'), sextractor_nnw.decode('utf8')])

        compress_list_of_files(root_target_field, '{}_index{:06d}.tar'.format(args.resources_archive_prefix, index),
                               args.sextractor_resources_path, res_files)
        add_to_compressed_list_of_files(root_target_field, '{}_index{:06d}.tar'
                                        .format(args.resources_archive_prefix, index), args.star_catalogues_path,
                                        external_star_catalogue.decode('utf8'))

        if not os.path.isdir(args.h5pytable_folder):
            os.makedirs(args.h5pytable_folder, exist_ok=True)

        with h5py.File(os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'
                       .format(args.h5pytable_prefix, index)), mode='w') as h5table:
            h5table.create_dataset(name='telescope_name', data=telescope_name_encoded)
            h5table.create_dataset(name='target_field_name', data=target_field_name_encoded)
            h5table.create_dataset(name='sci_images', data=sci_images_encoded)
            h5table.create_dataset(name='rms_images', data=rms_images_encoded)
            h5table.create_dataset(name='exp_images', data=exp_images_encoded)
            h5table.create_dataset(name='wavebands', data=wavebands_encoded)
            h5table.create_dataset(name='pixel_scale', data=args.pixel_scale)
            h5table.create_dataset(name='photo_cmd', data=photo_cmd)
            h5table.create_dataset(name='psf_fwhm_init_guesses', data=psf_fwhm_init_guesses)
            h5table.create_dataset(name='E(B-V)', data=e_b_v)
            h5table.create_dataset(name='sextractor_binary', data=sextractor_binary)
            h5table.create_dataset(name='sextractor_config', data=sextractor_config)
            h5table.create_dataset(name='sextractor_params', data=sextractor_params)
            h5table.create_dataset(name='sextractor_filter', data=sextractor_filter)
            h5table.create_dataset(name='sextractor_nnw', data=sextractor_nnw)
            h5table.create_dataset(name='sextractor_checkimages', data=sextractor_checkimages)
            h5table.create_dataset(name='sextractor_checkimages_endings', data=sextractor_checkimages_endings)
            h5table.create_dataset(name='external_star_catalogue', data=external_star_catalogue)

        yield index


def check_missing(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)
    list_missing = []

    for index in indices:

        output_table = os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'
                                    .format(args.h5pytable_prefix, index))

        if os.path.exists(output_table):
            current_is_missing = False
        else:
            current_is_missing = True

        if current_is_missing:
            list_missing.append(index)
            logger.info('%d table missing' % index)
        else:
            logger.debug('%d table OK' % index)

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

    description = "Create parameters table to run SExtractor on Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--root_path', type=str, action='store', default=cwd,
                        help='root files path')
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_prefix', type=str, action='store', default='sextractor_run_table',
                        help='h5py table prefix')
    parser.add_argument('--telescope_name', type=str, action='store', default='HST',
                        help='survey telescope name')
    parser.add_argument('--target_field_names', type=str, action='store', default='abells1063',
                        help='list of comma-separated target field names')
    parser.add_argument('--wavebands_list', type=str, action='store', default='f814w',
                        help='list of comma-separated wavebands')
    parser.add_argument('--sci_images_suffix', type=str, action='store', default='drz.fits',
                        help='filename suffix of scientific images')
    parser.add_argument('--rms_images_suffix', type=str, action='store', default='rms.fits',
                        help='filename suffix of rms images')
    parser.add_argument('--exp_images_suffix', type=str, action='store', default='exp.fits',
                        help='filename suffix of exposure time images')
    parser.add_argument('--ext_star_cat_suffix', type=str, action='store', default='star_positions_f160w.fits',
                        help='filename suffix of external star catalogue')
    parser.add_argument('--image_archive_prefix', type=str, action='store', default='images',
                        help='filename prefix of tar file containing images')
    parser.add_argument('--resources_archive_prefix', type=str, action='store', default='res_sextractor_files',
                        help='filename prefix of tar file containing res files')
    parser.add_argument('--star_catalogues_path', type=str, action='store',
                        default=resource_filename(morphofit.__name__, 'res/star_catalogues'),
                        help='path to star catalogues folder')
    parser.add_argument('--pixel_scale', type=float, action='store', default=0.060,
                        help='image pixel scale')
    parser.add_argument('--psf_fwhm_init_guesses', type=str, action='store', default='0.1',
                        help='list of seeing initial guesses for all wavebands')
    parser.add_argument('--e_b_v', type=str, action='store', default='0.0',
                        help='list of E(B-V) extinctions')
    parser.add_argument('--detect_minarea', type=float, action='store', default=10,
                        help='SExtractor DETECT_MINAREA parameter')
    parser.add_argument('--detect_thresh', type=float, action='store', default=1.0,
                        help='SExtractor DETECT_THRESH parameter')
    parser.add_argument('--analysis_thresh', type=float, action='store', default=1.5,
                        help='SExtractor ANALYSIS_THRESH parameter')
    parser.add_argument('--deblend_nthresh', type=float, action='store', default=64,
                        help='SExtractor DEBLEND_NTHRESH parameter')
    parser.add_argument('--deblend_mincont', type=float, action='store', default=0.0001,
                        help='SExtractor DEBLEND_MINCONT parameter')
    parser.add_argument('--phot_apertures', type=str, action='store', default='5,10,15,20,25',
                        help='SExtractor PHOT_APERTURES parameter')
    parser.add_argument('--phot_autoparams', type=str, action='store', default='2.5,3.5',
                        help='SExtractor PHOT_AUTOPARAMS parameter')
    parser.add_argument('--phot_petroparams', type=str, action='store', default='2.0,3.5',
                        help='SExtractor PHOT_PETROPARAMS parameter')
    parser.add_argument('--phot_autoapers', type=str, action='store', default='0.0,0.0',
                        help='SExtractor PHOT_AUTOAPERS parameter')
    parser.add_argument('--phot_fluxfrac', type=float, action='store', default=0.5,
                        help='SExtractor PHOT_FLUXFRAC parameter')
    parser.add_argument('--back_size', type=float, action='store', default=64,
                        help='SExtractor BACK_SIZE parameter')
    parser.add_argument('--back_filtersize', type=float, action='store', default=3,
                        help='SExtractor BACK_FILTERSIZE parameter')
    parser.add_argument('--backphoto_thick', type=float, action='store', default=24,
                        help='SExtractor BACKPHOTO_THICK parameter')
    parser.add_argument('--sextractor_binary_filename', type=str, action='store', default='/usr/local/bin/sex',
                        help='SExtractor binary filename')
    parser.add_argument('--sextractor_config_filename', type=str, action='store', default='default.sex',
                        help='SExtractor config filename')
    parser.add_argument('--sextractor_params_filename', type=str, action='store', default='default.param',
                        help='SExtractor params filename')
    parser.add_argument('--sextractor_filter_filename', type=str, action='store', default='gauss_3.0_5x5.conv',
                        help='SExtractor filter filename')
    parser.add_argument('--sextractor_nnw_filename', type=str, action='store', default='default.nnw',
                        help='SExtractor nnw filename')
    parser.add_argument('--sextractor_checkimages', type=str, action='store', default='SEGMENTATION',
                        help='SExtractor checkimages list, only values allowed are SEGMENTATION,APERTURES, '
                             'SEGMENTATION must be always present as first')
    parser.add_argument('--sextractor_checkimages_endings', type=str, action='store', default='seg',
                        help='SExtractor checkimages suffix list, only values allowed are seg,ap. seg must'
                             'be always present as first')
    parser.add_argument('--sextractor_resources_path', type=str, action='store',
                        default=resource_filename(morphofit.__name__, 'res/sextractor'),
                        help='path to SExtractor resources folder')
    args = parser.parse_args(args)

    return args
