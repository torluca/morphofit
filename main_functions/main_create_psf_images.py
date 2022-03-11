#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
import h5py
import argparse

# morphofit imports
from morphofit.psf_estimation import create_moffat_psf_image, create_observed_psf_image, create_pca_psf_image
from morphofit.utils import get_logger

logger = get_logger(__file__)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    filename_h5pytable = setup(args)
    h5table = h5py.File(filename_h5pytable, 'r')

    for index in indices:
        logger.info('=============================== running on index={}'.format(index))

        root_target = h5table['root_targets'].value[index].decode('utf8')
        sci_images = [[name.decode('utf8') for name in image] for image in h5table['sci_images'].value]
        seg_images = [[name.decode('utf8') for name in image] for image in h5table['seg_images'].value]
        target_star_positions = [[name.decode('utf8') for name in table]
                                 for table in h5table['target_star_positions'].value]
        target_param_tables = [[name.decode('utf8') for name in table]
                               for table in h5table['target_param_tables'].value]

        psf_image_size = h5table['psf_image_size'].value
        pixel_scale = h5table['pixel_scale'].value
        wavebands = [band.decode('utf8') for band in h5table['wavebands'].value[index]]

        os.makedirs(root_target + 'stars', exist_ok=True)

        logger.info('=============================== Creating Moffat PSF images')
        create_moffat_psf_image(root_target, target_star_positions, sci_images, seg_images,
                                psf_image_size, wavebands, pixel_scale, target_param_tables)

        logger.info('=============================== Creating Observed PSF images')
        create_observed_psf_image(root_target, target_star_positions, sci_images, seg_images,
                                  psf_image_size, wavebands)

        logger.info('=============================== Creating PCA PSF images')
        create_pca_psf_image(root_target, target_star_positions, sci_images, seg_images,
                             psf_image_size, wavebands)

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

        wavebands = [band.decode('utf8') for band in h5table['wavebands'].value[index]]
        root_target = h5table['root_targets'].value[index].decode('utf8')

        for band in wavebands:
            obs_psf_path = root_target + 'stars/observed_psf_{}.fits'.format(band)
            moffat_psf_path = root_target + 'stars/moffat_psf_{}.fits'.format(band)

            if not os.path.isfile(obs_psf_path):
                logger.error('error opening psf image')
                current_is_missing = True
            if not os.path.isfile(moffat_psf_path):
                logger.error('error opening psf image')
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

    description = "Create PSF images"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--filename_h5pytable', type=str, action='store', default='table.h5',
                        help='h5py table of the file to run on')
    args = parser.parse_args(args)

    return args.filename_h5pytable
