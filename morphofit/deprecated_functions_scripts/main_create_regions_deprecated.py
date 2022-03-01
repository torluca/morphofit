#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import h5py
import argparse
from astropy.table import Table
import os


# morphofit imports
from morphofit.utils import get_logger
from morphofit.image_utils import crop_images, create_regions

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

        root_files = h5table['root_targets'].value[index].decode('utf8')
        sci_image_filenames = [name.decode('utf8') for name in h5table['sci_image_filenames'].value[index]]
        rms_image_filenames = [name.decode('utf8') for name in h5table['rms_image_filenames'].value[index]]
        seg_image_filenames = [name.decode('utf8') for name in h5table['seg_image_filenames'].value[index]]
        exp_image_filenames = [name.decode('utf8') for name in h5table['exp_image_filenames'].value[index]]
        wavebands = [band.decode('utf8') for band in h5table['wavebands'].value[index]]
        crop_routine = h5table['crop_routine'].value.decode('utf8')
        external_catalogue_filename = h5table['external_catalogue_filenames'].value[index].decode('utf8')
        external_catalogue = Table.read(external_catalogue_filename, format='fits')
        crop_suffix = h5table['crop_suffix'].value.decode('utf8')
        x_keyword = h5table['x_keyword'].value.decode('utf8')
        y_keyword = h5table['y_keyword'].value.decode('utf8')
        number_of_regions_perside = h5table['number_of_regions_perside'].value

        cropped_sci_image_filenames, cropped_rms_image_filenames, cropped_seg_image_filenames, \
            cropped_exp_image_filenames = \
            crop_images(sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames,
                        wavebands, root_files, crop_routine=crop_routine, external_catalogue=external_catalogue,
                        size_range_x=None, size_range_y=None,
                        crop_suffix=crop_suffix, x_keyword=x_keyword, y_keyword=y_keyword)

        output_directory = root_files + 'regions/'
        os.makedirs(output_directory, exist_ok=True)
        create_regions(cropped_sci_image_filenames, cropped_rms_image_filenames, cropped_seg_image_filenames,
                       cropped_exp_image_filenames, wavebands, output_directory, number_of_regions_perside)

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
        root_files = h5table['root_targets'].value[index].decode('utf8')
        output_directory = root_files + 'regions/'
        sci_image_filenames = [name.decode('utf8') for name in h5table['sci_image_filenames'].value[index]]
        crop_suffix = h5table['crop_suffix'].value.decode('utf8')
        number_of_regions = h5table['number_of_regions_perside'].value

        for name in sci_image_filenames:
            crop_filename = os.path.basename(name).split('.fits')[0] + '_{}.fits'.format(crop_suffix)
            if not os.path.isfile(root_files + crop_filename):
                logger.error('error opening cropped image')
                current_is_missing = True
            for i in range(0, number_of_regions):
                for j in range(0, number_of_regions):
                    region_filename = os.path.basename(crop_filename).split('.fits')[0] + '_reg{}{}.fits'.format(i, j)
                    if not os.path.isfile(output_directory + region_filename):
                        logger.error('error opening region image')
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

    description = "Create regions"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--filename_h5pytable', type=str, action='store', default='table.h5',
                        help='h5py table of the file to run on')
    args = parser.parse_args(args)

    return args.filename_h5pytable
