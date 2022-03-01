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
import argparse
import morphofit
from pkg_resources import resource_filename

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

        target_field_name_encoded = target_field_names[index].encode('utf8')
        target_field_param_table = '{}_{}'.format(target_field_names[index],
                                                  args.parameters_table_suffix).encode('utf8')

        root_target_field = os.path.join(args.root_path, target_field_names[index])

        sci_images_list = [os.path.basename(name) for name in glob.glob(os.path.join(root_target_field, '*_{}'
                                                                                     .format(args.sci_images_suffix)))]

        wavebands_list = args.wavebands_list.split(',')
        wavebands_encoded = [name.encode('utf8') for name in wavebands_list]

        ordered_wavebands = {b: i for i, b in enumerate(wavebands_list)}

        sci_images_list = sorted(sci_images_list, key=lambda x: ordered_wavebands[x.split('_')[-2]])

        sci_images_encoded = [name.encode('utf8') for name in sci_images_list]

        compress_list_of_files(root_target_field, '{}_index{:06d}.tar'.format(args.image_archive_prefix, index),
                               root_target_field, sci_images_list)

        target_field_star_positions = ['{}_{}_{}'.format(target_field_names[index], waveband, args.ext_star_cat_suffix)
                                       for waveband in wavebands_list]
        target_field_star_positions_encoded = ['{}_{}_{}'.format(target_field_names[index], waveband,
                                                                 args.ext_star_cat_suffix).encode('utf8')
                                               for waveband in wavebands_list]

        compress_list_of_files(root_target_field, '{}_index{:06d}.tar'.format(args.resources_archive_prefix, index),
                               args.star_catalogues_path, target_field_star_positions)

        add_to_compressed_list_of_files(root_target_field, '{}_index{:06d}.tar'
                                        .format(args.resources_archive_prefix, index),
                                        root_target_field, target_field_param_table.decode('utf8'))

        with h5py.File(os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'
                       .format(args.h5pytable_prefix, index)), mode='w') as h5table:
            h5table.create_dataset(name='target_field_name', data=target_field_name_encoded)
            h5table.create_dataset(name='root_target_field', data=root_target_field.encode('utf8'))
            h5table.create_dataset(name='sci_images', data=sci_images_encoded)
            h5table.create_dataset(name='wavebands', data=wavebands_encoded)
            h5table.create_dataset(name='target_field_star_positions', data=target_field_star_positions_encoded)
            h5table.create_dataset(name='psf_image_size', data=args.psf_image_size)
            h5table.create_dataset(name='target_field_param_table', data=target_field_param_table)
            h5table.create_dataset(name='pixel_scale', data=args.pixel_scale)

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

    description = "Create parameters table for PSF images creation"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--root_path', type=str, action='store', default='table.h5',
                        help='root files path')
    parser.add_argument('--h5pytable_folder', type=str, action='store',
                        help='h5py table folder')
    parser.add_argument('--h5pytable_prefix', type=str, action='store', default='table_psf_creation',
                        help='h5py table prefix')
    parser.add_argument('--wavebands_list', type=str, action='store', default=['f814w'],
                        help='list of wavebands')
    parser.add_argument('--target_field_names', type=str, action='store', default='abells1063',
                        help='list of comma-separated target field names')
    parser.add_argument('--parameters_table_suffix', type=str, action='store', default='param_table.fits',
                        help='Parameters table suffix')
    parser.add_argument('--sci_images_suffix', type=str, action='store', default='drz.fits',
                        help='filename suffix of scientific images')
    parser.add_argument('--image_archive_prefix', type=str, action='store', default='images',
                        help='filename prefix of tar file containing images')
    parser.add_argument('--resources_archive_prefix', type=str, action='store', default='res_psf_files',
                        help='filename prefix of tar file containing res files')
    parser.add_argument('--ext_star_cat_suffix', type=str, action='store', default='star_positions.fits',
                        help='filename suffix of external star catalogue')
    parser.add_argument('--star_catalogues_path', type=str, action='store',
                        default=resource_filename(morphofit.__name__, 'res/star_catalogues'),
                        help='path to star catalogues folder')
    parser.add_argument('--psf_image_size', type=int, action='store', default=100,
                        help='images pixel scale')
    parser.add_argument('--pixel_scale', type=float, action='store', default=0.060,
                        help='images pixel scale')
    args = parser.parse_args(args)

    return args
