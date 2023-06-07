#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import argparse
import h5py
import os
import glob
import numpy as np

# morphofit imports
from morphofit.utils import get_logger, compress_list_of_files

logger = get_logger(__file__)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)

    for index in indices:

        target_field_names = args.target_field_names.split(',')

        root_target_field = os.path.join(args.root_path, target_field_names[index])

        sci_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field, '*_{}'
                                                              .format(args.sci_images_suffix)))]

        wavebands_list = args.wavebands_list.split(',')

        ordered_wavebands = {b: i for i, b in enumerate(wavebands_list)}

        sci_images_list = sorted(sci_images_list, key=lambda x: ordered_wavebands[x.split('_')[1]])

        sci_images_encoded = [name.encode('utf8') for name in sci_images_list]

        rms_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field, '*_{}'
                                                              .format(args.rms_images_suffix)))]

        if rms_images_list:
            rms_images_list = sorted(rms_images_list, key=lambda x: ordered_wavebands[x.split('_')[1]])
            rms_images_encoded = [name.encode('utf8') for name in rms_images_list]
        else:
            rms_images_encoded = list(np.full(len(sci_images_list), 'None'.encode('utf8')))

        exp_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field, '*_{}'
                                                              .format(args.exp_images_suffix)))]

        if exp_images_list:
            exp_images_list = sorted(exp_images_list, key=lambda x: ordered_wavebands[x.split('_')[1]])
            exp_images_encoded = [name.encode('utf8') for name in exp_images_list]
        else:
            exp_images_encoded = list(np.full(len(sci_images_list), 'None'.encode('utf8')))

        seg_images_list = [os.path.basename(name)
                           for name in glob.glob(os.path.join(root_target_field, '*{}'
                                                              .format(args.seg_images_suffix)))]
        seg_images_list = sorted(seg_images_list, key=lambda x: ordered_wavebands[x.split('_')[1]])
        seg_images_encoded = [name.encode('utf8') for name in seg_images_list]

        images_to_compress = sci_images_list + rms_images_list + exp_images_list + seg_images_list

        compress_list_of_files(root_target_field, '{}_index{:06d}.tar'.format(args.image_archive_prefix, index),
                               root_target_field, images_to_compress)

        with h5py.File(os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'.format(args.h5pytable_prefix,
                                                                                      index)), mode='w') as h5table:
            h5table.create_dataset(name='root_target_field', data=root_target_field.encode('utf8'))
            h5table.create_dataset(name='sci_image_filenames', data=sci_images_encoded)
            h5table.create_dataset(name='rms_image_filenames', data=rms_images_encoded)
            h5table.create_dataset(name='seg_image_filenames', data=seg_images_encoded)
            h5table.create_dataset(name='exp_image_filenames', data=exp_images_encoded)
            h5table.create_dataset(name='wavebands', data=[name.encode('utf8') for name in wavebands_list])
            h5table.create_dataset(name='region_image_suffix', data=args.region_image_suffix.encode('utf8'))
            h5table.create_dataset(name='number_of_regions_per_side', data=args.number_of_regions_per_side)

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

    description = "Create parameters table to cut regions of the full image"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--root_path', type=str, action='store', default=cwd,
                        help='root files path')
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_prefix', type=str, action='store', default='table_region_creation',
                        help='h5py table prefix')
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
    parser.add_argument('--seg_images_suffix', type=str, action='store', default='forced_seg.fits',
                        help='filename suffix of segmentation images')
    parser.add_argument('--image_archive_prefix', type=str, action='store', default='images',
                        help='filename prefix of tar file containing images')
    parser.add_argument('--number_of_regions_per_side', type=int, action='store', default=3,
                        help='Number of regions per image side to create out of the full image')
    parser.add_argument('--region_image_suffix', type=str, action='store', default='reg',
                        help='Filename suffix of the created region image')

    args = parser.parse_args(args)

    return args
