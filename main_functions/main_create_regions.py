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
import subprocess
from astropy.table import Table

# morphofit imports
from morphofit.utils import get_logger, uncompress_files
from morphofit.image_utils import crop_images, create_regions

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

        if args.local_or_cluster == 'cluster':
            temp_dir = os.environ['TMPDIR']
        elif args.local_or_cluster == 'local':
            temp_dir = os.path.join(args.temp_dir_path, 'tmp_index{:06d}'.format(index))
            os.makedirs(temp_dir, exist_ok=False)
        else:
            raise KeyError

        h5pytable_filename = os.path.join(os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'
                                                       .format(args.h5pytable_prefix, index)))

        subprocess.run(['cp', h5pytable_filename, temp_dir])

        h5pytable_filename = os.path.join(temp_dir, h5pytable_filename)

        h5table = h5py.File(h5pytable_filename, 'r')

        root_target_field = h5table['root_target_field'][()].decode('utf8')

        subprocess.run(['cp', os.path.join(root_target_field, '{}_index{:06d}.tar'.format(args.image_archive_prefix,
                                                                                          index)), temp_dir])
        uncompress_files(temp_dir, temp_dir, '{}_index{:06d}.tar'.format(args.image_archive_prefix, index))

        sci_image_filenames = [os.path.join(temp_dir, name.decode('utf8'))
                               for name in h5table['sci_image_filenames'][()]]
        rms_image_filenames = [os.path.join(temp_dir, name.decode('utf8'))
                               for name in h5table['rms_image_filenames'][()]]
        seg_image_filenames = [os.path.join(temp_dir, name.decode('utf8'))
                               for name in h5table['seg_image_filenames'][()]]
        exp_image_filenames = [os.path.join(temp_dir, name.decode('utf8'))
                               for name in h5table['exp_image_filenames'][()]]

        wavebands = [band.decode('utf8') for band in h5table['wavebands'][()]]
        region_image_suffix = h5table['region_image_suffix'][()].decode('utf8')
        number_of_regions_per_side = h5table['number_of_regions_per_side'][()]

        if args.crop_fullimages_first == 'True':
            external_catalogue_filenames = args.external_catalogue_filenames.split(',')
            external_catalogue = Table.read(os.path.join(root_target_field,
                                                         external_catalogue_filenames[index]), format='fits')
            sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames = \
                crop_images(sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames,
                            wavebands, temp_dir, external_catalogue=external_catalogue, crop_suffix=args.crop_suffix,
                            x_keyword=args.x_pixel_keyword, y_keyword=args.y_pixel_keyword)
            subprocess.run(['cp'] + sci_image_filenames + [root_target_field])
            subprocess.run(['cp'] + rms_image_filenames + [root_target_field])
            subprocess.run(['cp'] + seg_image_filenames + [root_target_field])
            subprocess.run(['cp'] + exp_image_filenames + [root_target_field])

        sci_image_region_filenames, seg_image_region_filenames, rms_image_region_filenames, \
            exp_image_region_filenames = create_regions(sci_image_filenames, rms_image_filenames, seg_image_filenames,
                                                        exp_image_filenames, wavebands, region_image_suffix,
                                                        number_of_regions_per_side, temp_dir)

        h5table.close()

        os.makedirs(os.path.join(root_target_field, 'regions'), exist_ok=True)

        subprocess.run(['cp'] + sci_image_region_filenames + [os.path.join(root_target_field, 'regions')])
        subprocess.run(['cp'] + seg_image_region_filenames + [os.path.join(root_target_field, 'regions')])
        subprocess.run(['cp'] + rms_image_region_filenames + [os.path.join(root_target_field, 'regions')])
        subprocess.run(['cp'] + exp_image_region_filenames + [os.path.join(root_target_field, 'regions')])

        if args.local_or_cluster == 'local':
            subprocess.run(['rm', '-rf', temp_dir])

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

        current_is_missing = False

        h5pytable_filename = os.path.join(os.path.join(args.h5pytable_folder, '{}_index{:06d}.h5'
                                                       .format(args.h5pytable_prefix, index)))
        h5table = h5py.File(h5pytable_filename, 'r')
        root_target_field = h5table['root_target_field'][()].decode('utf8')
        output_directory = os.path.join(root_target_field, 'regions')
        sci_image_filenames = [os.path.join(root_target_field, name.decode('utf8'))
                               for name in h5table['sci_image_filenames'][()]]
        number_of_regions_per_side = h5table['number_of_regions_per_side'][()]
        region_image_suffix = h5table['region_image_suffix'][()].decode('utf8')

        for name in sci_image_filenames:
            if args.crop_fullimages_first == 'True':
                name = os.path.splitext(os.path.basename(name))[0] + '_{}'.format(args.crop_suffix)
            for i in range(0, number_of_regions_per_side):
                for j in range(0, number_of_regions_per_side):
                    region_filename = os.path.splitext(os.path.basename(name))[0] + \
                                      '_{}{}{}.fits'.format(region_image_suffix, i, j)
                    if not os.path.isfile(os.path.join(output_directory, region_filename)):
                        logger.error('error opening region image')
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

    description = "Create regions of the full image"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_prefix', type=str, action='store', default='table_region_creation',
                        help='h5py table prefix')
    parser.add_argument('--temp_dir_path', type=str, action='store', default=cwd,
                        help='temporary folder where to make calculations locally, used only if --local_or_cluster'
                             ' is set to local')
    parser.add_argument('--image_archive_prefix', type=str, action='store', default='images',
                        help='filename prefix of tar file containing images')
    parser.add_argument('--crop_fullimages_first', type=str, action='store', default='False',
                        help='Reduce the size of the full image around detected sources from external catalogue'
                             'before creating regions')
    parser.add_argument('--external_catalogue_filenames', type=str, action='store', default='external_catalogue.cat',
                        help='Comma-separated list of external catalogue filenames. To be used with '
                             '--crop_fullimages_first=True')
    parser.add_argument('--crop_suffix', type=str, action='store', default='crop.fits',
                        help='Filename suffix of the cropped image. To be used with --crop_fullimages_first=True')
    parser.add_argument('--x_pixel_keyword', type=str, action='store', default='XWIN_IMAGE_f814w',
                        help='Catalogue header keyword of the x pixel coordinate position of galaxies. To be used'
                             'with -crop_fullimages_first=True')
    parser.add_argument('--y_pixel_keyword', type=str, action='store', default='YWIN_IMAGE_f814w',
                        help='Catalogue header keyword of the y pixel coordinate position of galaxies. To be used'
                             'with -crop_fullimages_first=True')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')
    args = parser.parse_args(args)

    return args
