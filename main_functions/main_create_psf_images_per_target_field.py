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
import argparse
from astropy.table import Table
import subprocess

# morphofit imports
from morphofit.psf_estimation import create_moffat_psf_image_per_target
from morphofit.psf_estimation import create_observed_psf_image_per_target
from morphofit.psf_estimation import create_pca_psf_image_per_target
from morphofit.psf_estimation import create_effective_psf_image_per_target
from morphofit.utils import get_logger, uncompress_files, save_psf_output_files

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

        h5pytable_filename = os.path.join(args.h5pytable_folder,
                                          '{}_index{:06d}.h5'.format(args.h5pytable_prefix, index))

        subprocess.run(['cp', h5pytable_filename, temp_dir])

        h5pytable_filename = os.path.join(temp_dir, '{}_index{:06d}.h5'.format(args.h5pytable_prefix, index))

        h5table = h5py.File(h5pytable_filename, 'r')

        target_field_name = h5table['target_field_name'][()].decode('utf8')
        root_target_field = h5table['root_target_field'][()].decode('utf8')

        subprocess.run(['cp', os.path.join(root_target_field,
                                           '{}_index{:06d}.tar'.format(args.image_archive_prefix, index)), temp_dir])
        subprocess.run(['cp', os.path.join(root_target_field,
                                           '{}_index{:06d}.tar'.format(args.resources_archive_prefix,
                                                                       index)), temp_dir])
        uncompress_files(temp_dir, temp_dir, '{}_index{:06d}.tar'.format(args.image_archive_prefix, index))
        uncompress_files(temp_dir, temp_dir, '{}_index{:06d}.tar'.format(args.resources_archive_prefix, index))

        sci_images = [os.path.join(temp_dir, name.decode('utf8')) for name in h5table['sci_images'][()]]
        target_field_star_positions = [os.path.join(temp_dir, name.decode('utf8'))
                                       for name in h5table['target_field_star_positions'][()]]

        psf_image_size = h5table['psf_image_size'][()]
        pixel_scale = h5table['pixel_scale'][()]
        wavebands = [band.decode('utf8') for band in h5table['wavebands'][()]]

        target_field_param_table_filename = os.path.join(temp_dir,
                                                         h5table['target_field_param_table'][()].decode('utf8'))

        target_param_table = Table.read(target_field_param_table_filename, format='fits')

        psf_methods = args.psf_methods.split(',')
        moffat_psf_flag, observed_psf_flag, pca_psf_flag, effective_psf_flag = False, False, False, False

        if 'moffat' in psf_methods:
            logger.info('=============================== Creating Moffat PSF images')
            try:
                create_moffat_psf_image_per_target(temp_dir, target_field_name, target_field_star_positions,
                                                   args.star_catalogue_x_keyword, args.star_catalogue_y_keyword,
                                                   sci_images, psf_image_size, wavebands,
                                                   pixel_scale, target_param_table)
                moffat_psf_flag = True
            except Exception as e:
                logger.info(e)

        if 'observed' in psf_methods:
            logger.info('=============================== Creating Observed PSF images')
            try:
                create_observed_psf_image_per_target(temp_dir, target_field_name, target_field_star_positions,
                                                     args.star_catalogue_x_keyword, args.star_catalogue_y_keyword,
                                                     sci_images, psf_image_size, wavebands)
                observed_psf_flag = True
            except Exception as e:
                logger.info(e)

        if 'pca' in psf_methods:
            logger.info('=============================== Creating PCA PSF images')
            try:
                create_pca_psf_image_per_target(temp_dir, target_field_name, target_field_star_positions,
                                                args.star_catalogue_x_keyword, args.star_catalogue_y_keyword,
                                                sci_images, psf_image_size, wavebands)
                pca_psf_flag = True
            except Exception as e:
                logger.info(e)

        if 'effective' in psf_methods:
            logger.info('=============================== Creating Effective PSF images')
            try:
                create_effective_psf_image_per_target(temp_dir, target_field_name, target_field_star_positions,
                                                      args.star_catalogue_x_keyword, args.star_catalogue_y_keyword,
                                                      sci_images, psf_image_size, wavebands)
                effective_psf_flag = True
            except Exception as e:
                logger.info(e)

        h5table.close()

        output_directory = os.path.join(root_target_field, 'stars')
        os.makedirs(output_directory, exist_ok=False)

        save_psf_output_files(temp_dir, output_directory, wavebands, target_field_name, moffat_psf_flag=moffat_psf_flag,
                              observed_psf_flag=observed_psf_flag, pca_psf_flag=pca_psf_flag,
                              effective_psf_flag=effective_psf_flag)

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

        current_is_missing = False

        h5pytable_filename = os.path.join(args.h5pytable_folder,
                                          '{}_index{:06d}.h5'.format(args.h5pytable_prefix, index))
        h5table = h5py.File(h5pytable_filename, 'r')

        target_field_name = h5table['target_field_name'][()].decode('utf8')
        root_target_field = h5table['root_target_field'][()].decode('utf8')
        wavebands = [band.decode('utf8') for band in h5table['wavebands'][()]]

        psf_methods = args.psf_methods.split(',')

        if 'moffat' in psf_methods:
            for waveband in wavebands:
                moffat_psf_path = os.path.join(root_target_field, 'stars/moffat_psf_{}_{}.fits'
                                               .format(target_field_name, waveband))
                if not os.path.isfile(moffat_psf_path):
                    logger.error('error opening psf image')
                    current_is_missing = True

        if 'observed' in psf_methods:
            for waveband in wavebands:
                observed_psf_path = os.path.join(root_target_field, 'stars/observed_psf_{}_{}.fits'
                                                 .format(target_field_name, waveband))
                if not os.path.isfile(observed_psf_path):
                    logger.error('error opening psf image')
                    current_is_missing = True

        if 'pca' in psf_methods:
            for waveband in wavebands:
                pca_psf_path = os.path.join(root_target_field, 'stars/pca_psf_{}_{}.fits'
                                            .format(target_field_name, waveband))
                if not os.path.isfile(pca_psf_path):
                    logger.error('error opening psf image')
                    current_is_missing = True

        if 'effective' in psf_methods:
            for waveband in wavebands:
                effective_psf_path = os.path.join(root_target_field, 'stars/effective_psf_{}_{}.fits'
                                                  .format(target_field_name, waveband))
                if not os.path.isfile(effective_psf_path):
                    logger.error('error opening psf image')
                    current_is_missing = True

        if current_is_missing:
            list_missing.append(index)
            logger.info('%d psf missing' % index)
        else:
            logger.debug('%d psf all OK' % index)

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

    description = "Create PSF images per target field"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_prefix', type=str, action='store', default='table_psf_creation',
                        help='h5py table prefix')
    parser.add_argument('--temp_dir_path', type=str, action='store', default=cwd,
                        help='temporary folder where to make calculations locally, used only if --local_or_cluster'
                             ' is set to local')
    parser.add_argument('--image_archive_prefix', type=str, action='store', default='images',
                        help='filename prefix of tar file containing images')
    parser.add_argument('--resources_archive_prefix', type=str, action='store', default='res_psf_files',
                        help='filename prefix of tar file containing res files')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')
    parser.add_argument('--psf_methods', type=str, action='store', default='observed',
                        help='list of psf types, allowed values are moffat,observed,pca,effective')
    parser.add_argument('--star_catalogue_x_keyword', type=str, action='store', default='x',
                        help='Star catalogue x keyword')
    parser.add_argument('--star_catalogue_y_keyword', type=str, action='store', default='y',
                        help='Star catalogue y keyword')
    args = parser.parse_args(args)

    return args
