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
import numpy as np
import argparse
from astropy.table import Table, unique
import itertools
import glob

# morphofit imports
from morphofit.utils import get_logger

logger = get_logger(__file__)


def select_images(path_to_target_field, waveband, image_type):
    """

    :param path_to_target_field:
    :param waveband:
    :param image_type:
    :return:
    """

    images = glob.glob(os.path.join(path_to_target_field, '*{}*{}'.format(waveband, image_type)))
    images = os.path.basename(images[0])

    return images


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)

    root_target_fields, target_field_names, psf_image_types, sigma_image_types, background_estimate_methods, wavebands,\
        stamp_indices, sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames, \
        exposure_times, magnitude_zeropoints, effective_gains, instrumental_gains, \
        background_values, target_galaxies_catalogues, source_galaxies_catalogues, target_galaxy_ids,\
        input_galfit_filenames, sigma_image_filenames, output_model_image_filenames, \
        psf_image_filenames, psf_sampling_factors, constraints_file_filenames, telescope_names = \
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
        [], [], [], [], [], [], [], [], []

    target_field_names_list = args.target_field_names.split(',')

    for target_field_name in target_field_names_list:

        root_target_field = os.path.join(args.root_path, target_field_name)

        parameters_table = Table.read(os.path.join(root_target_field, '{}_{}'.format(target_field_name,
                                                                                     args.parameters_table_suffix)))

        target_galaxies_catalogue = Table.read(os.path.join(root_target_field, '{}_{}_{}'
                                                            .format(args.telescope_name, target_field_name,
                                                                    args.target_galaxies_catalogue_suffix)))

        n_target_galaxies = len(np.unique(target_galaxies_catalogue[args.target_galaxies_id_key]))
        indices_stamps = np.arange(0, n_target_galaxies, 1, dtype=int)
        indices_stamps = ["{:02d}".format(item) for item in indices_stamps]
        target_galaxy_ids_list = unique(target_galaxies_catalogue,
                                        keys=args.target_galaxies_id_key)[args.target_galaxies_id_key]

        psf_image_types_list = args.psf_image_types_list.split(',')
        sigma_image_types_list = args.sigma_image_types_list.split(',')
        background_estimate_methods_list = args.background_estimate_methods_list.split(',')
        wavebands_list = args.wavebands_list.split(',')

        combinations = [['{}'.format(x), '{}'.format(y), '{}'.format(z), '{}'.format(m), '{}'.format(n)]
                        for x, y, z, m, n in itertools.product(psf_image_types_list, sigma_image_types_list,
                                                               background_estimate_methods_list,
                                                               wavebands_list, indices_stamps)]

        for j in range(len(combinations)):
            telescope_names.append(args.telescope_name.encode('utf8'))
            root_target_fields.append(root_target_field.encode('utf8'))
            target_field_names.append(target_field_name.encode('utf8'))
            psf_image_types.append(combinations[j][0].encode('utf8'))
            sigma_image_types.append(combinations[j][1].encode('utf8'))
            background_estimate_methods.append(combinations[j][2].encode('utf8'))
            wavebands.append(combinations[j][3].encode('utf8'))
            stamp_indices.append(combinations[j][4].encode('utf8'))

            sci_image_filenames.append(select_images(root_target_field, combinations[j][3],
                                                     args.sci_images_suffix).encode('utf8'))
            seg_image_filenames.append(select_images(root_target_field, combinations[j][3],
                                                     args.seg_images_suffix).encode('utf8'))
            try:
                rms_image_filenames.append(select_images(root_target_field, combinations[j][3],
                                                         args.rms_images_suffix).encode('utf8'))
            except Exception as e:
                print(e)
                rms_image_filenames.append('None'.encode('utf8'))

            try:
                exp_image_filenames.append(select_images(root_target_field, combinations[j][3],
                                                         args.exp_images_suffix).encode('utf8'))
            except Exception as e:
                print(e)
                exp_image_filenames.append('None'.encode('utf8'))

            w = np.where(parameters_table['wavebands'] == combinations[j][3])
            exposure_times.append(parameters_table[w]['exptimes'][0])
            magnitude_zeropoints.append(parameters_table[w]['zeropoints'][0])
            effective_gains.append(parameters_table[w]['effective_gains'][0])
            instrumental_gains.append(parameters_table[w]['instrumental_gains'][0])
            background_values.append(parameters_table[w]['bkg_amps'][0])
            target_galaxies_catalogues.append('{}_{}_{}'.format(args.telescope_name, target_field_name,
                                                                args.target_galaxies_catalogue_suffix).encode('utf8'))
            source_galaxies_catalogues.append('{}_{}_{}'.format(args.telescope_name, target_field_name,
                                                                args.source_galaxies_catalogue_suffix).encode('utf8'))
            target_galaxy_ids.append(target_galaxy_ids_list[int(combinations[j][4])])

            input_galfit_filenames.append('{}_{}_{}_{}_{}_{}_{}.INPUT'.format(args.telescope_name,
                                                                              target_field_name,
                                                                              combinations[j][3], combinations[j][4],
                                                                              combinations[j][0], combinations[j][1],
                                                                              combinations[j][2]).encode('utf8'))

            if combinations[j][1] == 'custom_sigma_image':
                sigma_image_filenames.append('{}_{}_{}_stamp{}_sigma_image.fits'.format(args.telescope_name,
                                                                                        target_field_name,
                                                                                        combinations[j][3],
                                                                                        combinations[j][4]).encode(
                    'utf8'))
            else:
                sigma_image_filenames.append('None'.encode('utf8'))

            output_model_image_filenames.append('{}_{}_{}_stamp{}_{}_{}_{}_imgblock.fits'
                                                .format(args.telescope_name,
                                                        target_field_name,
                                                        combinations[j][3],
                                                        combinations[j][4],
                                                        combinations[j][0],
                                                        combinations[j][1],
                                                        combinations[j][2])
                                                .encode('utf8'))

            psf_image_filenames.append('{}_{}_{}.fits'.format(combinations[j][0], target_field_name,
                                                              combinations[j][3]).encode('utf8'))

            if combinations[j][0] == 'effective_psf':
                psf_sampling_factors.append(2)
            else:
                psf_sampling_factors.append(1)

            # constraints_file_filenames.append('constraints_file_{}_{}.CONSTRAINTS'.format(target_field_name,
            #                                                                               combinations[j][3]).encode(
            #     'utf8'))
            constraints_file_filenames.append('None'.encode('utf8'))

    logger.info('Number of combinations: {}'.format(len(telescope_names)))

    with h5py.File(os.path.join(args.h5pytable_folder, args.h5pytable_filename), mode='w') as h5table:
        h5table.create_dataset(name='root_target_fields', data=root_target_fields)
        h5table.create_dataset(name='telescope_names', data=telescope_names)
        h5table.create_dataset(name='target_field_names', data=target_field_names)
        h5table.create_dataset(name='wavebands', data=wavebands)
        h5table.create_dataset(name='stamp_indices', data=stamp_indices)
        h5table.create_dataset(name='psf_image_types', data=psf_image_types)
        h5table.create_dataset(name='sigma_image_types', data=sigma_image_types)
        h5table.create_dataset(name='background_estimate_methods', data=background_estimate_methods)
        h5table.create_dataset(name='sci_image_filenames', data=sci_image_filenames)
        h5table.create_dataset(name='rms_image_filenames', data=rms_image_filenames)
        h5table.create_dataset(name='seg_image_filenames', data=seg_image_filenames)
        h5table.create_dataset(name='exp_image_filenames', data=exp_image_filenames)
        h5table.create_dataset(name='exposure_times', data=exposure_times)
        h5table.create_dataset(name='magnitude_zeropoints', data=magnitude_zeropoints)
        h5table.create_dataset(name='effective_gains', data=effective_gains)
        h5table.create_dataset(name='instrumental_gains', data=instrumental_gains)
        h5table.create_dataset(name='background_values', data=background_values)
        h5table.create_dataset(name='target_galaxies_catalogues', data=target_galaxies_catalogues)
        h5table.create_dataset(name='source_galaxies_catalogues', data=source_galaxies_catalogues)
        h5table.create_dataset(name='target_galaxy_ids', data=target_galaxy_ids)
        h5table.create_dataset(name='input_galfit_filenames', data=input_galfit_filenames)
        h5table.create_dataset(name='sigma_image_filenames', data=sigma_image_filenames)
        h5table.create_dataset(name='output_model_image_filenames', data=output_model_image_filenames)
        h5table.create_dataset(name='psf_image_filenames', data=psf_image_filenames)
        h5table.create_dataset(name='psf_sampling_factors', data=psf_sampling_factors)
        h5table.create_dataset(name='constraints_file_filenames', data=constraints_file_filenames)
        h5table.create_dataset(name='enlarging_image_factor', data=args.enlarging_image_factor)
        h5table.create_dataset(name='enlarging_separation_factor', data=args.enlarging_separation_factor)
        h5table.create_dataset(name='pixel_scale', data=args.pixel_scale)
        h5table.create_dataset(name='convolution_box_size', data=args.convolution_box_size)
        h5table.create_dataset(name='galfit_binary_file', data=args.galfit_binary_file.encode('utf8'))

    yield indices[0]


def check_missing(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    list_missing = []

    args = setup(args)

    for index in indices:

        output_table = os.path.join(args.h5pytable_folder, args.h5pytable_filename)

        if os.path.exists(output_table):
            current_is_missing = False
        else:
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

    cwd = os.getcwd()

    description = "Create parameters table to run GALFIT on stamps around target galaxies in Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--root_path', type=str, action='store', default=cwd,
                        help='root files path')
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_filename', type=str, action='store', default='table_galfit_on_stamps_run.h5',
                        help='h5py table filename')
    parser.add_argument('--telescope_name', type=str, action='store', default='HST',
                        help='survey telescope name')
    parser.add_argument('--target_field_names', type=str, action='store', default='abells1063',
                        help='list of comma-separated target field names')
    parser.add_argument('--wavebands_list', type=str, action='store', default='f814w',
                        help='list of comma-separated wavebands')
    parser.add_argument('--psf_image_types_list', type=str, action='store', default='observed_psf',
                        help='list of comma-separated wavebands, allowed values are moffat_psf,observed_psf,pca_psf,'
                             'effective_psf')
    parser.add_argument('--sigma_image_types_list', type=str, action='store', default='custom_sigma_image',
                        help='list of comma-separated wavebands, allowed values are '
                             'custom_sigma_image,internal_generated_sigma_image')
    parser.add_argument('--background_estimate_methods_list', type=str, action='store', default='background_free_fit',
                        help='list of comma-separated wavebands, allowed values are'
                             'background_free_fit,background_fixed_value')
    parser.add_argument('--enlarging_image_factor', type=float, action='store', default=20,
                        help='enlarging factor for the stamp creation around target galaxies')
    parser.add_argument('--enlarging_separation_factor', type=float, action='store', default=10,
                        help='enlarging factor for the target galaxy effective radius to find neighbouring objects')
    parser.add_argument('--pixel_scale', type=float, action='store', default=0.060,
                        help='image pixel scale')
    parser.add_argument('--convolution_box_size', type=int, action='store', default=256,
                        help='psf convolution box size used by GALFIT')
    parser.add_argument('--galfit_binary_file',  type=str, action='store', default='/Users/torluca/galfit',
                        help='path to GALFIT binary file')
    parser.add_argument('--target_galaxies_catalogue_suffix', type=str, action='store',
                        default='multiband_target.forced.cat',
                        help='Multiband target galaxies catalogue suffix')
    parser.add_argument('--target_galaxies_id_key', type=str, action='store', default='NUMBER',
                        help='Catalogue header keyword for target galaxies ID')
    parser.add_argument('--source_galaxies_catalogue_suffix', type=str, action='store',
                        default='multiband_sources.forced.cat',
                        help='Multiband source galaxies catalogue suffix')
    parser.add_argument('--parameters_table_suffix', type=str, action='store', default='param_table.fits',
                        help='Parameters table suffix')
    parser.add_argument('--sci_images_suffix', type=str, action='store', default='drz.fits',
                        help='filename suffix of scientific images')
    parser.add_argument('--rms_images_suffix', type=str, action='store', default='rms.fits',
                        help='filename suffix of rms images')
    parser.add_argument('--exp_images_suffix', type=str, action='store', default='exp.fits',
                        help='filename suffix of exposure time images')
    parser.add_argument('--seg_images_suffix', type=str, action='store', default='seg.fits',
                        help='filename suffix of segmentation images')

    args = parser.parse_args(args)

    return args
