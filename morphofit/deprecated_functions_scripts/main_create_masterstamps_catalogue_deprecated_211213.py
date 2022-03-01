#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import glob
import os
import argparse
import subprocess
import h5py
from astropy.table import Table, Column, vstack
import numpy as np

# morphofit imports
from morphofit.utils import get_logger
from morphofit.catalogue_managing import add_dictionary_item, save_property_dictionaries
from morphofit.catalogue_managing import get_median_properties, create_fixed_image_table
from morphofit.catalogue_managing import match_with_target_galaxies_catalogue

logger = get_logger(__file__)


def create_masterstamps_catalogue(args, root_stamps_target_fields, telescope_name, target_field_name,
                                  waveband_combinations, indices_target_galaxies, stamp_index_combinations,
                                  psf_image_type_combinations, sigma_image_type_combinations,
                                  background_estimate_method_combinations, wavebands, psf_image_types,
                                  sigma_image_types, background_estimate_methods,
                                  target_galaxies_catalogue, source_galaxies_catalogue, temp_dir):
    """

    :param args:
    :param root_stamps_target_fields:
    :param telescope_name:
    :param target_field_name:
    :param waveband_combinations:
    :param indices_target_galaxies:
    :param stamp_index_combinations:
    :param psf_image_type_combinations:
    :param sigma_image_type_combinations:
    :param background_estimate_method_combinations:
    :param wavebands:
    :param psf_image_types:
    :param sigma_image_types:
    :param background_estimate_methods:
    :param target_galaxies_catalogue:
    :param source_galaxies_catalogue:
    :param temp_dir:
    :return:
    """

    x_dictionary = {}
    y_dictionary = {}
    ra_dictionary = {}
    dec_dictionary = {}
    mag_dictionary = {}
    re_dictionary = {}
    n_dictionary = {}
    ar_dictionary = {}
    pa_dictionary = {}
    background_value_dictionary = {}
    background_x_gradient_dictionary = {}
    background_y_gradient_dictionary = {}
    reduced_chisquare_dictionary = {}

    for i in range(len(waveband_combinations)):
        try:
            subprocess.run(['cp', os.path.join(root_stamps_target_fields, '{}{}_{}_{}_{}/{}_{}_{}_{}_{}_{}_{}.h5'
                                               .format('stamp', stamp_index_combinations[i],
                                                       psf_image_type_combinations[i],
                                                       sigma_image_type_combinations[i],
                                                       background_estimate_method_combinations[i], telescope_name,
                                                       target_field_name, waveband_combinations[i],
                                                       stamp_index_combinations[i],
                                                       psf_image_type_combinations[i], sigma_image_type_combinations[i],
                                                       background_estimate_method_combinations[i])), temp_dir])

            best_fit_properties_h5table_filename = os.path.join(temp_dir, '{}_{}_{}_{}_{}_{}_{}.h5'
                                                                .format(telescope_name,
                                                                        target_field_name,
                                                                        waveband_combinations[i],
                                                                        stamp_index_combinations[i],
                                                                        psf_image_type_combinations[i],
                                                                        sigma_image_type_combinations[i],
                                                                        background_estimate_method_combinations[i]))

            add_dictionary_item(best_fit_properties_h5table_filename, target_field_name, waveband_combinations[i],
                                psf_image_type_combinations[i], sigma_image_type_combinations[i],
                                background_estimate_method_combinations[i],
                                x_dictionary, y_dictionary, ra_dictionary, dec_dictionary,
                                mag_dictionary, re_dictionary, n_dictionary, ar_dictionary, pa_dictionary,
                                background_value_dictionary, background_x_gradient_dictionary,
                                background_y_gradient_dictionary, reduced_chisquare_dictionary, kind='stamp',
                                index=stamp_index_combinations[i])

        except OSError as exception:
            logger.info(exception)
            logger.info('Missing {}_{}_{}_{}_{}_{}_{}.h5 table'.format(telescope_name, target_field_name,
                                                                       waveband_combinations[i],
                                                                       stamp_index_combinations[i],
                                                                       psf_image_type_combinations[i],
                                                                       sigma_image_type_combinations[i],
                                                                       background_estimate_method_combinations[i]))
            continue

    save_property_dictionaries(temp_dir, telescope_name, target_field_name,
                               x_dictionary, y_dictionary, ra_dictionary, dec_dictionary,
                               mag_dictionary, re_dictionary, n_dictionary, ar_dictionary, pa_dictionary,
                               background_value_dictionary, background_x_gradient_dictionary,
                               background_y_gradient_dictionary, reduced_chisquare_dictionary)

    subprocess.run(['cp'] + glob.glob(os.path.join(temp_dir, '*_dictionary.pkl')) + [root_stamps_target_fields])

    multiband_tables = []
    for stamp_number in indices_target_galaxies:

        try:
            number_galaxies_stamp = max([len(v) for k, v in mag_dictionary.items() if (target_field_name in k) &
                                         (str(stamp_number) == k.split('_')[2])])
        except ValueError as error:
            logger.info(error)
            continue

        x_positions, x_position_errors, y_positions, y_position_errors, ra, dec, \
            total_magnitudes, total_magnitude_errors, effective_radii, \
            effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios, \
            axis_ratio_errors, position_angles, position_angle_errors, background_values, \
            background_value_errors, background_x_gradients, background_x_gradient_errors, \
            background_y_gradients, background_y_gradient_errors, \
            covariance_effective_radii_position_angles, \
            covariance_effective_radii_position_angles_magnitudes = \
            get_median_properties(target_field_name, wavebands, psf_image_types, background_estimate_methods,
                                  sigma_image_types,
                                  number_galaxies_stamp, x_dictionary, y_dictionary,
                                  ra_dictionary, dec_dictionary, mag_dictionary, re_dictionary, n_dictionary,
                                  ar_dictionary, pa_dictionary, background_value_dictionary,
                                  background_x_gradient_dictionary, background_y_gradient_dictionary,
                                  kind='stamp', index=stamp_number)

        table = create_fixed_image_table(wavebands, x_positions, x_position_errors, y_positions,
                                         y_position_errors, ra, dec, total_magnitudes,
                                         total_magnitude_errors, effective_radii,
                                         effective_radius_errors, sersic_indices, sersic_index_errors,
                                         axis_ratios, axis_ratio_errors, position_angles,
                                         position_angle_errors, background_values,
                                         background_value_errors, background_x_gradients,
                                         background_x_gradient_errors, background_y_gradients,
                                         background_y_gradient_errors,
                                         covariance_effective_radii_position_angles,
                                         covariance_effective_radii_position_angles_magnitudes)

        stamp_number_column = Column(data=np.full(len(table), stamp_number), name='stamp_number')
        table.add_column(stamp_number_column)

        table = match_with_target_galaxies_catalogue(table, target_galaxies_catalogue, source_galaxies_catalogue,
                                                     wavebands, waveband_key=args.waveband_key)

        table.write(os.path.join(temp_dir, '{}_{}_stamp{}.cat'
                    .format(telescope_name, target_field_name, stamp_number)), format='fits', overwrite=True)
        multiband_tables.append(table)

    multiband_table = vstack(multiband_tables, join_type='exact')
    # multiband_table = delete_repeating_sources(multiband_table, wavebands)
    multiband_table.write(os.path.join(temp_dir, '{}_{}_stamps.cat'
                          .format(telescope_name, target_field_name)),
                          format='fits', overwrite=True)

    subprocess.run(['cp'] + glob.glob(os.path.join(temp_dir, '*stamp*.cat')) + [root_stamps_target_fields])


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

        h5pytable_filename = os.path.join(args.h5pytable_folder, args.h5pytable_filename)

        subprocess.run(['cp', h5pytable_filename, temp_dir])

        h5pytable_filename = os.path.join(temp_dir, args.h5pytable_filename)

        h5table = h5py.File(h5pytable_filename, 'r')

        grp = h5table['{}'.format(index)]

        target_galaxies_catalogue_filename = grp['target_galaxies_catalogues'][()].decode('utf8')
        subprocess.run(['cp', target_galaxies_catalogue_filename, temp_dir])
        target_galaxies_catalogue = Table.read(os.path.join(temp_dir,
                                                            os.path.basename(target_galaxies_catalogue_filename)),
                                               format='fits')

        source_galaxies_catalogue_filename = grp['source_galaxies_catalogues'][()].decode('utf8')
        subprocess.run(['cp', source_galaxies_catalogue_filename, temp_dir])
        source_galaxies_catalogue = Table.read(os.path.join(temp_dir,
                                                            os.path.basename(source_galaxies_catalogue_filename)),
                                               format='fits')

        root_stamps_target_fields = grp['root_stamps_target_fields'][()].decode('utf8')
        telescope_name = grp['telescope_name'][()].decode('utf8')
        target_field_name = grp['target_field_names'][()].decode('utf8')
        waveband_combinations = [name.decode('utf8') for name in grp['waveband_combinations'][()]]
        indices_target_galaxies = grp['indices_target_galaxies'][()]
        stamp_index_combinations = grp['stamp_index_combinations'][()]
        stamp_index_combinations = [int(name.decode('utf8')) for name in stamp_index_combinations]
        psf_image_type_combinations = [name.decode('utf8')
                                       for name in grp['psf_image_type_combinations'][()]]
        sigma_image_type_combinations = [name.decode('utf8')
                                         for name in grp['sigma_image_type_combinations'][()]]
        background_estimate_method_combinations = [name.decode('utf8')
                                                   for name in
                                                   grp['background_estimate_method_combinations'][()]]
        wavebands = [name.decode('utf8') for name in grp['wavebands'][()]]
        psf_image_types = [name.decode('utf8') for name in grp['psf_image_types'][()]]
        sigma_image_types = [name.decode('utf8') for name in grp['sigma_image_types'][()]]
        background_estimate_methods = [name.decode('utf8') for name in grp['background_estimate_methods'][()]]

        create_masterstamps_catalogue(args, root_stamps_target_fields, telescope_name, target_field_name,
                                      waveband_combinations, indices_target_galaxies, stamp_index_combinations,
                                      psf_image_type_combinations, sigma_image_type_combinations,
                                      background_estimate_method_combinations, wavebands, psf_image_types,
                                      sigma_image_types, background_estimate_methods,
                                      target_galaxies_catalogue, source_galaxies_catalogue, temp_dir)

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

        current_is_missing = False

        h5pytable_filename = os.path.join(args.h5pytable_folder, args.h5pytable_filename)
        h5table = h5py.File(h5pytable_filename, 'r')
        grp = h5table['{}'.format(index)]
        root_stamps_target_fields = grp['root_stamps_target_fields'][()].decode('utf8')
        telescope_name = grp['telescope_name'][()].decode('utf8')
        target_field_name = grp['target_field_names'][()].decode('utf8')

        try:
            table = Table.read(os.path.join(root_stamps_target_fields, '{}_{}_stamps.cat'
                                            .format(telescope_name, target_field_name)), format='fits')
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

    cwd = os.getcwd()

    description = "Create master catalogue of GALFIT run on stamps around target galaxies in Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_filename', type=str, action='store', default='table_masterstamps.h5',
                        help='h5py table filename')
    parser.add_argument('--temp_dir_path', type=str, action='store', default=cwd,
                        help='temporary folder where to make calculations locally, used only if --local_or_cluster'
                             ' is set to local')
    parser.add_argument('--waveband_key', type=str, action='store', default=cwd,
                        help='waveband catalogue header filter for matching with target galaxies catalogue')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')

    args = parser.parse_args(args)

    return args
