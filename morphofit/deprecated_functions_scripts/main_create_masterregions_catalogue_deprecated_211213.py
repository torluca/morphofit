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
from astropy.table import Table, vstack

# morphofit imports
from morphofit.utils import get_logger
from morphofit.catalogue_managing import add_dictionary_item, get_median_properties, delete_repeating_sources
from morphofit.catalogue_managing import create_fixed_image_table, match_with_target_galaxies_catalogue
from morphofit.catalogue_managing import save_property_dictionaries

logger = get_logger(__file__)


def create_masterregions_catalogue(root_files, output_directory, telescope_name, target_name, waveband_combinations,
                                   region_indices, region_index_combinations, psf_image_type_combinations,
                                   sigma_image_type_combinations, background_estimate_method_combinations,
                                   wavebands, psf_image_types, sigma_image_types, background_estimate_methods,
                                   target_galaxies_catalogue, full_galaxies_catalogue):
    """

    :param root_files:
    :param output_directory:
    :param telescope_name:
    :param target_name:
    :param waveband_combinations:
    :param region_indices:
    :param region_index_combinations:
    :param psf_image_type_combinations:
    :param sigma_image_type_combinations:
    :param background_estimate_method_combinations:
    :param wavebands:
    :param psf_image_types:
    :param sigma_image_types:
    :param background_estimate_methods:
    :param target_galaxies_catalogue:
    :param full_galaxies_catalogue:
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
            add_dictionary_item(root_files, telescope_name, target_name, waveband_combinations[i],
                                psf_image_type_combinations[i],
                                sigma_image_type_combinations[i], background_estimate_method_combinations[i],
                                x_dictionary, y_dictionary, ra_dictionary, dec_dictionary,
                                mag_dictionary, re_dictionary, n_dictionary, ar_dictionary, pa_dictionary,
                                background_value_dictionary, background_x_gradient_dictionary,
                                background_y_gradient_dictionary, reduced_chisquare_dictionary, kind='region',
                                index=region_index_combinations[i])
        except Exception as exception:
            logger.info(exception)
            logger.info('Missing {}_{}_{}_{}_{}_{}_{}.h5 table'.format(telescope_name, target_name,
                                                                       waveband_combinations[i],
                                                                       region_index_combinations[i],
                                                                       psf_image_type_combinations[i],
                                                                       sigma_image_type_combinations[i],
                                                                       background_estimate_method_combinations[i]))
            pass

    os.makedirs(output_directory, exist_ok=True)

    save_property_dictionaries(output_directory, telescope_name, target_name,
                               x_dictionary, y_dictionary, ra_dictionary, dec_dictionary,
                               mag_dictionary, re_dictionary, n_dictionary, ar_dictionary, pa_dictionary,
                               background_value_dictionary, background_x_gradient_dictionary,
                               background_y_gradient_dictionary, reduced_chisquare_dictionary)

    multiband_tables = []
    for region_number in region_indices:
        try:
            number_galaxies_region = max([len(v) for k, v in mag_dictionary.items() if (target_name in k) &
                                         (str(region_number) == k.split('_')[2])])
            x_positions, x_position_errors, y_positions, y_position_errors, ra, dec, \
                total_magnitudes, total_magnitude_errors, effective_radii, \
                effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios, \
                axis_ratio_errors, position_angles, position_angle_errors, background_values, \
                background_value_errors, background_x_gradients, background_x_gradient_errors, \
                background_y_gradients, background_y_gradient_errors, \
                covariance_effective_radii_position_angles, \
                covariance_effective_radii_position_angles_magnitudes = \
                get_median_properties(target_name, wavebands, psf_image_types, background_estimate_methods,
                                      sigma_image_types,
                                      number_galaxies_region, x_dictionary, y_dictionary,
                                      ra_dictionary, dec_dictionary, mag_dictionary, re_dictionary, n_dictionary,
                                      ar_dictionary, pa_dictionary, background_value_dictionary,
                                      background_x_gradient_dictionary, background_y_gradient_dictionary,
                                      kind='region', index=region_number)

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
            table = match_with_target_galaxies_catalogue(table, target_galaxies_catalogue, full_galaxies_catalogue,
                                                         wavebands, waveband_key='f814w')
            table.write(output_directory + '{}_{}_region{}_multiband.forced.sexcat'
                        .format(telescope_name, target_name, region_number), format='fits', overwrite=True)
            multiband_tables.append(table)
        except Exception as exception:
            logger.info(exception)
            logger.info('Missing {}'.format(region_number))
            pass

    multiband_table = vstack(multiband_tables, join_type='exact')
    multiband_table = delete_repeating_sources(multiband_table, wavebands)
    multiband_table.write(output_directory + '{}_{}_regions_multiband.forced.sexcat'
                          .format(telescope_name, target_name), format='fits', overwrite=True)


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

        grp = h5table['{}'.format(index)]

        root_files = grp['root_files'].value.decode('utf8')
        output_directory = grp['output_directories'].value.decode('utf8')
        telescope_name = grp['telescope_names'].value.decode('utf8')
        target_name = grp['target_names'].value.decode('utf8')
        waveband_combinations = [name.decode('utf8') for name in grp['waveband_combinations'].value]
        region_indices = [name.decode('utf8') for name in grp['region_indices'].value]
        region_index_combinations = grp['region_index_combinations'].value
        region_index_combinations = [name.decode('utf8') for name in region_index_combinations]
        psf_image_type_combinations = [name.decode('utf8')
                                       for name in grp['psf_image_type_combinations'].value]
        sigma_image_type_combinations = [name.decode('utf8')
                                         for name in grp['sigma_image_type_combinations'].value]
        background_estimate_method_combinations = [name.decode('utf8')
                                                   for name in grp['background_estimate_method_combinations'].value]
        wavebands = [name.decode('utf8') for name in grp['wavebands'].value]
        psf_image_types = [name.decode('utf8') for name in grp['psf_image_types'].value]
        sigma_image_types = [name.decode('utf8') for name in grp['sigma_image_types'].value]
        background_estimate_methods = [name.decode('utf8') for name in grp['background_estimate_methods'].value]
        target_galaxies_catalogue_filename = grp['target_galaxies_catalogue_filenames'].value.decode('utf8')
        target_galaxies_catalogue = Table.read(target_galaxies_catalogue_filename, format='fits')
        full_galaxies_catalogue_filename = grp['full_galaxies_catalogue_filenames'].value.decode('utf8')
        full_galaxies_catalogue = Table.read(full_galaxies_catalogue_filename, format='fits')

        create_masterregions_catalogue(root_files, output_directory, telescope_name, target_name, waveband_combinations,
                                       region_indices, region_index_combinations, psf_image_type_combinations,
                                       sigma_image_type_combinations, background_estimate_method_combinations,
                                       wavebands, psf_image_types, sigma_image_types, background_estimate_methods,
                                       target_galaxies_catalogue, full_galaxies_catalogue)

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

        grp = h5table['{}'.format(index)]

        current_is_missing = False

        output_directory = grp['output_directories'].value.decode('utf8')
        telescope_name = grp['telescope_names'].value.decode('utf8')
        target_name = grp['target_names'].value.decode('utf8')

        try:
            table = Table.read(output_directory + '{}_{}_regions_multiband.forced.sexcat'
                               .format(telescope_name, target_name), format='fits')
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

    description = "Create master catalogue from GALFIT on regions"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--filename_h5pytable', type=str, action='store', default='table.h5',
                        help='h5py table of the file to run on')
    args = parser.parse_args(args)

    return args.filename_h5pytable
