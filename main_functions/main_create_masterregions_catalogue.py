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
from astropy.table import Table, vstack
import numpy as np

# morphofit imports
from morphofit.utils import get_logger
from morphofit.catalogue_managing import combine_properties  # , delete_repeating_sources
from morphofit.catalogue_managing import match_with_source_galaxies_catalogue

logger = get_logger(__file__)


def create_masterregions_catalogue(args, root_target_fields, root_regions_target_fields, telescope_name,
                                   target_field_name, waveband_combinations, indices_regions, region_index_combinations,
                                   psf_image_type_combinations, sigma_image_type_combinations,
                                   background_estimate_method_combinations, wavebands,
                                   source_galaxies_catalogue, temp_dir):
    """

    :param args:
    :param root_target_fields:
    :param root_regions_target_fields:
    :param telescope_name:
    :param target_field_name:
    :param waveband_combinations:
    :param indices_regions:
    :param region_index_combinations:
    :param psf_image_type_combinations:
    :param sigma_image_type_combinations:
    :param background_estimate_method_combinations:
    :param wavebands:
    :param source_galaxies_catalogue:
    :param temp_dir:
    :return:
    """

    galfit_properties_mastertable = Table()

    for i in range(len(waveband_combinations)):
        try:
            subprocess.run(['cp', os.path.join(root_regions_target_fields, 'region{}_{}_{}_{}/{}_{}_{}_{}_{}_{}_{}.fits'
                                               .format(region_index_combinations[i],
                                                       psf_image_type_combinations[i],
                                                       sigma_image_type_combinations[i],
                                                       background_estimate_method_combinations[i], telescope_name,
                                                       target_field_name, waveband_combinations[i],
                                                       region_index_combinations[i],
                                                       psf_image_type_combinations[i], sigma_image_type_combinations[i],
                                                       background_estimate_method_combinations[i])), temp_dir])

            best_fit_properties_table_filename = os.path.join(temp_dir, '{}_{}_{}_{}_{}_{}_{}.fits'
                                                              .format(telescope_name,
                                                                      target_field_name,
                                                                      waveband_combinations[i],
                                                                      region_index_combinations[i],
                                                                      psf_image_type_combinations[i],
                                                                      sigma_image_type_combinations[i],
                                                                      background_estimate_method_combinations[i]))
            best_fit_properties_table = Table.read(best_fit_properties_table_filename, format='fits', memmap=True)
            galfit_properties_mastertable = vstack([galfit_properties_mastertable, best_fit_properties_table])
            subprocess.run(['rm', best_fit_properties_table_filename])
        except OSError as exception:
            logger.info(exception)
            logger.info('Missing {}_{}_{}_{}_{}_{}_{}.fits table'.format(telescope_name, target_field_name,
                                                                         waveband_combinations[i],
                                                                         region_index_combinations[i],
                                                                         psf_image_type_combinations[i],
                                                                         sigma_image_type_combinations[i],
                                                                         background_estimate_method_combinations[i]))
            pass

    galfit_properties_mastertable_filename = os.path.join(temp_dir, '{}_{}_{}'
                                                          .format(telescope_name, target_field_name,
                                                                  args.galfit_properties_mastertable_suffix))
    galfit_properties_mastertable.write(galfit_properties_mastertable_filename, format='fits', overwrite=True)

    subprocess.run(['cp', galfit_properties_mastertable_filename, root_regions_target_fields])

    multiband_tables = []

    for region_number in indices_regions:

        region_number_mask = np.where(galfit_properties_mastertable['REGION_INDEX'] == str(region_number))

        try:
            table = combine_properties(galfit_properties_mastertable[region_number_mask], wavebands, telescope_name,
                                       target_field_name, args.galaxy_ids_key, args.light_profiles_key,
                                       args.galaxy_components_key, fit_kind='regions', index=region_number)

            table = match_with_source_galaxies_catalogue(table, source_galaxies_catalogue,
                                                         args.source_galaxies_catalogue_id_key)

            table.write(os.path.join(temp_dir, '{}_{}_region{}.cat'.format(telescope_name, target_field_name,
                                                                           region_number)), format='fits',
                        overwrite=True)
            multiband_tables.append(table)
        except Exception as e:
            logger.info(e)

    multiband_table = vstack(multiband_tables, join_type='exact')
    multiband_table_filename = os.path.join(temp_dir, '{}_{}_regions_orig.cat'.format(telescope_name,
                                                                                      target_field_name))
    multiband_table.write(multiband_table_filename, format='fits', overwrite=True)

    subprocess.run(['cp', multiband_table_filename, root_target_fields])

    # multiband_table = delete_repeating_sources(multiband_table, wavebands)
    # multiband_table_filename = os.path.join(temp_dir, '{}_{}_regions.cat'.format(telescope_name, target_field_name))
    # multiband_table.write(multiband_table_filename, format='fits', overwrite=True)

    subprocess.run(['cp', multiband_table_filename, root_target_fields])
    subprocess.run(['cp'] + glob.glob(os.path.join(temp_dir, '*region*.cat')) + [root_regions_target_fields])


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

        source_galaxies_catalogue_filename = grp['source_galaxies_catalogues'][()].decode('utf8')
        subprocess.run(['cp', source_galaxies_catalogue_filename, temp_dir])
        source_galaxies_catalogue = Table.read(os.path.join(temp_dir,
                                                            os.path.basename(source_galaxies_catalogue_filename)),
                                               format='fits')

        root_target_fields = grp['root_target_fields'][()].decode('utf8')
        root_regions_target_fields = grp['root_regions_target_fields'][()].decode('utf8')
        telescope_name = grp['telescope_name'][()].decode('utf8')
        target_field_name = grp['target_field_names'][()].decode('utf8')
        waveband_combinations = [name.decode('utf8') for name in grp['waveband_combinations'][()]]
        indices_regions = [name.decode('utf8') for name in grp['indices_regions'][()]]
        region_index_combinations = [name.decode('utf8') for name in grp['region_index_combinations'][()]]
        psf_image_type_combinations = [name.decode('utf8')
                                       for name in grp['psf_image_type_combinations'][()]]
        sigma_image_type_combinations = [name.decode('utf8')
                                         for name in grp['sigma_image_type_combinations'][()]]
        background_estimate_method_combinations = [name.decode('utf8')
                                                   for name in
                                                   grp['background_estimate_method_combinations'][()]]
        wavebands = [name.decode('utf8') for name in grp['wavebands'][()]]

        h5table.close()

        create_masterregions_catalogue(args, root_target_fields, root_regions_target_fields, telescope_name,
                                       target_field_name, waveband_combinations, indices_regions,
                                       region_index_combinations, psf_image_type_combinations,
                                       sigma_image_type_combinations, background_estimate_method_combinations,
                                       wavebands, source_galaxies_catalogue, temp_dir)

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
        root_regions_target_fields = grp['root_regions_target_fields'][()].decode('utf8')
        telescope_name = grp['telescope_name'][()].decode('utf8')
        target_field_name = grp['target_field_names'][()].decode('utf8')
        h5table.close()

        try:
            table = Table.read(os.path.join(root_regions_target_fields, '{}_{}_regions.cat'
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

    description = "Create master catalogue of GALFIT run on regions containing target galaxies in Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_filename', type=str, action='store', default='table_masterregions.h5',
                        help='h5py table filename')
    parser.add_argument('--temp_dir_path', type=str, action='store', default=cwd,
                        help='temporary folder where to make calculations locally, used only if --local_or_cluster'
                             ' is set to local')
    parser.add_argument('--galfit_properties_mastertable_suffix', type=str, action='store', default='mastertable.fits',
                        help='Filename suffix for the master table of the run on regions')
    parser.add_argument('--source_galaxies_catalogue_id_key', type=str, action='store', default='NUMBER',
                        help='Catalogue header keywords for id')
    parser.add_argument('--galaxy_ids_key', type=str, action='store', default='NUMBER',
                        help='Catalogue header key of the galaxy Ids')
    parser.add_argument('--light_profiles_key', type=str, action='store', default='LIGHT_PROFILE',
                        help='Catalogue header key of the galaxy light profiles')
    parser.add_argument('--galaxy_components_key', type=str, action='store', default='COMPONENT_NUMBER',
                        help='Catalogue header key of the galaxy components')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')

    args = parser.parse_args(args)

    return args
