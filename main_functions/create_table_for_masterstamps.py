#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
import argparse
from astropy.table import Table
import numpy as np
import itertools
import h5py

# morphofit imports
from morphofit.utils import get_logger

logger = get_logger(__file__)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)

    target_field_names_list = args.target_field_names.split(',')

    psf_image_types = args.psf_image_types_list.split(',')
    sigma_image_types = args.sigma_image_types_list.split(',')
    background_estimate_methods = args.background_estimate_methods_list.split(',')
    wavebands = args.wavebands_list.split(',')

    root_target_fields = []
    root_stamps_target_fields = []
    source_galaxies_catalogues = []
    indices_target_galaxies = []
    waveband_combinations = []
    stamp_index_combinations = []
    psf_image_type_combinations = []
    sigma_image_type_combinations = []
    background_estimate_method_combinations = []

    for target_field_name in target_field_names_list:
        root_target_field = os.path.join(args.root_path, target_field_name)
        root_target_fields.append(root_target_field.encode('utf8'))
        root_stamps_target_fields.append(os.path.join(root_target_field, 'stamps').encode('utf8'))
        target_galaxies_catalogue = Table.read(os.path.join(root_target_field, '{}_{}_{}'
                                                            .format(args.telescope_name, target_field_name,
                                                                    args.target_galaxies_catalogue_suffix)))
        source_galaxies_catalogues.append(os.path.join(root_target_field, '{}_{}_{}'
                                                       .format(args.telescope_name, target_field_name,
                                                               args.source_galaxies_catalogue_suffix)).encode('utf8'))

        n_target_galaxies = len(np.unique(target_galaxies_catalogue[args.target_galaxies_id_key]))
        indices_stamps = np.arange(0, n_target_galaxies, 1, dtype=int)
        indices_target_galaxies.append(["{:02d}".format(item).encode('utf8') for item in indices_stamps])
        combinations = [['{}'.format(x).encode('utf8'), '{}'.format(y).encode('utf8'), '{}'.format(z).encode('utf8'),
                         '{}'.format(m).encode('utf8'), '{}'.format(n).encode('utf8')]
                        for x, y, z, m, n in itertools.product(psf_image_types, sigma_image_types,
                                                               background_estimate_methods, wavebands,
                                                               ["{:02d}".format(item) for item in indices_stamps])]
        combinations = np.array(combinations)
        psf_image_type_combinations.append(combinations[:, 0])
        sigma_image_type_combinations.append(combinations[:, 1])
        background_estimate_method_combinations.append(combinations[:, 2])
        waveband_combinations.append(combinations[:, 3])
        stamp_index_combinations.append(combinations[:, 4])

    with h5py.File(os.path.join(args.h5pytable_folder, args.h5pytable_filename), mode='w') as h5table:
        for target_field_name in target_field_names_list:
            idx = target_field_names_list.index(target_field_name)
            grp = h5table.create_group(name=str(idx))
            grp.create_dataset(name='root_target_fields', data=root_target_fields[idx])
            grp.create_dataset(name='root_stamps_target_fields', data=root_stamps_target_fields[idx])
            grp.create_dataset(name='telescope_name', data=args.telescope_name.encode('utf8'))
            grp.create_dataset(name='target_field_names', data=target_field_names_list[idx].encode('utf8'))
            grp.create_dataset(name='wavebands', data=[name.encode('utf8') for name in wavebands])
            grp.create_dataset(name='source_galaxies_catalogues',
                               data=source_galaxies_catalogues[idx])
            grp.create_dataset(name='indices_target_galaxies', data=indices_target_galaxies[idx])
            grp.create_dataset(name='waveband_combinations', data=waveband_combinations[idx])
            grp.create_dataset(name='stamp_index_combinations', data=stamp_index_combinations[idx])
            grp.create_dataset(name='psf_image_type_combinations', data=psf_image_type_combinations[idx])
            grp.create_dataset(name='sigma_image_type_combinations', data=sigma_image_type_combinations[idx])
            grp.create_dataset(name='background_estimate_method_combinations',
                               data=background_estimate_method_combinations[idx])

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

    description = "Create parameters table to obtain master catalogue of GALFIT run on " \
        "stamps around target galaxies in Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--root_path', type=str, action='store', default=cwd,
                        help='root files path')
    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_filename', type=str, action='store', default='table_masterstamps.h5',
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
    parser.add_argument('--target_galaxies_catalogue_suffix', type=str, action='store',
                        default='multiband_target.forced.cat',
                        help='Multiband target galaxies catalogue suffix')
    parser.add_argument('--source_galaxies_catalogue_suffix', type=str, action='store',
                        default='multiband_sources.forced.cat',
                        help='Multiband source galaxies catalogue suffix')
    parser.add_argument('--target_galaxies_id_key', type=str, action='store', default='NUMBER',
                        help='Catalogue header keyword for target galaxies ID')

    args = parser.parse_args(args)

    return args
