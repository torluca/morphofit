#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import itertools
import numpy as np
import h5py

# morphofit imports

targets = ['abells1063', 'macs0416', 'macs1149']
telescope_name = 'HST'
target_names = [name.encode('utf8') for name in targets]
telescope_names = np.full(len(targets), telescope_name)
psf_image_types = ['pca_psf', 'observed_psf', 'moffat_psf', 'effective_psf']
sigma_image_types = ['custom_sigma_image', 'internal_generated_sigma_image']
background_estimate_methods = ['background_free_fit', 'background_fixed_value']
wavebands = ['f435w', 'f606w', 'f814w', 'f105w',
             'f125w', 'f140w', 'f160w']
root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'

root_files = []
output_directories = []
target_galaxies_catalogue_filenames = []
full_galaxies_catalogue_filenames = []
region_indices = []
waveband_combinations = []
region_index_combinations = []
psf_image_type_combinations = []
sigma_image_type_combinations = []
background_estimate_method_combinations = []

for target in targets:

    root_files.append(root.encode('utf8') + '{}/regions/'.format(target).encode('utf8'))
    output_directories.append(root.encode('utf8') + '{}/regions/cats/'.format(target).encode('utf8'))
    target_galaxies_catalogue_filenames.append(root.encode('utf8') +
                                               '{}/{}_{}_target_multiband.forced.sexcat'
                                               .format(target, telescope_name, target).encode('utf8'))
    full_galaxies_catalogue_filenames.append(root.encode('utf8') +
                                             '{}/{}_{}_multiband.sources.forced.sexcat'
                                             .format(target, telescope_name, target).encode('utf8'))

    number_of_regions_perside = 3
    indices_regions = ['{}{}'.format(i, j) for i in range(number_of_regions_perside)
                       for j in range(number_of_regions_perside)]
    region_indices.append([name.encode('utf8') for name in indices_regions])

    combinations_length = len(psf_image_types) * len(background_estimate_methods) * len(sigma_image_types) * \
        len(wavebands) * len(indices_regions)
    combinations = [['{}'.format(x).encode('utf8'), '{}'.format(y).encode('utf8'), '{}'.format(z).encode('utf8'),
                     '{}'.format(m).encode('utf8'), '{}'.format(n).encode('utf8')]
                    for x, y, z, m, n in itertools.product(psf_image_types, sigma_image_types,
                                                           background_estimate_methods, wavebands,
                                                           indices_regions)]
    combinations = np.array(combinations)
    psf_image_type_combinations.append(combinations[:, 0])
    sigma_image_type_combinations.append(combinations[:, 1])
    background_estimate_method_combinations.append(combinations[:, 2])
    waveband_combinations.append(combinations[:, 3])
    region_index_combinations.append(combinations[:, 4])

psf_image_types = [name.encode('utf8') for name in psf_image_types]
sigma_image_types = [name.encode('utf8') for name in sigma_image_types]
background_estimate_methods = [name.encode('utf8') for name in background_estimate_methods]
wavebands = [name.encode('utf8') for name in wavebands]
telescope_names = [name.encode('utf8') for name in telescope_names]

with h5py.File(root + 'h5tables/table_masterregions.h5', mode='w') as h5table:
    for target in targets:
        idx = targets.index(target)
        grp = h5table.create_group(name=str(idx))
        grp.create_dataset(name='root_files', data=root_files[idx])
        grp.create_dataset(name='output_directories', data=output_directories[idx])
        grp.create_dataset(name='telescope_names', data=telescope_names[idx])
        grp.create_dataset(name='target_names', data=target_names[idx])
        grp.create_dataset(name='wavebands', data=wavebands)
        grp.create_dataset(name='psf_image_types', data=psf_image_types)
        grp.create_dataset(name='sigma_image_types', data=sigma_image_types)
        grp.create_dataset(name='background_estimate_methods', data=background_estimate_methods)
        grp.create_dataset(name='target_galaxies_catalogue_filenames', data=target_galaxies_catalogue_filenames[idx])
        grp.create_dataset(name='full_galaxies_catalogue_filenames',
                           data=full_galaxies_catalogue_filenames[idx])
        grp.create_dataset(name='region_indices', data=region_indices[idx])
        grp.create_dataset(name='waveband_combinations', data=waveband_combinations[idx])
        grp.create_dataset(name='region_index_combinations', data=region_index_combinations[idx])
        grp.create_dataset(name='psf_image_type_combinations', data=psf_image_type_combinations[idx])
        grp.create_dataset(name='sigma_image_type_combinations', data=sigma_image_type_combinations[idx])
        grp.create_dataset(name='background_estimate_method_combinations',
                           data=background_estimate_method_combinations[idx])
