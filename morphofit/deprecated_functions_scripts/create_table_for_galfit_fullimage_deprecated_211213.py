#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
from astropy.table import Table
import itertools
import numpy as np
import glob
import h5py
import os

# morphofit imports


def select_images(path, target, waveband, image_type, crop_suffix=None):
    """

    :param path:
    :param target:
    :param waveband:
    :param image_type:
    :param crop_suffix:
    :return:
    """

    if crop_suffix is None:
        images = glob.glob(path + target + '/*{}*_{}.fits'.format(waveband, image_type))
    else:
        images = glob.glob(path + target + '/*{}*_{}_{}.fits'.format(waveband, image_type, crop_suffix))
    images = os.path.basename(images[0])

    return images


root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'
telescope_name = 'HST'
target_names_list = ['abells1063', 'macs0416', 'macs1149']
psf_image_types_list = ['pca_psf', 'observed_psf', 'moffat_psf', 'effective_psf']
sigma_image_types_list = ['custom_sigma_image', 'internal_generated_sigma_image']
background_estimate_methods_list = ['background_free_fit', 'background_fixed_value']
wavebands_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']

root_files, target_names, psf_image_types, sigma_image_types, background_estimate_methods, wavebands, \
    sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames, \
    exposure_times, magnitude_zeropoints, effective_gains, instrumental_gains,\
    background_values, regions_mastercatalogue_filenames, \
    input_galfit_filenames, sigma_image_filenames, output_model_image_filenames, \
    psf_image_filenames, psf_sampling_factors, constraints_file_filenames, telescope_names = \
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
    [], [], [], [], [], []

for target_name in target_names_list:

    parameters_table = Table.read(root + '{}/{}_param_table.fits'.format(target_name, target_name),
                                  format='fits')

    combinations_length = len(psf_image_types_list) * len(sigma_image_types_list) * \
        len(background_estimate_methods_list) * len(wavebands_list)

    combinations = [['{}'.format(x), '{}'.format(y), '{}'.format(z), '{}'.format(m)]
                    for x, y, z, m in itertools.product(psf_image_types_list, sigma_image_types_list,
                                                        background_estimate_methods_list, wavebands_list)]

    for j in range(len(combinations)):
        telescope_names.append(telescope_name.encode('utf8'))
        root_files.append(root + '{}/'.format(target_name))
        target_names.append(target_name.encode('utf8'))
        psf_image_types.append(combinations[j][0].encode('utf8'))
        sigma_image_types.append(combinations[j][1].encode('utf8'))
        background_estimate_methods.append(combinations[j][2].encode('utf8'))
        wavebands.append(combinations[j][3].encode('utf8'))

        sci_image_filename = select_images(root, target_name, combinations[j][3], 'drz',
                                           crop_suffix='muse')
        seg_image_filename = select_images(root, target_name, combinations[j][3], 'forced_seg',
                                           crop_suffix='muse')
        sci_image_filenames.append(sci_image_filename.encode('utf8'))
        seg_image_filenames.append(seg_image_filename.encode('utf8'))
        try:
            rms_image_filename = select_images(root, target_name, combinations[j][3],
                                               'rms', crop_suffix='muse')
            rms_image_filenames.append(rms_image_filename.encode('utf8'))
        except Exception as e:
            print(e)
            rms_image_filenames.append('None'.encode('utf8'))
        try:
            exp_image_filename = select_images(root, target_name, combinations[j][3],
                                               'exp', crop_suffix='muse')
            exp_image_filenames.append(exp_image_filename.encode('utf8'))
        except Exception as e:
            print(e)
            exp_image_filenames.append('None'.encode('utf8'))

        w = np.where(parameters_table['wavebands'] == combinations[j][3])
        exposure_times.append(parameters_table[w]['exptimes'][0])
        magnitude_zeropoints.append(parameters_table[w]['zeropoints'][0])
        effective_gains.append(parameters_table[w]['effective_gains'][0])
        instrumental_gains.append(parameters_table[w]['instrumental_gains'][0])
        background_values.append(parameters_table[w]['bkg_amps'][0])

        regions_mastercatalogue_filenames.append('{}_{}_regions_multiband.forced.sexcat'
                                                 .format(telescope_name, target_name).encode('utf8'))

        input_galfit_filenames.append('{}_{}_{}_{}_{}_{}.INPUT'.format(telescope_name, target_name,
                                                                       combinations[j][3], combinations[j][0],
                                                                       combinations[j][1],
                                                                       combinations[j][2]).encode('utf8'))

        if combinations[j][1] == 'custom_sigma_image':
            sigma_image_filenames.append('{}_{}_{}_sigma_image.fits'.format(telescope_name, target_name,
                                                                            combinations[j][3]).encode('utf8'))
        else:
            sigma_image_filenames.append('None'.encode('utf8'))

        output_model_image_filenames.append('{}_{}_{}_{}_{}_{}_imgblock.fits'.format(telescope_name,
                                                                                     target_name,
                                                                                     combinations[j][3],
                                                                                     combinations[j][0],
                                                                                     combinations[j][1],
                                                                                     combinations[j][2])
                                            .encode('utf8'))

        psf_image_filenames.append('{}_{}_{}.fits'.format(combinations[j][0], target_name, combinations[j][3])
                                   .encode('utf8'))

        if combinations[j][0] == 'effective_psf':
            psf_sampling_factors.append(2)
        else:
            psf_sampling_factors.append(1)

        # constraints_file_filenames.append('constraints_file_{}_{}.CONSTRAINTS'.format(target_name,
        #                                                                               combinations[j][3])
        #                                   .encode('utf8'))
        constraints_file_filenames.append('None'.encode('utf8'))

root_files = [name.encode('utf8') for name in root_files]

pixel_scale = 0.060
convolution_box_size = 256
galfit_binary_file = '/Users/torluca/galfit'.encode('utf8')
id_key_sources_catalogue = 'NUMBER'.encode('utf8')

print('Number of combinations: {}'.format(len(telescope_names)))

with h5py.File(root + 'h5tables/table_galfit_on_fullimage_run.h5', mode='w') as h5table:
    h5table.create_dataset(name='root_files', data=root_files)
    h5table.create_dataset(name='telescope_names', data=telescope_names)
    h5table.create_dataset(name='target_names', data=target_names)
    h5table.create_dataset(name='wavebands', data=wavebands)
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
    h5table.create_dataset(name='regions_mastercatalogue_filenames', data=regions_mastercatalogue_filenames)
    h5table.create_dataset(name='input_galfit_filenames', data=input_galfit_filenames)
    h5table.create_dataset(name='sigma_image_filenames', data=sigma_image_filenames)
    h5table.create_dataset(name='output_model_image_filenames', data=output_model_image_filenames)
    h5table.create_dataset(name='psf_image_filenames', data=psf_image_filenames)
    h5table.create_dataset(name='psf_sampling_factors', data=psf_sampling_factors)
    h5table.create_dataset(name='constraints_file_filenames', data=constraints_file_filenames)
    h5table.create_dataset(name='pixel_scale', data=pixel_scale)
    h5table.create_dataset(name='convolution_box_size', data=convolution_box_size)
    h5table.create_dataset(name='galfit_binary_file', data=galfit_binary_file)
    h5table.create_dataset(name='id_key_sources_catalogue', data=id_key_sources_catalogue)
