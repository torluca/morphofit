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


def select_images(path, target, waveband, image_type):
    images = glob.glob(path + target + '/*{}*_{}'.format(waveband, image_type))
    images = os.path.basename(images[0])

    return images


root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'
telescope_name = 'HST'
target_names_list = ['abells1063', 'macs0416', 'macs1149']
psf_image_types_list = ['pca_psf', 'observed_psf', 'moffat_psf', 'effective_psf']
sigma_image_types_list = ['custom_sigma_image', 'internal_generated_sigma_image']
background_estimate_methods_list = ['background_free_fit', 'background_fixed_value']
wavebands_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']

root_files, target_names, psf_image_types, sigma_image_types, background_estimate_methods, wavebands, stamp_indices, \
    sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames, exposure_times, \
    magnitude_zeropoints, effective_gains, instrumental_gains,\
    background_values, sources_catalogues, target_galaxies_id, target_galaxies_x,\
    target_galaxies_y, target_galaxies_ra, target_galaxies_dec, target_galaxies_magnitudes, \
    target_galaxies_effective_radii, target_galaxies_reference_effective_radii, target_galaxies_minor_axis, \
    target_galaxies_major_axis, \
    target_galaxies_position_angles, input_galfit_filenames, sigma_image_filenames, output_model_image_filenames, \
    psf_image_filenames, psf_sampling_factors, constraints_file_filenames, telescope_names =\
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],\
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

for target_name in target_names_list:
    target_galaxies_catalogue = Table.read(root + '{}/{}_{}_target_multiband.forced.sexcat'.format(target_name,
                                                                                                   telescope_name,
                                                                                                   target_name))
    parameters_table = Table.read(root + '{}/{}_param_table.fits'.format(target_name, target_name),
                                  format='fits')
    n_target_galaxies = len(target_galaxies_catalogue)
    indices_stamps = np.arange(0, n_target_galaxies, 1, dtype=int)
    combinations_length = len(psf_image_types_list) * len(sigma_image_types_list) * \
        len(background_estimate_methods_list) * len(wavebands_list) * n_target_galaxies

    combinations = [['{}'.format(x), '{}'.format(y), '{}'.format(z), '{}'.format(m), '{}'.format(n)]
                    for x, y, z, m, n in itertools.product(psf_image_types_list, sigma_image_types_list,
                                                           background_estimate_methods_list, wavebands_list,
                                                           indices_stamps)]

    for j in range(len(combinations)):
        telescope_names.append(telescope_name.encode('utf8'))
        root_files.append(root + '{}/'.format(target_name))
        target_names.append(target_name.encode('utf8'))
        psf_image_types.append(combinations[j][0].encode('utf8'))
        sigma_image_types.append(combinations[j][1].encode('utf8'))
        background_estimate_methods.append(combinations[j][2].encode('utf8'))
        wavebands.append(combinations[j][3].encode('utf8'))
        stamp_indices.append(int(combinations[j][4]))
        sci_image_filenames.append(select_images(root, target_name, combinations[j][3], 'drz.fits').encode('utf8'))
        seg_image_filenames.append(
            select_images(root, target_name, combinations[j][3], 'drz_forced_seg.fits').encode('utf8'))
        try:
            rms_image_filenames.append(
                select_images(root, target_name, combinations[j][3], 'rms.fits').encode('utf8'))
        except Exception as e:
            print(e)
            rms_image_filenames.append('None'.encode('utf8'))
        try:
            exp_image_filenames.append(
                select_images(root, target_name, combinations[j][3], 'exp.fits').encode('utf8'))
        except Exception as e:
            print(e)
            exp_image_filenames.append('None'.encode('utf8'))

        w = np.where(parameters_table['wavebands'] == combinations[j][3])
        exposure_times.append(parameters_table[w]['exptimes'][0])
        magnitude_zeropoints.append(parameters_table[w]['zeropoints'][0])
        effective_gains.append(parameters_table[w]['effective_gains'][0])
        instrumental_gains.append(parameters_table[w]['instrumental_gains'][0])
        background_values.append(parameters_table[w]['bkg_amps'][0])
        sources_catalogues.append('{}_{}_multiband.sources.forced.sexcat'.format(telescope_name,
                                                                                 target_name).encode('utf8'))
        idx = int(combinations[j][4])
        target_galaxies_id.append(target_galaxies_catalogue['NUMBER'][idx])
        target_galaxies_x.append(target_galaxies_catalogue['XWIN_IMAGE_f814w'][idx])
        target_galaxies_y.append(target_galaxies_catalogue['YWIN_IMAGE_f814w'][idx])
        target_galaxies_ra.append(target_galaxies_catalogue['ALPHAWIN_J2000_f814w'][idx])
        target_galaxies_dec.append(target_galaxies_catalogue['DELTAWIN_J2000_f814w'][idx])
        target_galaxies_magnitudes.append(target_galaxies_catalogue['MAG_AUTO_{}'
                                          .format(combinations[j][3])][idx])
        target_galaxies_effective_radii.append(target_galaxies_catalogue['FLUX_RADIUS_{}'
                                               .format(combinations[j][3])][idx])
        target_galaxies_reference_effective_radii.append(target_galaxies_catalogue['FLUX_RADIUS_f814w'][idx])
        target_galaxies_minor_axis.append(target_galaxies_catalogue['BWIN_IMAGE_{}'
                                          .format(combinations[j][3])][idx])
        target_galaxies_major_axis.append(target_galaxies_catalogue['AWIN_IMAGE_{}'
                                          .format(combinations[j][3])][idx])
        target_galaxies_position_angles.append(target_galaxies_catalogue['THETAWIN_SKY_{}'
                                               .format(combinations[j][3])][idx])
        input_galfit_filenames.append('{}_{}_{}_{}_{}_{}_{}.INPUT'.format(telescope_name, target_name,
                                                                          combinations[j][3], combinations[j][4],
                                                                          combinations[j][0], combinations[j][1],
                                                                          combinations[j][2]).encode('utf8'))
        if combinations[j][1] == 'custom_sigma_image':
            sigma_image_filenames.append('{}_{}_{}_stamp{}_sigma_image.fits'.format(telescope_name, target_name,
                                                                                    combinations[j][3],
                                                                                    combinations[j][4]).encode('utf8'))
        else:
            sigma_image_filenames.append('None'.encode('utf8'))
        output_model_image_filenames.append('{}_{}_{}_stamp{}_{}_{}_{}_imgblock.fits'.format(telescope_name,
                                                                                             target_name,
                                                                                             combinations[j][3],
                                                                                             combinations[j][4],
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
        #                                                                               combinations[j][3]).encode(
        #     'utf8'))
        constraints_file_filenames.append('None'.encode('utf8'))

root_files = [name.encode('utf8') for name in root_files]

enlarging_image_factor = 20
enlarging_separation_factor = 10
pixel_scale = 0.060
convolution_box_size = 256
galfit_binary_file = '/Users/torluca/galfit'.encode('utf8')
ra_key_sources_catalogue = 'ALPHAWIN_J2000_f814w'.encode('utf8')
dec_key_sources_catalogue = 'DELTAWIN_J2000_f814w'.encode('utf8')
x_key_neighbouring_sources_catalogue = 'XWIN_IMAGE_f814w'.encode('utf8')
y_key_neighbouring_sources_catalogue = 'YWIN_IMAGE_f814w'.encode('utf8')
ra_key_neighbouring_sources_catalogue = 'ALPHAWIN_J2000_f814w'.encode('utf8')
dec_key_neighbouring_sources_catalogue = 'DELTAWIN_J2000_f814w'.encode('utf8')
mag_key_neighbouring_sources_catalogue = 'MAG_AUTO'.encode('utf8')
re_key_neighbouring_sources_catalogue = 'FLUX_RADIUS'.encode('utf8')
minor_axis_key_neighbouring_sources_catalogue = 'BWIN_IMAGE'.encode('utf8')
major_axis_key_neighbouring_sources_catalogue = 'AWIN_IMAGE'.encode('utf8')
position_angle_key_neighbouring_sources_catalogue = 'THETAWIN_SKY'.encode('utf8')
id_key_neighbouring_sources_catalogue = 'NUMBER'.encode('utf8')

print('Number of combinations: {}'.format(len(telescope_names)))

with h5py.File(root + 'h5tables/table_galfit_on_stamps_run.h5', mode='w') as h5table:
    h5table.create_dataset(name='root_files', data=root_files)
    h5table.create_dataset(name='telescope_names', data=telescope_names)
    h5table.create_dataset(name='target_names', data=target_names)
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
    h5table.create_dataset(name='sources_catalogues', data=sources_catalogues)
    h5table.create_dataset(name='target_galaxies_id', data=target_galaxies_id)
    h5table.create_dataset(name='target_galaxies_x', data=target_galaxies_x)
    h5table.create_dataset(name='target_galaxies_y', data=target_galaxies_y)
    h5table.create_dataset(name='target_galaxies_ra', data=target_galaxies_ra)
    h5table.create_dataset(name='target_galaxies_dec', data=target_galaxies_dec)
    h5table.create_dataset(name='target_galaxies_magnitudes', data=target_galaxies_magnitudes)
    h5table.create_dataset(name='target_galaxies_effective_radii', data=target_galaxies_effective_radii)
    h5table.create_dataset(name='target_galaxies_reference_effective_radii',
                           data=target_galaxies_reference_effective_radii)
    h5table.create_dataset(name='target_galaxies_minor_axis', data=target_galaxies_minor_axis)
    h5table.create_dataset(name='target_galaxies_major_axis', data=target_galaxies_major_axis)
    h5table.create_dataset(name='target_galaxies_position_angles', data=target_galaxies_position_angles)
    h5table.create_dataset(name='input_galfit_filenames', data=input_galfit_filenames)
    h5table.create_dataset(name='sigma_image_filenames', data=sigma_image_filenames)
    h5table.create_dataset(name='output_model_image_filenames', data=output_model_image_filenames)
    h5table.create_dataset(name='psf_image_filenames', data=psf_image_filenames)
    h5table.create_dataset(name='psf_sampling_factors', data=psf_sampling_factors)
    h5table.create_dataset(name='constraints_file_filenames', data=constraints_file_filenames)
    h5table.create_dataset(name='enlarging_image_factor', data=enlarging_image_factor)
    h5table.create_dataset(name='enlarging_separation_factor', data=enlarging_separation_factor)
    h5table.create_dataset(name='pixel_scale', data=pixel_scale)
    h5table.create_dataset(name='convolution_box_size', data=convolution_box_size)
    h5table.create_dataset(name='galfit_binary_file', data=galfit_binary_file)
    h5table.create_dataset(name='ra_key_sources_catalogue', data=ra_key_sources_catalogue)
    h5table.create_dataset(name='dec_key_sources_catalogue', data=dec_key_sources_catalogue)
    h5table.create_dataset(name='x_key_neighbouring_sources_catalogue', data=x_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='y_key_neighbouring_sources_catalogue', data=y_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='ra_key_neighbouring_sources_catalogue', data=ra_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='dec_key_neighbouring_sources_catalogue', data=dec_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='mag_key_neighbouring_sources_catalogue', data=mag_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='re_key_neighbouring_sources_catalogue', data=re_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='minor_axis_key_neighbouring_sources_catalogue',
                           data=minor_axis_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='major_axis_key_neighbouring_sources_catalogue',
                           data=major_axis_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='position_angle_key_neighbouring_sources_catalogue',
                           data=position_angle_key_neighbouring_sources_catalogue)
    h5table.create_dataset(name='id_key_neighbouring_sources_catalogue', data=id_key_neighbouring_sources_catalogue)
