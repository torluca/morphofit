#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import numpy as np
import os
import subprocess

# morphofit imports
from morphofit.utils import single_ra_dec_2_xy
from morphofit.utils import get_logger

logger = get_logger(__file__)


def fitting_choice(x_neighbouring_galaxies, y_neighbouring_galaxies, image_size):
    """

    :param x_neighbouring_galaxies:
    :param y_neighbouring_galaxies:
    :param image_size:
    :return:
    """

    if (x_neighbouring_galaxies < 0) | (x_neighbouring_galaxies > image_size[0]) | \
            (y_neighbouring_galaxies < 0) | (y_neighbouring_galaxies > image_size[1]):
        tofit = 0
    else:
        tofit = 1

    return tofit


def format_positions(sci_image_filename, source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param sci_image_filename:
    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    source_positions = np.empty((len(source_galaxies_catalogue), 4))

    for i in range(len(source_galaxies_catalogue)):
        x_source, y_source = single_ra_dec_2_xy(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[1],
                                                                                         waveband)][i],
                                                source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[2],
                                                                                         waveband)][i],
                                                sci_image_filename)
        source_positions[i, :] = [x_source, y_source,
                                  source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)][i],
                                  source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)][i]]

    return source_positions


def format_ra_dec(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    ra = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[1], waveband)])
    dec = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[2], waveband)])

    return ra, dec


def format_magnitudes(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    total_magnitudes = np.empty((len(source_galaxies_catalogue), 2))
    total_magnitudes[:, 0] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[3], waveband)])
    total_magnitudes[:, 1] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)])

    return total_magnitudes


def format_effective_radii(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    effective_radii = np.empty((len(source_galaxies_catalogue), 2))
    effective_radii[:, 0] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[4], waveband)])
    effective_radii[:, 1] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)])

    return effective_radii


def format_axis_ratios(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    axis_ratios = np.empty((len(source_galaxies_catalogue), 2))
    if source_galaxies_keys[6] == source_galaxies_keys[7]:
        axis_ratios[:, 0] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[6], waveband)])
        axis_ratios[:, 1] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)])
    else:
        axis_ratios[:, 0] = (np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[6], waveband)]) /
                             np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[7], waveband)]))**2
        axis_ratios[:, 1] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)])

    return axis_ratios


def format_position_angles(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    position_angles = np.empty((len(source_galaxies_catalogue), 2))
    position_angles[:, 0] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[8], waveband)])
    position_angles[:, 1] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)])

    return position_angles


def format_light_profiles(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    light_profiles = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-1], waveband)],
                              dtype='U10')
    light_profiles = np.char.strip(light_profiles)

    return light_profiles


def format_sersic_indices(source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    sersic_indices = np.empty((len(source_galaxies_catalogue), 2))
    sersic_indices[:, 0] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[5], waveband)])
    sersic_indices[:, 1] = np.array(source_galaxies_catalogue['{}_{}'.format(source_galaxies_keys[-3], waveband)])

    return sersic_indices


def format_properties_for_galfit(sci_image_filename, source_galaxies_catalogue, source_galaxies_keys, waveband):
    """

    :param sci_image_filename:
    :param source_galaxies_catalogue:
    :param source_galaxies_keys:
    :param waveband:
    :return:
    """

    source_positions = format_positions(sci_image_filename, source_galaxies_catalogue, source_galaxies_keys, waveband)

    ra, dec = format_ra_dec(source_galaxies_catalogue, source_galaxies_keys, waveband)

    total_magnitudes = format_magnitudes(source_galaxies_catalogue, source_galaxies_keys, waveband)

    effective_radii = format_effective_radii(source_galaxies_catalogue, source_galaxies_keys, waveband)

    axis_ratios = format_axis_ratios(source_galaxies_catalogue, source_galaxies_keys, waveband)

    position_angles = format_position_angles(source_galaxies_catalogue, source_galaxies_keys, waveband)

    light_profiles = format_light_profiles(source_galaxies_catalogue, source_galaxies_keys, waveband)

    sersic_indices = format_sersic_indices(source_galaxies_catalogue, source_galaxies_keys, waveband)

    subtract = np.full(len(source_galaxies_catalogue), '0')

    return light_profiles, source_positions, ra, dec, total_magnitudes, effective_radii, sersic_indices, axis_ratios, \
        position_angles, subtract


def write_sersic_to_galfit_inputfile(text_file, index, source_positions, total_magnitudes, effective_radii,
                                     sersic_indices, axis_ratios, position_angles, subtract):
    """

    :param text_file:
    :param index:
    :param source_positions:
    :param total_magnitudes:
    :param effective_radii:
    :param sersic_indices:
    :param axis_ratios:
    :param position_angles:
    :param subtract:
    :return:
    """

    text_file.write('# Object number {} \n'
                    '0) sersic # object type \n'
                    '1) {} {} {} {} #  position x, y \n'
                    '3) {} {} # Integrated magnitude \n'
                    '4) {} {} #  R_e (half-light radius) [pix] \n'
                    '5) {} {} #  Sersic index n (de Vaucouleurs n=4) \n'
                    '9) {} {} #  axis ratio (b/a) \n'
                    '10) {} {} #  position angle (PA) [deg: Up=0, Left=90] \n'
                    'Z) {} #  output option (0 = resid., 1 = Do not subtract)\n\n'
                    .format(str(index + 2),
                            source_positions[index][0],
                            source_positions[index][1],
                            int(source_positions[index][2]),
                            int(source_positions[index][3]),
                            total_magnitudes[index][0],
                            total_magnitudes[index][1],
                            effective_radii[index][0],
                            effective_radii[index][1],
                            sersic_indices[index][0],
                            sersic_indices[index][1],
                            axis_ratios[index][0],
                            axis_ratios[index][1],
                            position_angles[index][0],
                            position_angles[index][1],
                            subtract[index]))


def write_devauc_to_galfit_inputfile(text_file, index, source_positions, total_magnitudes, effective_radii,
                                     sersic_indices, axis_ratios, position_angles, subtract):
    """

    :param text_file:
    :param index:
    :param source_positions:
    :param total_magnitudes:
    :param effective_radii:
    :param sersic_indices:
    :param axis_ratios:
    :param position_angles:
    :param subtract:
    :return:
    """

    text_file.write('# Object number {} \n'
                    '0) devauc # object type \n'
                    '1) {} {} {} {} #  position x, y \n'
                    '3) {} {} # Integrated magnitude \n'
                    '4) {} {} #  R_e (half-light radius) [pix] \n'
                    '9) {} {} #  axis ratio (b/a) \n'
                    '10) {} {} #  position angle (PA) [deg: Up=0, Left=90] \n'
                    'Z) {} #  output option (0 = resid., 1 = Do not subtract)\n\n'
                    .format(str(index + 2),
                            source_positions[index][0],
                            source_positions[index][1],
                            int(source_positions[index][2]),
                            int(source_positions[index][3]),
                            total_magnitudes[index][0],
                            total_magnitudes[index][1],
                            effective_radii[index][0],
                            effective_radii[index][1],
                            axis_ratios[index][0],
                            axis_ratios[index][1],
                            position_angles[index][0],
                            position_angles[index][1],
                            subtract[index]))


def write_expdisk_to_galfit_inputfile(text_file, index, source_positions, total_magnitudes, effective_radii,
                                      sersic_indices, axis_ratios, position_angles, subtract):
    """

    :param text_file:
    :param index:
    :param source_positions:
    :param total_magnitudes:
    :param effective_radii:
    :param sersic_indices:
    :param axis_ratios:
    :param position_angles:
    :param subtract:
    :return:
    """

    text_file.write('# Object number {} \n'
                    '0) expdisk # object type \n'
                    '1) {} {} {} {} #  position x, y \n'
                    '3) {} {} # Integrated magnitude \n'
                    '4) {} {} #  Rs (scale radius) [pix] \n'
                    '9) {} {} #  axis ratio (b/a) \n'
                    '10) {} {} #  position angle (PA) [deg: Up=0, Left=90] \n'
                    'Z) {} #  output option (0 = resid., 1 = Do not subtract)\n\n'
                    .format(str(index + 2),
                            source_positions[index][0],
                            source_positions[index][1],
                            int(source_positions[index][2]),
                            int(source_positions[index][3]),
                            total_magnitudes[index][0],
                            total_magnitudes[index][1],
                            effective_radii[index][0],
                            effective_radii[index][1],
                            axis_ratios[index][0],
                            axis_ratios[index][1],
                            position_angles[index][0],
                            position_angles[index][1],
                            subtract[index]))


def create_galfit_inputfile(input_galfit_filename, sci_image_filename, output_model_image_filename,
                            sigma_image_filename, psf_image_filename, psf_sampling_factor, bad_pixel_mask_filename,
                            constraints_file_filename, image_size, convolution_box_size, magnitude_zeropoint,
                            pixel_scale, light_profiles, source_positions, total_magnitudes, effective_radii,
                            sersic_indices, axis_ratios, position_angles, subtract, initial_background_value,
                            background_x_gradient, background_y_gradient, background_subtraction,
                            display_type='regular', options='0'):
    """
    This function automatizes the GALFIT input file creation.

    :param input_galfit_filename:
    :param sci_image_filename:
    :param output_model_image_filename:
    :param sigma_image_filename:
    :param psf_image_filename:
    :param psf_sampling_factor:
    :param bad_pixel_mask_filename:
    :param constraints_file_filename:
    :param image_size:
    :param convolution_box_size:
    :param magnitude_zeropoint:
    :param pixel_scale:
    :param light_profiles:
    :param source_positions:
    :param total_magnitudes:
    :param effective_radii:
    :param sersic_indices:
    :param axis_ratios:
    :param position_angles:
    :param subtract:
    :param initial_background_value:
    :param background_x_gradient:
    :param background_y_gradient:
    :param background_subtraction:
    :param display_type:
    :param options:
    :return:
    """

    write_to_galfit_inputfile_switcher = {'sersic': write_sersic_to_galfit_inputfile,
                                          'devauc': write_devauc_to_galfit_inputfile,
                                          'expdisk': write_expdisk_to_galfit_inputfile}

    with open(input_galfit_filename, 'w') as f:
        f.write('================================================================================ \n\n'
                '# IMAGE and GALFIT CONTROL PARAMETERS \n'
                'A) {} # Input data image (FITS file) \n'
                'B) {} # Output data image block \n'
                'C) {} # Sigma image name (made from data if blank or "none") \n'
                'D) {} # Input PSF image and (optional) diffusion kernel \n'
                'E) {} # PSF fine sampling factor relative to data \n'
                'F) {} # Bad pixel mask (FITS image or ASCII coord list) \n'
                'G) {} # File with parameter constraints (ASCII file) \n'
                'H) 1 {} 1 {} # Image region to fit (xmin xmax ymin ymax) \n'
                'I) {} {} # Size of the convolution box (x y) \n'
                'J) {} # Magnitude photometric zeropoint \n'
                'K) {} {} # Plate scale (dx dy)   [arcsec per pixel] \n'
                'O) {} # Display type (regular, curses, both) \n'
                'P) {} # Options: 0=normal run; 1,2=make model/imgblock & quit \n\n\n'
                '# INITIAL FITTING PARAMETERS \n\n'.format(sci_image_filename,
                                                           output_model_image_filename,
                                                           sigma_image_filename,
                                                           psf_image_filename,
                                                           psf_sampling_factor,
                                                           bad_pixel_mask_filename,
                                                           constraints_file_filename,
                                                           int(image_size[0]), int(image_size[1]),
                                                           convolution_box_size, convolution_box_size,
                                                           magnitude_zeropoint, pixel_scale, pixel_scale,
                                                           display_type, options))
        f.write('# sky estimate \n'
                '0) sky # object type \n'
                '1) {} {} # sky background at center of fitting region [ADUs] \n'
                '2) {} {} #  dsky/dx (sky gradient in x) \n'
                '3) {} {} #  dsky/dy (sky gradient in y) \n'
                'Z) {} # output option (0 = resid., 1 = Do not subtract) \n\n'.format(initial_background_value[0],
                                                                                      initial_background_value[1],
                                                                                      background_x_gradient[0],
                                                                                      background_x_gradient[1],
                                                                                      background_y_gradient[0],
                                                                                      background_y_gradient[1],
                                                                                      background_subtraction))
        for i in range(len(light_profiles)):
            write_to_galfit_inputfile_function = write_to_galfit_inputfile_switcher.get(light_profiles[i],
                                                                                        lambda: 'To be implemented...')
            write_to_galfit_inputfile_function(f, i, source_positions, total_magnitudes, effective_radii,
                                               sersic_indices, axis_ratios, position_angles, subtract)

        f.write('================================================================================\n\n')
        f.close()


def run_galfit(galfit_binary_file, input_galfit_filename, working_directory, local_or_cluster='local'):
    """
    This function runs GALFIT from the command line.

    :param galfit_binary_file:
    :param input_galfit_filename:
    :param working_directory:
    :param local_or_cluster:
    :return None.
    """

    if local_or_cluster == 'local':
        current_directory = os.getcwd()
    elif local_or_cluster == 'cluster':
        current_directory = os.environ['TMPDIR']
    else:
        raise KeyError

    os.chdir(working_directory)

    subprocess.run(['cp', galfit_binary_file, working_directory])
    subprocess.run([os.path.join(working_directory, os.path.basename(galfit_binary_file)),
                    os.path.basename(input_galfit_filename)])

    os.chdir(current_directory)


def format_sky_subtraction(background_estimate_method, background_value):
    """
    This function reads the sky background from the parameters table.

    :param background_estimate_method:
    :param background_value:
    :return initial_background_value, background_x_gradient, background_y_gradient,
     background_subtraction: initial background parameters for GALFIT.
    """

    if background_estimate_method == 'background_free_fit':
        initial_background_value = np.array([background_value, 1])
        background_x_gradient = np.array([0, 1])
        background_y_gradient = np.array([0, 1])
        background_subtraction = 0
    elif background_estimate_method == 'background_fixed_value':
        initial_background_value = np.array([background_value, 0])
        background_x_gradient = np.array([0, 0])
        background_y_gradient = np.array([0, 0])
        background_subtraction = 0
    else:
        logger.info('not implemented')
        raise ValueError

    return initial_background_value, background_x_gradient, background_y_gradient, background_subtraction


def create_constraints_file_for_galfit(constraints_file_path, n_galaxies):
    """

    :param constraints_file_path:
    :param n_galaxies:
    :return:
    """

    if constraints_file_path == 'None':
        pass
    else:
        with open(constraints_file_path, 'w') as f:
            f.write('# Component/    parameter   constraint	Comment \n'
                    '# operation	(see below)   range \n\n')
            for i in range(n_galaxies):
                f.write('{} x -5 5 \n'
                        '{} y -5 5 \n'
                        '{} mag 12 to 32 \n'
                        '{} re 0.2 to 200 \n'
                        '{} n 0.2 to 10 \n'
                        '{} q 0.02 to 1 \n'.format(i + 1, i + 1, i + 1, i + 1, i + 1, i + 1))
