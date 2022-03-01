#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
import argparse
import h5py
from astropy.table import Table
import numpy as np
from astropy.io import fits

# morphofit imports
from morphofit.image_utils import cut_stamp, create_bad_pixel_mask_for_stamps, create_sigma_image_for_galfit
from morphofit.background_estimation import local_background_estimate
from morphofit.catalogue_managing import find_neighbouring_galaxies_in_stamps
from morphofit.catalogue_managing import get_single_sersic_best_fit_parameters_from_model_image
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_sky_subtraction, format_properties_for_stamps_galfit_single_sersic
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import create_constraints_file_for_galfit, create_galfit_inputfile, run_galfit
from morphofit.utils import save_best_fit_properties_h5table, get_logger

logger = get_logger(__file__)


def galfit_on_stamps(stamp_index, telescope_name, target_name, waveband, sci_image_filename,
                     seg_image_filename, exposure_time, magnitude_zeropoint, instrumental_gain,
                     target_galaxies_id, target_galaxies_x, target_galaxies_y, target_galaxies_ra, target_galaxies_dec,
                     target_galaxies_magnitude,  target_galaxies_effective_radius,
                     target_galaxies_reference_effective_radius, target_galaxies_minor_axis,
                     target_galaxies_major_axis, target_galaxies_position_angles,
                     enlarging_image_factor, enlarging_separation_factor,
                     pixel_scale, sources_catalogue, ra_key_sources_catalogue, dec_key_sources_catalogue,
                     ra_key_neighbouring_sources_catalogue, dec_key_neighbouring_sources_catalogue,
                     mag_key_neighbouring_sources_catalogue, re_key_neighbouring_sources_catalogue,
                     minor_axis_key_neighbouring_sources_catalogue, major_axis_key_neighbouring_sources_catalogue,
                     pos_angle_key_neighbouring_sources_catalogue, id_key_neighbouring_sources_catalogue,
                     input_galfit_filename, sigma_image_filename, sigma_image_type, background_value,
                     background_estimate_method, output_model_image_filename, psf_image_type, psf_image_filename,
                     psf_sampling_factor, convolution_box_size, galfit_binary_file,
                     best_fit_properties_h5table_filename, rms_image_filename=None, exp_image_filename=None,
                     constraints_file_filename=None, working_directory=os.getcwd()):
    """

    :param stamp_index:
    :param telescope_name:
    :param target_name:
    :param waveband:
    :param sci_image_filename:
    :param seg_image_filename:
    :param exposure_time:
    :param magnitude_zeropoint:
    :param instrumental_gain:
    :param target_galaxies_id:
    :param target_galaxies_x:
    :param target_galaxies_y:
    :param target_galaxies_ra:
    :param target_galaxies_dec:
    :param target_galaxies_magnitude:
    :param target_galaxies_effective_radius:
    :param target_galaxies_reference_effective_radius:
    :param target_galaxies_minor_axis:
    :param target_galaxies_major_axis:
    :param target_galaxies_position_angles:
    :param enlarging_image_factor:
    :param enlarging_separation_factor:
    :param pixel_scale:
    :param sources_catalogue:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param pos_angle_key_neighbouring_sources_catalogue:
    :param id_key_neighbouring_sources_catalogue:
    :param input_galfit_filename:
    :param sigma_image_filename:
    :param sigma_image_type:
    :param background_value:
    :param background_estimate_method:
    :param output_model_image_filename:
    :param psf_image_type:
    :param psf_image_filename:
    :param psf_sampling_factor:
    :param convolution_box_size:
    :param galfit_binary_file:
    :param best_fit_properties_h5table_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param constraints_file_filename:
    :param working_directory:
    :return:
    """

    logger.info('=============================== {}, {}, stamp {}, {}, {}, {}'.format(target_name, waveband,
                                                                                      stamp_index,
                                                                                      psf_image_type,
                                                                                      sigma_image_type,
                                                                                      background_estimate_method))

    logger.info('=============================== cut stamp around target galaxy')
    sci_image_stamp_filename, rms_image_stamp_filename, seg_image_stamp_filename, exp_image_stamp_filename =\
        cut_stamp(stamp_index, sci_image_filename, rms_image_filename, seg_image_filename,
                  exp_image_filename, target_galaxies_x, target_galaxies_y,
                  target_galaxies_reference_effective_radius,
                  enlarging_image_factor, target_galaxies_minor_axis,
                  target_galaxies_major_axis, target_galaxies_position_angles)

    logger.info('=============================== find neighbouring galaxies in stamp')
    neighbouring_sources_catalogue = \
        find_neighbouring_galaxies_in_stamps(target_galaxies_ra,
                                             target_galaxies_dec,
                                             enlarging_separation_factor,
                                             target_galaxies_reference_effective_radius,
                                             target_galaxies_minor_axis,
                                             target_galaxies_major_axis,
                                             target_galaxies_position_angles,
                                             pixel_scale,
                                             sources_catalogue,
                                             ra_key_sources_catalogue,
                                             dec_key_sources_catalogue)

    logger.info('=============================== create bad pixel mask for GALFIT')
    bad_pixel_mask_filename = create_bad_pixel_mask_for_stamps(target_galaxies_id,
                                                               neighbouring_sources_catalogue,
                                                               id_key_neighbouring_sources_catalogue,
                                                               seg_image_stamp_filename)

    logger.info('=============================== background estimate in stamp')
    local_background_value = local_background_estimate(sci_image_stamp_filename, seg_image_stamp_filename,
                                                       background_value, local_estimate=True)

    logger.info('=============================== create sigma image for GALFIT')
    magnitude_zeropoint, local_background_value = create_sigma_image_for_galfit(telescope_name,
                                                                                sigma_image_filename,
                                                                                sci_image_stamp_filename,
                                                                                rms_image_stamp_filename,
                                                                                exp_image_stamp_filename,
                                                                                sigma_image_type,
                                                                                local_background_value,
                                                                                exposure_time,
                                                                                magnitude_zeropoint,
                                                                                instrumental_gain)

    logger.info('=============================== format sky subtraction for GALFIT')
    initial_background_value, background_x_gradient, background_y_gradient, background_subtraction = \
        format_sky_subtraction(background_estimate_method, local_background_value)

    logger.info('=============================== format properties in single Sersic fit for GALFIT')
    light_profiles, source_positions, ra, dec, total_magnitudes, effective_radii, sersic_indices, axis_ratios, \
        position_angles, subtract =  \
        format_properties_for_stamps_galfit_single_sersic(sci_image_stamp_filename, waveband,
                                                          target_galaxies_ra, target_galaxies_dec,
                                                          target_galaxies_magnitude,
                                                          target_galaxies_effective_radius,
                                                          target_galaxies_minor_axis,
                                                          target_galaxies_major_axis,
                                                          target_galaxies_position_angles,
                                                          neighbouring_sources_catalogue,
                                                          ra_key_neighbouring_sources_catalogue,
                                                          dec_key_neighbouring_sources_catalogue,
                                                          mag_key_neighbouring_sources_catalogue,
                                                          re_key_neighbouring_sources_catalogue,
                                                          minor_axis_key_neighbouring_sources_catalogue,
                                                          major_axis_key_neighbouring_sources_catalogue,
                                                          pos_angle_key_neighbouring_sources_catalogue,
                                                          enlarging_image_factor)

    logger.info('=============================== create soft constraints file for GALFIT')
    create_constraints_file_for_galfit(constraints_file_filename, len(neighbouring_sources_catalogue) + 1)

    axis_ratio = target_galaxies_minor_axis / target_galaxies_major_axis
    angle = target_galaxies_position_angles * (2 * np.pi) / 360

    image_size_x = target_galaxies_reference_effective_radius * enlarging_image_factor * (abs(np.cos(angle)) +
                                                                                          axis_ratio *
                                                                                          abs(np.sin(angle)))
    image_size_y = target_galaxies_reference_effective_radius * enlarging_image_factor * (abs(np.sin(angle)) +
                                                                                          axis_ratio *
                                                                                          abs(np.cos(angle)))
    image_size = [image_size_x, image_size_y]

    logger.info('=============================== create GALFIT input file')
    create_galfit_inputfile(input_galfit_filename, sci_image_stamp_filename, output_model_image_filename,
                            sigma_image_filename, psf_image_filename, psf_sampling_factor,
                            bad_pixel_mask_filename, constraints_file_filename, image_size,
                            convolution_box_size, magnitude_zeropoint, pixel_scale, light_profiles,
                            source_positions, total_magnitudes, effective_radii,
                            sersic_indices, axis_ratios, position_angles, subtract,
                            initial_background_value, background_x_gradient, background_y_gradient,
                            background_subtraction, display_type='regular', options='0')

    logger.info('=============================== run GALFIT')

    run_galfit(galfit_binary_file, input_galfit_filename,
               sci_image_stamp_filename, output_model_image_filename,
               sigma_image_filename, psf_image_filename,
               bad_pixel_mask_filename, constraints_file_filename, working_directory)

    logger.info('=============================== get best-fitting parameters from model image')
    best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, best_fit_effective_radii,\
        best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles, best_fit_background_value, \
        best_fit_background_x_gradient, best_fit_background_y_gradient, reduced_chisquare = \
        get_single_sersic_best_fit_parameters_from_model_image(output_model_image_filename,
                                                               len(neighbouring_sources_catalogue)
                                                               + 1)

    logger.info('=============================== save best-fitting parameters table')
    save_best_fit_properties_h5table(best_fit_properties_h5table_filename, light_profiles,
                                     psf_image_type, sigma_image_type, background_estimate_method,
                                     best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                                     best_fit_total_magnitudes,
                                     best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios,
                                     best_fit_position_angles, best_fit_background_value,
                                     best_fit_background_x_gradient, best_fit_background_y_gradient,
                                     reduced_chisquare)

    out_dir = os.path.dirname(best_fit_properties_h5table_filename)
    neighbouring_sources_catalogue.write('{}/neighbouring_sources_catalogue.fits'.format(out_dir), format='fits',
                                         overwrite=True)
    os.system('cp {} {}'.format(sci_image_stamp_filename, out_dir))
    os.system('cp {} {}'.format(input_galfit_filename, out_dir))
    os.system('cp {} {}'.format(output_model_image_filename, out_dir))
    os.system('cp {} {}'.format(sigma_image_filename, out_dir))
    os.system('cp {} {}'.format(bad_pixel_mask_filename, out_dir))
    os.system('rm -rf {}'.format(working_directory))


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

        root_files = h5table['root_files'].value[index].decode('utf8')
        target_name = h5table['target_names'].value[index].decode('utf8')
        waveband = h5table['wavebands'].value[index].decode('utf8')
        stamp_index = h5table['stamp_indices'].value[index]
        telescope_name = h5table['telescope_names'].value[index].decode('utf8')
        exposure_time = h5table['exposure_times'].value[index]
        magnitude_zeropoint = h5table['magnitude_zeropoints'].value[index]
        instrumental_gain = h5table['instrumental_gains'].value[index]
        background_value = h5table['background_values'].value[index]
        target_galaxies_id = h5table['target_galaxies_id'].value[index]
        target_galaxies_x = h5table['target_galaxies_x'].value[index]
        target_galaxies_y = h5table['target_galaxies_y'].value[index]
        target_galaxies_ra = h5table['target_galaxies_ra'].value[index]
        target_galaxies_dec = h5table['target_galaxies_dec'].value[index]
        target_galaxies_magnitude = h5table['target_galaxies_magnitudes'].value[index]
        target_galaxies_effective_radius = h5table['target_galaxies_effective_radii'].value[index]
        target_galaxies_reference_effective_radius = h5table['target_galaxies_reference_effective_radii'].value[index]
        target_galaxies_minor_axis = h5table['target_galaxies_minor_axis'].value[index]
        target_galaxies_major_axis = h5table['target_galaxies_major_axis'].value[index]
        target_galaxies_position_angles = h5table['target_galaxies_position_angles'].value[index]
        enlarging_image_factor = h5table['enlarging_image_factor'].value
        enlarging_separation_factor = h5table['enlarging_separation_factor'].value
        pixel_scale = h5table['pixel_scale'].value
        ra_key_sources_catalogue = h5table['ra_key_sources_catalogue'].value.decode('utf8')
        dec_key_sources_catalogue = h5table['dec_key_sources_catalogue'].value.decode('utf8')
        ra_key_neighbouring_sources_catalogue = h5table['ra_key_neighbouring_sources_catalogue'].value.decode('utf8')
        dec_key_neighbouring_sources_catalogue = h5table['dec_key_neighbouring_sources_catalogue'].value.decode('utf8')
        mag_key_neighbouring_sources_catalogue = h5table['mag_key_neighbouring_sources_catalogue'].value.decode('utf8')
        re_key_neighbouring_sources_catalogue = h5table['re_key_neighbouring_sources_catalogue'].value.decode('utf8')
        minor_axis_key_neighbouring_sources_catalogue = \
            h5table['minor_axis_key_neighbouring_sources_catalogue'].value.decode('utf8')
        major_axis_key_neighbouring_sources_catalogue = \
            h5table['major_axis_key_neighbouring_sources_catalogue'].value.decode('utf8')
        position_angle_key_neighbouring_sources_catalogue = \
            h5table['position_angle_key_neighbouring_sources_catalogue'].value.decode('utf8')
        id_key_neighbouring_sources_catalogue = h5table['id_key_neighbouring_sources_catalogue'].value.decode('utf8')
        sigma_image_type = h5table['sigma_image_types'].value[index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'].value[index].decode('utf8')
        psf_image_type = h5table['psf_image_types'].value[index].decode('utf8')
        psf_sampling_factor = h5table['psf_sampling_factors'].value[index]
        convolution_box_size = h5table['convolution_box_size'].value
        galfit_binary_file = h5table['galfit_binary_file'].value.decode('utf8')

        os.makedirs(root_files + 'stamps/stamp{}_{}_{}_{}'.format(stamp_index, psf_image_type, sigma_image_type,
                                                                  background_estimate_method), exist_ok=True)
        # os.chdir(root_files + 'stamps/stamp{}_{}_{}_{}'.format(stamp_index, psf_image_type, sigma_image_type,
        #                                                        background_estimate_method))
        output_dir = root_files + 'stamps/stamp{}_{}_{}_{}/'.format(stamp_index, psf_image_type, sigma_image_type,
                                                                    background_estimate_method)
        temp_dir = os.path.join(root_files, 'temp_dir')
        os.makedirs(temp_dir)

        # cwd = root_files + 'stamps/stamp{}_{}_{}_{}/'.format(stamp_index, psf_image_type, sigma_image_type,
        #                                                      background_estimate_method)

        sci_image_filename = temp_dir + h5table['sci_image_filenames'].value[index].decode('utf8')
        rms_image_filename = temp_dir + h5table['rms_image_filenames'].value[index].decode('utf8')
        seg_image_filename = temp_dir + h5table['seg_image_filenames'].value[index].decode('utf8')
        exp_image_filename = temp_dir + h5table['exp_image_filenames'].value[index].decode('utf8')
        sources_catalogue_filename = temp_dir + h5table['sources_catalogues'].value[index].decode('utf8')
        input_galfit_filename = temp_dir + h5table['input_galfit_filenames'].value[index].decode('utf8')
        sigma_image_filename = temp_dir + h5table['sigma_image_filenames'].value[index].decode('utf8')
        output_model_image_filename = temp_dir + h5table['output_model_image_filenames'].value[index].decode('utf8')
        psf_image_filename = temp_dir + h5table['psf_image_filenames'].value[index].decode('utf8')
        constraints_file_filename = temp_dir + h5table['constraints_file_filenames'].value[index].decode('utf8')
        if os.path.basename(constraints_file_filename) == 'None':
            constraints_file_filename = None

        # print(root_files,telescope_name,target_name,waveband,stamp_index,
        #      sci_image_filename,rms_image_filename,seg_image_filename,
        #      exp_image_filename,)
        # cwd = os.getcwd()  # os.environ['TMPDIR'] # create dir for stamp locally, no for cluster

        os.system('cp -sf {}{} {}'.format(root_files, os.path.basename(sci_image_filename), temp_dir))
        os.system('cp -sf {}{} {}'.format(root_files, os.path.basename(seg_image_filename), temp_dir))
        os.system('cp -sf {}{} {}'.format(root_files, os.path.basename(sources_catalogue_filename), temp_dir))
        os.system('cp -sf {}stars/{} {}'.format(root_files, os.path.basename(psf_image_filename), temp_dir))

        if os.path.basename(rms_image_filename) == 'None':
            rms_image_filename = None
        else:
            os.system('cp -sf {}{} {}'.format(root_files, os.path.basename(rms_image_filename), temp_dir))
        if os.path.basename(exp_image_filename) == 'None':
            exp_image_filename = None
        else:
            os.system('cp -sf {}{} {}'.format(root_files, os.path.basename(exp_image_filename), temp_dir))

        sources_catalogue = Table.read(sources_catalogue_filename, format='fits', memmap=True)
        best_fit_properties_h5table_filename = output_dir + '{}_{}_{}_{}_{}_{}_{}.h5'.format(telescope_name,
                                                                                             target_name,
                                                                                             waveband, stamp_index,
                                                                                             psf_image_type,
                                                                                             sigma_image_type,
                                                                                             background_estimate_method)

        galfit_on_stamps(stamp_index, telescope_name, target_name, waveband, sci_image_filename,
                         seg_image_filename, exposure_time, magnitude_zeropoint, instrumental_gain,
                         target_galaxies_id, target_galaxies_x, target_galaxies_y, target_galaxies_ra,
                         target_galaxies_dec, target_galaxies_magnitude,
                         target_galaxies_effective_radius, target_galaxies_reference_effective_radius,
                         target_galaxies_minor_axis, target_galaxies_major_axis,
                         target_galaxies_position_angles, enlarging_image_factor, enlarging_separation_factor,
                         pixel_scale, sources_catalogue, ra_key_sources_catalogue, dec_key_sources_catalogue,
                         ra_key_neighbouring_sources_catalogue, dec_key_neighbouring_sources_catalogue,
                         mag_key_neighbouring_sources_catalogue, re_key_neighbouring_sources_catalogue,
                         minor_axis_key_neighbouring_sources_catalogue,
                         major_axis_key_neighbouring_sources_catalogue,
                         position_angle_key_neighbouring_sources_catalogue, id_key_neighbouring_sources_catalogue,
                         input_galfit_filename, sigma_image_filename, sigma_image_type, background_value,
                         background_estimate_method, output_model_image_filename, psf_image_type,
                         psf_image_filename, psf_sampling_factor, convolution_box_size, galfit_binary_file,
                         best_fit_properties_h5table_filename, rms_image_filename=rms_image_filename,
                         exp_image_filename=exp_image_filename,
                         constraints_file_filename=constraints_file_filename, working_directory=temp_dir)

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

        current_is_missing = False

        root_files = h5table['root_files'].value[index].decode('utf8')
        stamp_index = h5table['stamp_indices'].value[index]
        telescope_name = h5table['telescope_names'].value[index].decode('utf8')
        target_name = h5table['target_names'].value[index].decode('utf8')
        waveband = h5table['wavebands'].value[index].decode('utf8')
        psf_image_type = h5table['psf_image_types'].value[index].decode('utf8')
        sigma_image_type = h5table['sigma_image_types'].value[index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'].value[index].decode('utf8')

        cwd = root_files + 'stamps/stamp{}_{}_{}_{}/'.format(stamp_index, psf_image_type, sigma_image_type,
                                                             background_estimate_method)
        best_fit_properties_imgblock_filename = cwd + \
            '{}_{}_{}_stamp{}_{}_{}_{}_imgblock.fits'.format(telescope_name, target_name, waveband, stamp_index,
                                                             psf_image_type, sigma_image_type,
                                                             background_estimate_method)

        try:
            fits.getdata(best_fit_properties_imgblock_filename, ext=2)
            logger.info('Model image successfully created')
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

    description = "Run GALFIT on stamps"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--filename_h5pytable', type=str, action='store', default='table.h5',
                        help='h5py table of the file to run on')
    args = parser.parse_args(args)

    return args.filename_h5pytable
