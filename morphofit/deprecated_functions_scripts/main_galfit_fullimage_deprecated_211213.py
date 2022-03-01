#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import h5py
import argparse
from astropy.io import fits
import os
from astropy.table import Table

# morphofit imports
from morphofit.utils import get_logger, save_best_fit_properties_h5table
from morphofit.background_estimation import local_background_estimate
from morphofit.catalogue_managing import check_parameters_for_next_iteration
from morphofit.catalogue_managing import get_single_sersic_best_fit_parameters_from_model_image
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_properties_for_regions_galfit_single_sersic, create_galfit_inputfile
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import create_constraints_file_for_galfit, format_sky_subtraction, run_galfit
from morphofit.image_utils import create_bad_pixel_mask_for_regions, create_sigma_image_for_galfit

logger = get_logger(__file__)


def galfit_on_fullimage(telescope_name, target_name, waveband, sci_image_filename,
                        seg_image_filename, exposure_time, magnitude_zeropoint, instrumental_gain, pixel_scale,
                        regions_mastercatalogue, id_key_sources_catalogue, input_galfit_filename,
                        sigma_image_filename, sigma_image_type, background_value,
                        background_estimate_method, output_model_image_filename, psf_image_type, psf_image_filename,
                        psf_sampling_factor, convolution_box_size, galfit_binary_file,
                        best_fit_properties_h5table_filename,
                        rms_image_filename=None,
                        exp_image_filename=None,
                        constraints_file_filename=None, check_parameters=False, working_directory=os.getcwd()):
    """

    :param telescope_name:
    :param target_name:
    :param waveband:
    :param sci_image_filename:
    :param seg_image_filename:
    :param exposure_time:
    :param magnitude_zeropoint:
    :param instrumental_gain:
    :param pixel_scale:
    :param regions_mastercatalogue:
    :param id_key_sources_catalogue:
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
    :param check_parameters:
    :param working_directory:
    :return:
    """

    logger.info('=============================== {}, {}, {}, {}, {}'.format(target_name, waveband,
                                                                            psf_image_type, sigma_image_type,
                                                                            background_estimate_method))

    logger.info('=============================== check galaxy parameters')
    if check_parameters:
        regions_mastercatalogue = check_parameters_for_next_iteration(regions_mastercatalogue, waveband,
                                                                      magnitude_keyword='MAG_AUTO',
                                                                      size_keyword='FLUX_RADIUS',
                                                                      minor_axis_keyword='BWIN_IMAGE',
                                                                      major_axis_keyword='AWIN_IMAGE',
                                                                      position_angle_keyword='THETAWIN_SKY',
                                                                      magnitude_error_limit=0.1,
                                                                      magnitude_upper_limit=30,
                                                                      size_error_limit=1, size_upper_limit=30,
                                                                      sersic_index_error_limit=0.1,
                                                                      sersic_index_upper_limit=8,
                                                                      sersic_index_lower_limit=0.3,
                                                                      key='mean_error')

    logger.info('=============================== create bad pixel mask for GALFIT')
    bad_pixel_mask_filename = create_bad_pixel_mask_for_regions(regions_mastercatalogue, seg_image_filename,
                                                                id_key_sources_catalogue=id_key_sources_catalogue)

    logger.info('=============================== background estimate on full image')
    local_background_value = local_background_estimate(sci_image_filename, seg_image_filename,
                                                       background_value, local_estimate=True)

    logger.info('=============================== create sigma image for GALFIT')
    magnitude_zeropoint, local_background_value = create_sigma_image_for_galfit(telescope_name,
                                                                                sigma_image_filename,
                                                                                sci_image_filename,
                                                                                rms_image_filename,
                                                                                exp_image_filename,
                                                                                sigma_image_type,
                                                                                local_background_value,
                                                                                exposure_time,
                                                                                magnitude_zeropoint, instrumental_gain)

    logger.info('=============================== format sky subtraction for GALFIT')
    initial_background_value, background_x_gradient, background_y_gradient, background_subtraction = \
        format_sky_subtraction(background_estimate_method, local_background_value)

    logger.info('=============================== format properties in single Sersic fit for GALFIT')
    light_profiles, source_positions, ra, dec, total_magnitudes, effective_radii, sersic_indices, axis_ratios, \
        position_angles, subtract = \
        format_properties_for_regions_galfit_single_sersic(sci_image_filename, regions_mastercatalogue, waveband,
                                                           ra_keyword='ALPHAWIN_J2000', dec_keyword='DELTAWIN_J2000',
                                                           waveband_keyword='f814w',
                                                           initial_positions_choice='sextractor')

    logger.info('=============================== create soft constraints file for GALFIT')
    create_constraints_file_for_galfit(constraints_file_filename, len(regions_mastercatalogue))

    header = fits.getheader(sci_image_filename)
    image_size_x = int(header['NAXIS1'])
    image_size_y = int(header['NAXIS2'])
    image_size = [image_size_x, image_size_y]

    logger.info('=============================== create GALFIT input file')
    create_galfit_inputfile(input_galfit_filename, sci_image_filename, output_model_image_filename,
                            sigma_image_filename, psf_image_filename, psf_sampling_factor,
                            bad_pixel_mask_filename, constraints_file_filename, image_size,
                            convolution_box_size, magnitude_zeropoint, pixel_scale, light_profiles,
                            source_positions, total_magnitudes, effective_radii,
                            sersic_indices, axis_ratios, position_angles, subtract,
                            initial_background_value, background_x_gradient, background_y_gradient,
                            background_subtraction, display_type='regular', options='0')

    logger.info('=============================== run GALFIT')
    run_galfit(galfit_binary_file, input_galfit_filename,
               sci_image_filename, output_model_image_filename,
               sigma_image_filename, psf_image_filename,
               bad_pixel_mask_filename, constraints_file_filename, working_directory)

    logger.info('=============================== get best-fitting parameters from model image')
    best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, best_fit_effective_radii, \
        best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles, best_fit_background_value, \
        best_fit_background_x_gradient, best_fit_background_y_gradient, reduced_chisquare = \
        get_single_sersic_best_fit_parameters_from_model_image(output_model_image_filename,
                                                               len(regions_mastercatalogue))

    logger.info('=============================== save best-fitting parameters table')
    save_best_fit_properties_h5table(best_fit_properties_h5table_filename, light_profiles,
                                     psf_image_type, sigma_image_type, background_estimate_method,
                                     best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                                     best_fit_total_magnitudes,
                                     best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios,
                                     best_fit_position_angles, best_fit_background_value,
                                     best_fit_background_x_gradient, best_fit_background_y_gradient,
                                     reduced_chisquare)

    # out_dir = os.path.dirname(best_fit_properties_h5table_filename)
    # os.system('cp {} {}'.format(input_galfit_filename, out_dir))
    # os.system('cp {} {}'.format(output_model_image_filename, out_dir))
    # os.system('cp {} {}'.format(sigma_image_filename, out_dir))
    # os.system('cp {} {}'.format(bad_pixel_mask_filename, out_dir))
    # os.system('rm -rf {}'.format(output_model_image_filename))


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
        telescope_name = h5table['telescope_names'].value[index].decode('utf8')
        exposure_time = h5table['exposure_times'].value[index]
        magnitude_zeropoint = h5table['magnitude_zeropoints'].value[index]
        instrumental_gain = h5table['instrumental_gains'].value[index]
        background_value = h5table['background_values'].value[index]
        pixel_scale = h5table['pixel_scale'].value
        id_key_sources_catalogue = h5table['id_key_sources_catalogue'].value.decode('utf8')
        sigma_image_type = h5table['sigma_image_types'].value[index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'].value[index].decode('utf8')
        psf_image_type = h5table['psf_image_types'].value[index].decode('utf8')
        psf_sampling_factor = h5table['psf_sampling_factors'].value[index]
        convolution_box_size = h5table['convolution_box_size'].value
        galfit_binary_file = h5table['galfit_binary_file'].value.decode('utf8')

        os.makedirs(root_files + 'full_image/{}_{}_{}'.format(psf_image_type, sigma_image_type,
                                                              background_estimate_method), exist_ok=True)
        # os.chdir(root_files + 'regions/region{}_{}_{}_{}'.format(region_index, psf_image_type, sigma_image_type,
        #                                                        background_estimate_method))
        cwd = root_files + 'full_image/{}_{}_{}/'.format(psf_image_type, sigma_image_type,
                                                         background_estimate_method)

        sci_image_filename = cwd + h5table['sci_image_filenames'].value[index].decode('utf8')
        rms_image_filename = cwd + h5table['rms_image_filenames'].value[index].decode('utf8')
        seg_image_filename = cwd + h5table['seg_image_filenames'].value[index].decode('utf8')
        exp_image_filename = cwd + h5table['exp_image_filenames'].value[index].decode('utf8')
        regions_mastercatalogue_filename = cwd + \
            h5table['regions_mastercatalogue_filenames'].value[index].decode('utf8')
        input_galfit_filename = cwd + h5table['input_galfit_filenames'].value[index].decode('utf8')
        sigma_image_filename = cwd + h5table['sigma_image_filenames'].value[index].decode('utf8')
        output_model_image_filename = cwd + h5table['output_model_image_filenames'].value[index].decode('utf8')
        psf_image_filename = cwd + h5table['psf_image_filenames'].value[index].decode('utf8')
        constraints_file_filename = cwd + h5table['constraints_file_filenames'].value[index].decode('utf8')
        if os.path.basename(constraints_file_filename) == 'None':
            constraints_file_filename = None

        # print(root_files,telescope_name,target_name,waveband,region_index,
        #      sci_image_filename,rms_image_region_filename,seg_image_region_filename,
        #      exp_image_region_filename,)
        # cwd = os.getcwd()  # os.environ['TMPDIR'] # create dir for region locally, no for cluster

        os.system('ln -sf {}{} {}'.format(root_files, os.path.basename(sci_image_filename), cwd))
        os.system('ln -sf {}{} {}'.format(root_files, os.path.basename(seg_image_filename), cwd))
        os.system('ln -sf {}regions/cats/{} {}'.format(root_files, os.path.basename(regions_mastercatalogue_filename),
                                                       cwd))
        os.system('ln -sf {}stars/{} {}'.format(root_files, os.path.basename(psf_image_filename), cwd))

        if os.path.basename(rms_image_filename) == 'None':
            rms_image_filename = None
        else:
            os.system('ln -sf {}{} {}'.format(root_files, os.path.basename(rms_image_filename), cwd))
        if os.path.basename(exp_image_filename) == 'None':
            exp_image_filename = None
        else:
            os.system('ln -sf {}{} {}'.format(root_files, os.path.basename(exp_image_filename), cwd))

        regions_mastercatalogue = Table.read(regions_mastercatalogue_filename, format='fits', memmap=True)
        out_dir = root_files + 'full_image/{}_{}_{}/'.format(psf_image_type, sigma_image_type,
                                                             background_estimate_method)
        best_fit_properties_h5table_filename = out_dir + '{}_{}_{}_{}_{}_{}.h5'.format(telescope_name, target_name,
                                                                                       waveband,
                                                                                       psf_image_type,
                                                                                       sigma_image_type,
                                                                                       background_estimate_method)

        galfit_on_fullimage(telescope_name, target_name, waveband, sci_image_filename,
                            seg_image_filename, exposure_time, magnitude_zeropoint, instrumental_gain, pixel_scale,
                            regions_mastercatalogue, id_key_sources_catalogue, input_galfit_filename,
                            sigma_image_filename, sigma_image_type, background_value,
                            background_estimate_method, output_model_image_filename, psf_image_type, psf_image_filename,
                            psf_sampling_factor, convolution_box_size, galfit_binary_file,
                            best_fit_properties_h5table_filename,
                            rms_image_filename=rms_image_filename,
                            exp_image_filename=exp_image_filename,
                            constraints_file_filename=constraints_file_filename, check_parameters=True,
                            working_directory=cwd)

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
        telescope_name = h5table['telescope_names'].value[index].decode('utf8')
        target_name = h5table['target_names'].value[index].decode('utf8')
        waveband = h5table['wavebands'].value[index].decode('utf8')
        psf_image_type = h5table['psf_image_types'].value[index].decode('utf8')
        sigma_image_type = h5table['sigma_image_types'].value[index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'].value[index].decode('utf8')

        cwd = root_files + 'full_image/{}_{}_{}/'.format(psf_image_type, sigma_image_type,
                                                         background_estimate_method)
        best_fit_properties_imgblock_filename = cwd + \
            '{}_{}_{}_{}_{}_{}_imgblock.fits'.format(telescope_name, target_name, waveband, psf_image_type,
                                                     sigma_image_type, background_estimate_method)

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

    description = "Run GALFIT on full image"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('--filename_h5pytable', type=str, action='store', default='table.h5',
                        help='h5py table of the file to run on')
    args = parser.parse_args(args)

    return args.filename_h5pytable
