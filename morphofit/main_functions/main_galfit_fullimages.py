#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
import h5py
import argparse
import subprocess
from astropy.table import Table
from astropy.io import fits

# morphofit imports
from morphofit.catalogue_managing import get_best_fit_parameters_from_model_image
from morphofit.utils import get_logger, compress_and_copy_files_for_galfit
from morphofit.image_utils import create_bad_pixel_mask_for_galfit, create_sigma_image_for_galfit
from morphofit.background_estimation import local_background_estimate
from morphofit.galfit_utils import create_constraints_file_for_galfit, format_sky_subtraction
from morphofit.galfit_utils import format_properties_for_galfit, create_galfit_inputfile, run_galfit
from morphofit.utils import save_best_fit_properties, save_galfit_output_files
from morphofit.plot_utils import create_diagnostic_images, create_diagnostic_pixel_counts_histogram
from morphofit.plot_utils import create_gaussian_fit_residual_image_counts, create_best_fitting_photometry_comparison

logger = get_logger(__file__)


def galfit_on_fullimages(args, telescope_name, target_field_name, waveband, sci_image_filename,
                         seg_image_filename, exposure_time, magnitude_zeropoint, instrumental_gain, pixel_scale,
                         source_galaxies_catalogue, input_galfit_filename, sigma_image_filename, sigma_image_type,
                         background_value, background_estimate_method, output_model_image_filename, psf_image_type,
                         psf_image_filename, psf_sampling_factor, convolution_box_size, galfit_binary_file,
                         best_fit_properties_table_filename, rms_image_filename=None,
                         exp_image_filename=None, constraints_file_filename='None',
                         working_directory=os.getcwd(), output_directory=os.getcwd()):
    """

    :param args:
    :param telescope_name:
    :param target_field_name:
    :param waveband:
    :param sci_image_filename:
    :param seg_image_filename:
    :param exposure_time:
    :param magnitude_zeropoint:
    :param instrumental_gain:
    :param pixel_scale:
    :param source_galaxies_catalogue:
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
    :param best_fit_properties_table_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param constraints_file_filename:
    :param working_directory:
    :param output_directory:
    :return:
    """

    logger.info('=============================== {}, {}, {}, {}, {}'.format(target_field_name, waveband,
                                                                            psf_image_type, sigma_image_type,
                                                                            background_estimate_method))

    logger.info('=============================== create bad pixel mask for GALFIT')
    source_galaxies_keys = args.source_galaxies_keys.split(',')
    bad_pixel_mask_filename = create_bad_pixel_mask_for_galfit(source_galaxies_catalogue, seg_image_filename,
                                                               id_key_sources_catalogue=source_galaxies_keys[0])

    logger.info('=============================== local background estimate')
    print(sci_image_filename, seg_image_filename, background_value)
    local_background_value = local_background_estimate(sci_image_filename, seg_image_filename, background_value,
                                                       local_estimate=True)

    logger.info('=============================== create sigma image for GALFIT')
    magnitude_zeropoint, local_background_value = create_sigma_image_for_galfit(telescope_name,
                                                                                sigma_image_filename,
                                                                                sci_image_filename,
                                                                                rms_image_filename,
                                                                                exp_image_filename,
                                                                                sigma_image_type,
                                                                                local_background_value,
                                                                                exposure_time,
                                                                                magnitude_zeropoint,
                                                                                instrumental_gain)

    logger.info('=============================== format sky subtraction for GALFIT')
    initial_background_value, background_x_gradient, background_y_gradient, background_subtraction = \
        format_sky_subtraction(background_estimate_method, local_background_value)

    logger.info('=============================== format properties for GALFIT')
    light_profiles, source_positions, ra, dec, total_magnitudes, effective_radii, sersic_indices, axis_ratios, \
        position_angles, subtract = format_properties_for_galfit(sci_image_filename, source_galaxies_catalogue,
                                                                 source_galaxies_keys, waveband)

    n_fitted_components = len(light_profiles)

    logger.info('=============================== create soft constraints file for GALFIT')
    create_constraints_file_for_galfit(constraints_file_filename, n_fitted_components)

    header = fits.getheader(sci_image_filename)
    image_size_x = int(header['NAXIS1'])
    image_size_y = int(header['NAXIS2'])
    image_size = [image_size_x, image_size_y]

    logger.info('=============================== create GALFIT input file')
    create_galfit_inputfile(input_galfit_filename, os.path.basename(sci_image_filename),
                            os.path.basename(output_model_image_filename),
                            os.path.basename(sigma_image_filename), os.path.basename(psf_image_filename),
                            psf_sampling_factor,
                            os.path.basename(bad_pixel_mask_filename), os.path.basename(constraints_file_filename),
                            image_size, convolution_box_size, magnitude_zeropoint, pixel_scale, light_profiles,
                            source_positions, total_magnitudes, effective_radii,
                            sersic_indices, axis_ratios, position_angles, subtract,
                            initial_background_value, background_x_gradient, background_y_gradient,
                            background_subtraction, display_type=args.display_type_galfit, options=args.galfit_options)

    logger.info('=============================== run GALFIT')
    run_galfit(galfit_binary_file, input_galfit_filename, working_directory, local_or_cluster=args.local_or_cluster)

    logger.info('=============================== get best-fitting parameters from model image')
    best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, best_fit_effective_radii, \
        best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles, best_fit_background_value, \
        best_fit_background_x_gradient, best_fit_background_y_gradient, reduced_chisquare = \
        get_best_fit_parameters_from_model_image(output_model_image_filename, n_fitted_components, light_profiles)

    logger.info('=============================== save best-fitting parameters table')
    save_best_fit_properties(best_fit_properties_table_filename, source_galaxies_catalogue,
                             target_field_name, waveband, telescope_name, source_galaxies_keys[0],
                             source_galaxies_keys[-2], source_galaxies_keys[-1], light_profiles,
                             psf_image_type, sigma_image_type, background_estimate_method,
                             best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                             best_fit_total_magnitudes,
                             best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios,
                             best_fit_position_angles, best_fit_background_value,
                             best_fit_background_x_gradient,
                             best_fit_background_y_gradient, reduced_chisquare)

    logger.info('=============================== save files to output directory')
    save_galfit_output_files(output_directory, sci_image_filename, input_galfit_filename,
                             output_model_image_filename, sigma_image_filename, bad_pixel_mask_filename,
                             best_fit_properties_table_filename)


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

        root_target_field = h5table['root_target_fields'][()][index].decode('utf8')
        sci_image_filename = os.path.join(temp_dir, h5table['sci_image_filenames'][()][index].decode('utf8'))
        rms_image_filename = os.path.join(temp_dir, h5table['rms_image_filenames'][()][index].decode('utf8'))
        exp_image_filename = os.path.join(temp_dir, h5table['exp_image_filenames'][()][index].decode('utf8'))
        seg_image_filename = os.path.join(temp_dir, h5table['seg_image_filenames'][()][index].decode('utf8'))
        source_galaxies_catalogue_filename = os.path.join(temp_dir,
                                                          h5table['source_galaxies_catalogue_filenames'][()][index].
                                                          decode('utf8'))

        psf_image_filename = os.path.join(temp_dir, h5table['psf_image_filenames'][()][index].decode('utf8'))
        constraints_file_filename = os.path.join(temp_dir,
                                                 h5table['constraints_file_filenames'][()][index].decode('utf8'))

        rms_image_filename, exp_image_filename, constraints_file_filename = \
            compress_and_copy_files_for_galfit(index, root_target_field, temp_dir, args.files_archive_prefix,
                                               sci_image_filename, rms_image_filename,
                                               exp_image_filename,
                                               seg_image_filename, source_galaxies_catalogue_filename,
                                               psf_image_filename, constraints_file_filename)

        source_galaxies_catalogue = Table.read(source_galaxies_catalogue_filename, format='fits', memmap=True)

        target_field_name = h5table['target_field_names'][()][index].decode('utf8')
        waveband = h5table['wavebands'][()][index].decode('utf8')
        telescope_name = h5table['telescope_names'][()][index].decode('utf8')
        exposure_time = h5table['exposure_times'][()][index]
        magnitude_zeropoint = h5table['magnitude_zeropoints'][()][index]
        instrumental_gain = h5table['instrumental_gains'][()][index]
        background_value = h5table['background_values'][()][index]

        sigma_image_type = h5table['sigma_image_types'][()][index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'][()][index].decode('utf8')
        psf_image_type = h5table['psf_image_types'][()][index].decode('utf8')

        input_galfit_filename = os.path.join(temp_dir, h5table['input_galfit_filenames'][()][index].decode('utf8'))
        sigma_image_filename = os.path.join(temp_dir, h5table['sigma_image_filenames'][()][index].decode('utf8'))
        output_model_image_filename = os.path.join(temp_dir,
                                                   h5table['output_model_image_filenames'][()][index].decode('utf8'))

        pixel_scale = h5table['pixel_scale'][()]
        psf_sampling_factor = h5table['psf_sampling_factors'][()][index]
        convolution_box_size = h5table['convolution_box_size'][()]
        galfit_binary_file = h5table['galfit_binary_file'][()].decode('utf8')
        h5table.close()

        best_fit_properties_table_filename = os.path.join(temp_dir, '{}_{}_{}_{}_{}_{}.fits'
                                                          .format(telescope_name, target_field_name, waveband,
                                                                  psf_image_type, sigma_image_type,
                                                                  background_estimate_method))

        os.makedirs(os.path.join(root_target_field, 'full_images/{}_{}_{}'.format(psf_image_type, sigma_image_type,
                                                                                  background_estimate_method)),
                    exist_ok=True)
        output_dir = os.path.join(root_target_field, 'full_images/{}_{}_{}'.format(psf_image_type, sigma_image_type,
                                                                                   background_estimate_method))

        galfit_on_fullimages(args, telescope_name, target_field_name, waveband, sci_image_filename,
                             seg_image_filename, exposure_time, magnitude_zeropoint, instrumental_gain,
                             pixel_scale, source_galaxies_catalogue,
                             input_galfit_filename, sigma_image_filename, sigma_image_type, background_value,
                             background_estimate_method, output_model_image_filename, psf_image_type,
                             psf_image_filename, psf_sampling_factor, convolution_box_size, galfit_binary_file,
                             best_fit_properties_table_filename, rms_image_filename=rms_image_filename,
                             exp_image_filename=exp_image_filename,
                             constraints_file_filename=constraints_file_filename, working_directory=temp_dir,
                             output_directory=output_dir)

        if args.diagnostic_plots == 'True':
            logger.info('=============================== plot diagnostics')
            try:
                source_galaxies_keys = args.source_galaxies_keys.split(',')
                create_diagnostic_images(output_model_image_filename, output_dir, color_map='jet')
                create_diagnostic_pixel_counts_histogram(output_model_image_filename, output_dir)
                create_gaussian_fit_residual_image_counts(output_model_image_filename, output_dir)
                create_best_fitting_photometry_comparison(best_fit_properties_table_filename,
                                                          source_galaxies_catalogue,
                                                          waveband, pixel_scale, source_galaxies_keys[1] +
                                                          '_{}'.format(args.source_galaxies_reference_waveband),
                                                          source_galaxies_keys[2] +
                                                          '_{}'.format(args.source_galaxies_reference_waveband),
                                                          source_galaxies_keys[0], source_galaxies_keys[-2],
                                                          source_galaxies_keys[-1], args.phot_apertures, output_dir)
            except Exception as e:
                logger.info(e)
                pass

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

    h5pytable_filename = os.path.join(args.h5pytable_folder, args.h5pytable_filename)

    h5table = h5py.File(h5pytable_filename, 'r')

    for index in indices:

        current_is_missing = False

        root_target_field = h5table['root_target_fields'][()][index].decode('utf8')
        sigma_image_type = h5table['sigma_image_types'][()][index].decode('utf8')
        background_estimate_method = h5table['background_estimate_methods'][()][index].decode('utf8')
        psf_image_type = h5table['psf_image_types'][()][index].decode('utf8')

        output_directory = os.path.join(root_target_field, 'full_images/{}_{}_{}'.format(psf_image_type,
                                                                                         sigma_image_type,
                                                                                         background_estimate_method))

        output_model_image_filename = os.path.join(output_directory,
                                                   h5table['output_model_image_filenames'][()][index].decode('utf8'))

        try:
            fits.getdata(output_model_image_filename, ext=2)
            logger.info('Model image successfully created')
        except Exception as errmsg:
            logger.error('error opening catalogue: errmsg: %s' % errmsg)
            current_is_missing = True

        if current_is_missing:
            list_missing.append(index)
            logger.info('%d catalogue missing' % index)
        else:
            logger.debug('%d tile all OK' % index)

    h5table.close()

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

    description = "Run GALFIT on images containing target galaxies in Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('--h5pytable_folder', type=str, action='store', default=cwd,
                        help='h5py table folder')
    parser.add_argument('--h5pytable_filename', type=str, action='store', default='table_galfit_on_fullimages_run.h5',
                        help='h5py table filename')
    parser.add_argument('--temp_dir_path', type=str, action='store', default=cwd,
                        help='temporary folder where to make calculations locally, used only if --local_or_cluster'
                             ' is set to local')
    parser.add_argument('--files_archive_prefix', type=str, action='store', default='galfit_res',
                        help='tar file prefix to fast copy data to temporary directory')
    parser.add_argument('--source_galaxies_keys', type=str, action='store', default='NUMBER,ALPHAWIN_J2000,'
                                                                                    'DELTAWIN_J2000,MAG_GALFIT,'
                                                                                    'RE_GALFIT,N_GALFIT,AR_GALFIT,'
                                                                                    'AR_GALFIT,PA_GALFIT,TOFIT,'
                                                                                    'COMPONENT_NUMBER,LIGHT_PROFILE',
                        help='table header keys for source galaxies, keys must be in the following order: ra,dec')
    parser.add_argument('--source_galaxies_reference_waveband', type=str, action='store', default='f814w',
                        help='Reference wavelength for galaxy positions')
    parser.add_argument('--display_type_galfit', type=str, action='store', default='regular',
                        help='Display type (regular, curses, both)')
    parser.add_argument('--galfit_options', type=str, action='store', default='0',
                        help='Options: 0=normal run; 1,2=make model/imgblock & quit')
    parser.add_argument('--diagnostic_plots', type=str, action='store', default='True',
                        help='True to plot diagnostics')
    parser.add_argument('--phot_apertures', type=str, action='store', default='5,10,15,20,25',
                        help='Aperture diameters of SExtractor photometry in pixels')
    parser.add_argument('--local_or_cluster', type=str, action='store', default='local',
                        help='system type: local machine or hpc')
    args = parser.parse_args(args)

    return args
