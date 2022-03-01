#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
from astropy.table import Table, Column, hstack, join, unique
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import numpy as np
import os
import itertools
import h5py
from statsmodels.stats.weightstats import DescrStatsW
import pickle
import warnings

# morphofit imports
from morphofit.utils import ra_dec_2_xy
from morphofit.utils import get_logger

logger = get_logger(__file__)
warnings.filterwarnings('ignore')


def psf_corr_mag(forced_catalogue, forced_detect_catalogues):
    """
    This function computes PSF corrected magnitudes.

    :param forced_catalogue:
    :param forced_detect_catalogues:
    :return:
    """
    match_cat = join(forced_catalogue, forced_detect_catalogues, keys='NUMBER')
    corr_mag = (match_cat['MAG_ISO_1'] - match_cat['MAG_ISO_2']) + match_cat['MAG_AUTO_2']
    corr_mag_col = Column(corr_mag, name='MAG_ISO_CORR')
    forced_catalogue.add_column(corr_mag_col)

    return forced_catalogue


def remove_stars(catalogue, wavebands):
    """
    Logical OR of masks where CLASS_STAR greater than 0.95

    :param catalogue:
    :param wavebands:
    :return:
    """

    mask_list = []
    for wave in wavebands:
        if wave != 'f814w':
            mask_list.append((catalogue['CLASS_STAR_{}'.format(wave)] >= 0.95))
    combined_mask = np.array(sum(mask_list), dtype=np.bool)

    return catalogue[~combined_mask]


def clean_catalogue(catalogue, wavebands):
    """
    Remove objects where at least one band has negative FLUX_RADIUS or MAG_AUTO greater than 30.
    This cuts are subject to changes according to the kind of sample one wants to analyze.

    :param catalogue:
    :param wavebands:
    :return:
    """

    mask_list = []
    for wave in wavebands:
        mask_list.append((catalogue['FLUX_RADIUS_{}'.format(wave)] <= 0))
        mask_list.append((catalogue['MAG_AUTO_{}'.format(wave)] > 35))
    combined_mask = np.array(sum(mask_list), dtype=np.bool)

    return catalogue[~combined_mask]


def get_multiband_catalogue(forced_catalogues, forced_detect_catalogue, wavebands):
    """
    This function merges catalogues to create a multiband one.

    :param forced_catalogues:
    :param forced_detect_catalogue:
    :param wavebands:
    :return cat: multiband fits catalogue
    """

    catalogue = Table.read(forced_catalogues[0], format='fits')
    detection_catalogue = Table.read(forced_detect_catalogue)
    catalogue = psf_corr_mag(catalogue, detection_catalogue)
    col_name_list = catalogue.colnames
    col_name_list.remove('NUMBER')
    for j in range(len(col_name_list)):
        catalogue.rename_column(col_name_list[j], col_name_list[j] + '_%s' % wavebands[0])
    for i in range(1, len(wavebands)):
        try:
            catalogue.rename_column('NUMBER_1', 'NUMBER')
        except KeyError:
            pass
        newcatalogue = Table.read(forced_catalogues[i], format='fits')
        newcatalogue = psf_corr_mag(newcatalogue, detection_catalogue)
        col_name_list = newcatalogue.colnames
        col_name_list.remove('NUMBER')
        for j in range(len(col_name_list)):
            newcatalogue.rename_column(col_name_list[j], col_name_list[j] + '_%s' % wavebands[i])
        catalogue = join(catalogue, newcatalogue, keys='NUMBER')

    catalogue_removed_stars = remove_stars(catalogue, wavebands)
    final_catalogue = clean_catalogue(catalogue_removed_stars, wavebands)

    return catalogue, catalogue_removed_stars, final_catalogue


def match_catalogues(cat1, cat2, cat1_ra_key='ALPHAWIN_J2000_f814w', cat1_dec_key='DELTAWIN_J2000_f814w',
                     cat2_ra_key='RA', cat2_dec_key='DEC'):
    """
    This function matches catalogues by coordinates.

    :param cat1:
    :param cat2:
    :param cat1_ra_key:
    :param cat1_dec_key:
    :param cat2_ra_key:
    :param cat2_dec_key:
    :return matchedtable: fits table, matched catalogues.
    """

    catalog = SkyCoord(ra=cat1['{}'.format(cat1_ra_key)],
                       dec=cat1['{}'.format(cat1_dec_key)])
    c = SkyCoord(ra=cat2['{}'.format(cat2_ra_key)] * u.degree, dec=cat2['{}'.format(cat2_dec_key)] * u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    separation = Column(d2d.arcsecond, name='d2d')
    cat1_matched = cat1[idx]
    cat1_matched.add_column(separation)
    matchedtable = hstack([cat1_matched, cat2])
    mask = (matchedtable['d2d'] < 1.0)
    matchedtable = matchedtable[mask]
    matchedtable.remove_column('d2d')

    return matchedtable


def find_neighbouring_galaxies_in_stamps(ra_central_source, dec_central_source, enlarging_separation_factor,
                                         effective_radius_central_source, minor_axis_central_source,
                                         major_axis_central_source, position_angle_central_source,
                                         pixel_scale, sources_catalogue,
                                         ra_key_sources_catalogue, dec_key_sources_catalogue):
    """
    This function finds galaxies which are within an enlarged elliptical aperture from the spectroscopic
    confirmed member.

    :param ra_central_source:
    :param dec_central_source:
    :param enlarging_separation_factor:
    :param effective_radius_central_source:
    :param minor_axis_central_source:
    :param major_axis_central_source:
    :param position_angle_central_source:
    :param pixel_scale:
    :param sources_catalogue:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :return: neighbouring_sources_catalogue: SE catalogues containing neighbouring galaxies.
    """

    axis_ratio = minor_axis_central_source / major_axis_central_source
    angle = position_angle_central_source * (2*np.pi) / 360

    coor_obj_central_source = SkyCoord(ra=ra_central_source * u.degree, dec=dec_central_source * u.degree)
    max_separation_x = effective_radius_central_source * enlarging_separation_factor * (abs(np.cos(angle)) +
                                                                                        axis_ratio *
                                                                                        abs(np.sin(angle)))
    max_separation_y = effective_radius_central_source * enlarging_separation_factor * (abs(np.sin(angle)) +
                                                                                        axis_ratio *
                                                                                        abs(np.cos(angle)))
    max_separation_arcsec = np.sqrt((max_separation_x / 2)**2 + (max_separation_y / 2)**2) * pixel_scale
    index_neighbouring_galaxies = []
    for i in range(len(sources_catalogue)):
        coord_obj_source = SkyCoord(ra=sources_catalogue[i][ra_key_sources_catalogue] * u.degree,
                                    dec=sources_catalogue[i][dec_key_sources_catalogue] * u.degree)
        separation = coor_obj_central_source.separation(coord_obj_source)
        if (separation.arcsecond <= max_separation_arcsec) & (separation.arcsecond > 0.05):
            index_neighbouring_galaxies.append(i)
    neighbouring_sources_catalogue = sources_catalogue[index_neighbouring_galaxies]

    return neighbouring_sources_catalogue


def delete_star_character(array):
    """
    This function deletes the `*' from the output parameters.

    :param array:
    :return:
    """

    a = '*'
    for i in range(len(array)):
        if a in array[i][0]:
            array[i][0] = array[i][0][1:-1]
            array[i][1] = array[i][1][1:-1]
    return array


def get_sersic_parameters_from_header(output_model_image_header, index):
    """

    :param output_model_image_header:
    :param index:
    :return:
    """

    try:
        source_x_position = [output_model_image_header['{}_XC'.format(index + 2)].split(' +/- ')[0],
                             output_model_image_header['{}_XC'.format(index + 2)].split(' +/- ')[1]]
        source_y_position = [output_model_image_header['{}_YC'.format(index + 2)].split(' +/- ')[0],
                             output_model_image_header['{}_YC'.format(index + 2)].split(' +/- ')[1]]
        total_magnitude = [output_model_image_header['{}_MAG'.format(index + 2)].split(' +/- ')[0],
                           output_model_image_header['{}_MAG'.format(index + 2)].split(' +/- ')[1]]
        effective_radius = [output_model_image_header['{}_RE'.format(index + 2)].split(' +/- ')[0],
                            output_model_image_header['{}_RE'.format(index + 2)].split(' +/- ')[1]]
        sersic_index = [output_model_image_header['{}_N'.format(index + 2)].split(' +/- ')[0],
                        output_model_image_header['{}_N'.format(index + 2)].split(' +/- ')[1]]
        axis_ratio = [output_model_image_header['{}_AR'.format(index + 2)].split(' +/- ')[0],
                      output_model_image_header['{}_AR'.format(index + 2)].split(' +/- ')[1]]
        position_angle = [output_model_image_header['{}_PA'.format(index + 2)].split(' +/- ')[0],
                          output_model_image_header['{}_PA'.format(index + 2)].split(' +/- ')[1]]
    except Exception as e:
        logger.info(e)
        try:
            source_x_position = [output_model_image_header['{}_XC'.format(index + 2)][1:-1], 0]
            source_y_position = [output_model_image_header['{}_YC'.format(index + 2)][1:-1], 0]
            total_magnitude = [output_model_image_header['{}_MAG'.format(index + 2)][1:-1], 0]
            effective_radius = [output_model_image_header['{}_RE'.format(index + 2)][1:-1], 0]
            sersic_index = [output_model_image_header['{}_N'.format(index + 2)][1:-1], 0]
            axis_ratio = [output_model_image_header['{}_AR'.format(index + 2)][1:-1], 0]
            position_angle = [output_model_image_header['{}_PA'.format(index + 2)][1:-1], 0]
        except Exception as e:
            logger.info(e)
            source_x_position = [np.nan, 0]
            source_y_position = [np.nan, 0]
            total_magnitude = [np.nan, 0]
            effective_radius = [np.nan, 0]
            sersic_index = [np.nan, 0]
            axis_ratio = [np.nan, 0]
            position_angle = [np.nan, 0]

    return source_x_position, source_y_position, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle


def get_expdisk_parameters_from_header(output_model_image_header, index):
    """

    :param output_model_image_header:
    :param index:
    :return:
    """

    try:
        source_x_position = [output_model_image_header['{}_XC'.format(index + 2)].split(' +/- ')[0],
                             output_model_image_header['{}_XC'.format(index + 2)].split(' +/- ')[1]]
        source_y_position = [output_model_image_header['{}_YC'.format(index + 2)].split(' +/- ')[0],
                             output_model_image_header['{}_YC'.format(index + 2)].split(' +/- ')[1]]
        total_magnitude = [output_model_image_header['{}_MAG'.format(index + 2)].split(' +/- ')[0],
                           output_model_image_header['{}_MAG'.format(index + 2)].split(' +/- ')[1]]
        effective_radius = [output_model_image_header['{}_RS'.format(index + 2)].split(' +/- ')[0],
                            output_model_image_header['{}_RS'.format(index + 2)].split(' +/- ')[1]]
        sersic_index = [1.0, 0]
        axis_ratio = [output_model_image_header['{}_AR'.format(index + 2)].split(' +/- ')[0],
                      output_model_image_header['{}_AR'.format(index + 2)].split(' +/- ')[1]]
        position_angle = [output_model_image_header['{}_PA'.format(index + 2)].split(' +/- ')[0],
                          output_model_image_header['{}_PA'.format(index + 2)].split(' +/- ')[1]]
    except Exception as e:
        logger.info(e)
        try:
            source_x_position = [output_model_image_header['{}_XC'.format(index + 2)][1:-1], 0]
            source_y_position = [output_model_image_header['{}_YC'.format(index + 2)][1:-1], 0]
            total_magnitude = [output_model_image_header['{}_MAG'.format(index + 2)][1:-1], 0]
            effective_radius = [output_model_image_header['{}_RS'.format(index + 2)][1:-1], 0]
            sersic_index = [1.0, 0]
            axis_ratio = [output_model_image_header['{}_AR'.format(index + 2)][1:-1], 0]
            position_angle = [output_model_image_header['{}_PA'.format(index + 2)][1:-1], 0]
        except Exception as e:
            logger.info(e)
            source_x_position = [np.nan, 0]
            source_y_position = [np.nan, 0]
            total_magnitude = [np.nan, 0]
            effective_radius = [np.nan, 0]
            sersic_index = [np.nan, 0]
            axis_ratio = [np.nan, 0]
            position_angle = [np.nan, 0]

    return source_x_position, source_y_position, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle


def get_devauc_parameters_from_header(output_model_image_header, index):
    """

    :param output_model_image_header:
    :param index:
    :return:
    """

    try:
        source_x_position = [output_model_image_header['{}_XC'.format(index + 2)].split(' +/- ')[0],
                             output_model_image_header['{}_XC'.format(index + 2)].split(' +/- ')[1]]
        source_y_position = [output_model_image_header['{}_YC'.format(index + 2)].split(' +/- ')[0],
                             output_model_image_header['{}_YC'.format(index + 2)].split(' +/- ')[1]]
        total_magnitude = [output_model_image_header['{}_MAG'.format(index + 2)].split(' +/- ')[0],
                           output_model_image_header['{}_MAG'.format(index + 2)].split(' +/- ')[1]]
        effective_radius = [output_model_image_header['{}_RS'.format(index + 2)].split(' +/- ')[0],
                            output_model_image_header['{}_RS'.format(index + 2)].split(' +/- ')[1]]
        sersic_index = [4.0, 0]
        axis_ratio = [output_model_image_header['{}_AR'.format(index + 2)].split(' +/- ')[0],
                      output_model_image_header['{}_AR'.format(index + 2)].split(' +/- ')[1]]
        position_angle = [output_model_image_header['{}_PA'.format(index + 2)].split(' +/- ')[0],
                          output_model_image_header['{}_PA'.format(index + 2)].split(' +/- ')[1]]
    except Exception as e:
        logger.info(e)
        try:
            source_x_position = [output_model_image_header['{}_XC'.format(index + 2)][1:-1], 0]
            source_y_position = [output_model_image_header['{}_YC'.format(index + 2)][1:-1], 0]
            total_magnitude = [output_model_image_header['{}_MAG'.format(index + 2)][1:-1], 0]
            effective_radius = [output_model_image_header['{}_RS'.format(index + 2)][1:-1], 0]
            sersic_index = [4.0, 0]
            axis_ratio = [output_model_image_header['{}_AR'.format(index + 2)][1:-1], 0]
            position_angle = [output_model_image_header['{}_PA'.format(index + 2)][1:-1], 0]
        except Exception as e:
            logger.info(e)
            source_x_position = [np.nan, 0]
            source_y_position = [np.nan, 0]
            total_magnitude = [np.nan, 0]
            effective_radius = [np.nan, 0]
            sersic_index = [np.nan, 0]
            axis_ratio = [np.nan, 0]
            position_angle = [np.nan, 0]

    return source_x_position, source_y_position, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle


def get_background_parameters_from_header(output_model_image_filename):
    """

    :param output_model_image_filename:
    :return:
    """

    try:
        background_value = np.array([output_model_image_filename['1_SKY'].split(' +/- ')[0],
                                     output_model_image_filename['1_SKY'].split(' +/- ')[1]])
        background_x_gradient = np.array([output_model_image_filename['1_DSDX'].split(' +/- ')[0],
                                          output_model_image_filename['1_DSDX'].split(' +/- ')[1]])
        background_y_gradient = np.array([output_model_image_filename['1_DSDY'].split(' +/- ')[0],
                                          output_model_image_filename['1_DSDY'].split(' +/- ')[1]])
        reduced_chisquare = output_model_image_filename['CHI2NU']
    except Exception as e:
        logger.info(e)
        try:
            background_value = np.array([output_model_image_filename['1_SKY'][1:-1], 0])
            background_x_gradient = np.array([output_model_image_filename['1_DSDX'][1:-1], 0])
            background_y_gradient = np.array([output_model_image_filename['1_DSDY'][1:-1], 0])
            reduced_chisquare = output_model_image_filename['CHI2NU']
        except Exception as e:
            logger.info(e)
            background_value = np.array([np.nan, 0])
            background_x_gradient = np.array([np.nan, 0])
            background_y_gradient = np.array([np.nan, 0])
            reduced_chisquare = np.nan

    return background_value, background_x_gradient, background_y_gradient, reduced_chisquare


def get_best_fit_parameters_from_model_image(output_model_image_filename, n_fitted_components, light_profiles):
    """
    This function reads the GALFIT best fit parameters from the imgblock header.

    :param output_model_image_filename:
    :param n_fitted_components:
    :param light_profiles:
    :return:
    """

    parameters_from_header_switcher = {'sersic': get_sersic_parameters_from_header,
                                       'expdisk': get_expdisk_parameters_from_header,
                                       'devauc': get_devauc_parameters_from_header}

    best_fit_source_x_positions = np.empty((n_fitted_components, 2), dtype='U50')
    best_fit_source_y_positions = np.empty((n_fitted_components, 2), dtype='U50')
    best_fit_total_magnitudes = np.empty((n_fitted_components, 2), dtype='U50')
    best_fit_effective_radii = np.empty((n_fitted_components, 2), dtype='U50')
    best_fit_sersic_indices = np.empty((n_fitted_components, 2), dtype='U50')
    best_fit_axis_ratios = np.empty((n_fitted_components, 2), dtype='U50')
    best_fit_position_angles = np.empty((n_fitted_components, 2), dtype='U50')

    try:
        output_model_image_header = fits.getheader(output_model_image_filename, ext=2)

        for i in range(n_fitted_components):

            get_parameters_from_header = parameters_from_header_switcher.get(light_profiles[i],
                                                                             lambda: 'Not implemented...')
            best_fit_source_x_positions[i, :], best_fit_source_y_positions[i, :], best_fit_total_magnitudes[i, :], \
                best_fit_effective_radii[i, :], best_fit_sersic_indices[i, :], best_fit_axis_ratios[i, :], \
                best_fit_position_angles[i, :] = get_parameters_from_header(output_model_image_header, i)

        best_fit_background_value, best_fit_background_x_gradient, best_fit_background_y_gradient, reduced_chisquare = \
            get_background_parameters_from_header(output_model_image_header)

        best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, best_fit_effective_radii, \
            best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles = \
            delete_star_character(best_fit_source_x_positions), delete_star_character(best_fit_source_y_positions), \
            delete_star_character(best_fit_total_magnitudes), delete_star_character(best_fit_effective_radii), \
            delete_star_character(best_fit_sersic_indices), delete_star_character(best_fit_axis_ratios), \
            delete_star_character(best_fit_position_angles)

    except Exception as e:
        logger.info(e)
        logger.info('GALFIT crashed...')
        best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, best_fit_effective_radii, \
            best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles, best_fit_background_value, \
            best_fit_background_x_gradient, best_fit_background_y_gradient, reduced_chisquare = \
            manage_crashed_galfit(n_fitted_components, best_fit_source_x_positions, best_fit_source_y_positions,
                                  best_fit_total_magnitudes, best_fit_effective_radii, best_fit_sersic_indices,
                                  best_fit_axis_ratios, best_fit_position_angles)

    return best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, \
        best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles, \
        best_fit_background_value, best_fit_background_x_gradient, best_fit_background_y_gradient, \
        reduced_chisquare


def manage_crashed_galfit(n_fitted_components, best_fit_source_x_positions, best_fit_source_y_positions,
                          best_fit_total_magnitudes, best_fit_effective_radii, best_fit_sersic_indices,
                          best_fit_axis_ratios, best_fit_position_angles):
    """

    :param n_fitted_components:
    :param best_fit_source_x_positions:
    :param best_fit_source_y_positions:
    :param best_fit_total_magnitudes:
    :param best_fit_effective_radii:
    :param best_fit_sersic_indices:
    :param best_fit_axis_ratios:
    :param best_fit_position_angles:
    :return:
    """

    for i in range(n_fitted_components):
        best_fit_source_x_positions[i, :] = np.array([np.nan, 0])
        best_fit_source_y_positions[i, :] = np.array([np.nan, 0])
        best_fit_total_magnitudes[i, :] = np.array([np.nan, 0])
        best_fit_effective_radii[i, :] = np.array([np.nan, 0])
        best_fit_sersic_indices[i, :] = np.array([np.nan, 0])
        best_fit_axis_ratios[i, :] = np.array([np.nan, 0])
        best_fit_position_angles[i, :] = np.array([np.nan, 0])
    best_fit_background_value = np.array([np.nan, 0])
    best_fit_background_x_gradient = np.array([np.nan, 0])
    best_fit_background_y_gradient = np.array([np.nan, 0])
    reduced_chisquare = np.nan

    return best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, \
        best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles, \
        best_fit_background_value, best_fit_background_x_gradient, best_fit_background_y_gradient, \
        reduced_chisquare


def add_group_values_to_dictionary_stamps(target_name, waveband, index, psf_image_type, sigma_image_type,
                                          background_estimate_method, group, x_dictionary, y_dictionary,
                                          ra_dictionary, dec_dictionary, mag_dictionary, re_dictionary,
                                          n_dictionary, ar_dictionary, pa_dictionary, background_value_dictionary,
                                          background_x_gradient_dictionary, background_y_gradient_dictionary,
                                          reduced_chisquare_dictionary):
    """

    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param group:
    :param x_dictionary:
    :param y_dictionary:
    :param ra_dictionary:
    :param dec_dictionary:
    :param mag_dictionary:
    :param re_dictionary:
    :param n_dictionary:
    :param ar_dictionary:
    :param pa_dictionary:
    :param background_value_dictionary:
    :param background_x_gradient_dictionary:
    :param background_y_gradient_dictionary:
    :param reduced_chisquare_dictionary:
    :return:
    """

    x_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                            sigma_image_type, background_estimate_method)] = group['x'][()]
    y_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                            sigma_image_type, background_estimate_method)] = group['y'][()]
    ra_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                             sigma_image_type, background_estimate_method)] = group['ra'][()]
    dec_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                              sigma_image_type, background_estimate_method)] = group['dec'][()]
    mag_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                              sigma_image_type, background_estimate_method)] = group['mag'][()]
    re_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                             sigma_image_type, background_estimate_method)] = group['Re'][()]
    n_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                            sigma_image_type, background_estimate_method)] = group['n'][()]
    ar_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                             sigma_image_type, background_estimate_method)] = group['ar'][()]
    pa_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                             sigma_image_type, background_estimate_method)] = group['pa'][()]
    background_value_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                           sigma_image_type,
                                                           background_estimate_method)] = group['bkg_amp'][()]
    background_x_gradient_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                                sigma_image_type,
                                                                background_estimate_method)] = group[
        'bkg_x_grad'][()]
    background_y_gradient_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                                sigma_image_type,
                                                                background_estimate_method)] = group[
        'bkg_y_grad'][()]
    reduced_chisquare_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                            sigma_image_type,
                                                            background_estimate_method)] = group['chi_red'][()]


def add_group_values_to_dictionary_fullimage(target_name, waveband, index, psf_image_type, sigma_image_type,
                                             background_estimate_method, group, x_dictionary, y_dictionary,
                                             ra_dictionary, dec_dictionary, mag_dictionary, re_dictionary,
                                             n_dictionary, ar_dictionary, pa_dictionary, background_value_dictionary,
                                             background_x_gradient_dictionary, background_y_gradient_dictionary,
                                             reduced_chisquare_dictionary):
    """

    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param group:
    :param x_dictionary:
    :param y_dictionary:
    :param ra_dictionary:
    :param dec_dictionary:
    :param mag_dictionary:
    :param re_dictionary:
    :param n_dictionary:
    :param ar_dictionary:
    :param pa_dictionary:
    :param background_value_dictionary:
    :param background_x_gradient_dictionary:
    :param background_y_gradient_dictionary:
    :param reduced_chisquare_dictionary:
    :return:
    """

    x_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                         sigma_image_type, background_estimate_method)] = group['x'][()]
    y_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                         sigma_image_type, background_estimate_method)] = group['y'][()]
    ra_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                          sigma_image_type, background_estimate_method)] = group['ra'][()]
    dec_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                           sigma_image_type, background_estimate_method)] = group['dec'][()]
    mag_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                           sigma_image_type, background_estimate_method)] = group['mag'][()]
    re_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                          sigma_image_type, background_estimate_method)] = group['Re'][()]
    n_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                         sigma_image_type, background_estimate_method)] = group['n'][()]
    ar_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                          sigma_image_type, background_estimate_method)] = group['ar'][()]
    pa_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                          sigma_image_type, background_estimate_method)] = group['pa'][()]
    background_value_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                        sigma_image_type,
                                                        background_estimate_method)] = group['bkg_amp'][()]
    background_x_gradient_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                             sigma_image_type,
                                                             background_estimate_method)] = group[
        'bkg_x_grad'][()]
    background_y_gradient_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                             sigma_image_type,
                                                             background_estimate_method)] = group[
        'bkg_y_grad'][()]
    reduced_chisquare_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                         sigma_image_type,
                                                         background_estimate_method)] = group['chi_red'][()]


def add_dictionary_item(best_fit_properties_h5table_filename, target_name, waveband, psf_image_type, sigma_image_type,
                        background_estimate_method, x_dictionary, y_dictionary, ra_dictionary, dec_dictionary,
                        mag_dictionary, re_dictionary, n_dictionary, ar_dictionary, pa_dictionary,
                        background_value_dictionary, background_x_gradient_dictionary,
                        background_y_gradient_dictionary, reduced_chisquare_dictionary, kind='stamp', index=None):
    """

    :param best_fit_properties_h5table_filename:
    :param target_name:
    :param waveband:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param x_dictionary:
    :param y_dictionary:
    :param ra_dictionary:
    :param dec_dictionary:
    :param mag_dictionary:
    :param re_dictionary:
    :param n_dictionary:
    :param ar_dictionary:
    :param pa_dictionary:
    :param background_value_dictionary:
    :param background_x_gradient_dictionary:
    :param background_y_gradient_dictionary:
    :param reduced_chisquare_dictionary:
    :param kind:
    :param index:
    :return:
    """

    fit_table = h5py.File(best_fit_properties_h5table_filename, 'r')
    group = fit_table['{}_{}_{}'.format(psf_image_type, sigma_image_type,
                                        background_estimate_method)]

    add_dictionary_switcher = {'stamp': add_group_values_to_dictionary_stamps,
                               'region': add_group_values_to_dictionary_stamps,
                               'fullimage': add_group_values_to_dictionary_fullimage}

    add_dictionary_function = add_dictionary_switcher.get(kind, lambda: 'Not implemented...')
    add_dictionary_function(target_name, waveband, index,psf_image_type, sigma_image_type,background_estimate_method,
                            group, x_dictionary, y_dictionary, ra_dictionary, dec_dictionary, mag_dictionary,
                            re_dictionary, n_dictionary, ar_dictionary, pa_dictionary, background_value_dictionary,
                            background_x_gradient_dictionary, background_y_gradient_dictionary,
                            reduced_chisquare_dictionary)

    fit_table.close()


def save_property_dictionaries(output_directory, telescope_name, target_name,
                               x_dictionary, y_dictionary, ra_dictionary, dec_dictionary,
                               mag_dictionary, re_dictionary, n_dictionary, ar_dictionary, pa_dictionary,
                               background_value_dictionary, background_x_gradient_dictionary,
                               background_y_gradient_dictionary, reduced_chisquare_dictionary):
    """

    :param output_directory:
    :param telescope_name:
    :param target_name:
    :param x_dictionary:
    :param y_dictionary:
    :param ra_dictionary:
    :param dec_dictionary:
    :param mag_dictionary:
    :param re_dictionary:
    :param n_dictionary:
    :param ar_dictionary:
    :param pa_dictionary:
    :param background_value_dictionary:
    :param background_x_gradient_dictionary:
    :param background_y_gradient_dictionary:
    :param reduced_chisquare_dictionary:
    :return:
    """

    with open(os.path.join(output_directory, '{}_{}_x_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(x_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_y_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(y_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_ra_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(ra_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_dec_dictionary.pkl'.format(telescope_name, target_name)),
              'wb') as f:
        pickle.dump(dec_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_mag_dictionary.pkl'.format(telescope_name, target_name)),
              'wb') as f:
        pickle.dump(mag_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_re_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(re_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_n_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(n_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_ar_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(ar_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_pa_dictionary.pkl'.format(telescope_name, target_name)), 'wb') as f:
        pickle.dump(pa_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_background_value_dictionary.pkl'.format(telescope_name,
                                                                                            target_name)), 'wb') as f:
        pickle.dump(background_value_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_background_x_gradient_dictionary.pkl'.format(telescope_name,
                                                                                                 target_name)),
              'wb') as f:
        pickle.dump(background_x_gradient_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_background_y_gradient_dictionary.pkl'.format(telescope_name,
                                                                                                 target_name)),
              'wb') as f:
        pickle.dump(background_y_gradient_dictionary, f)
    with open(os.path.join(output_directory, '{}_{}_reduced_chisquare_dictionary.pkl'.format(telescope_name,
                                                                                             target_name)),
              'wb') as f:
        pickle.dump(reduced_chisquare_dictionary, f)


def create_empty_property_arrays(number_galaxies_in_stamp, number_wavebands):
    """

    :param number_galaxies_in_stamp:
    :param number_wavebands:
    :return:
    """

    x, x_err, y, y_err, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err, \
        sky_x_grad, sky_x_grad_err, sky_y_grad, sky_y_grad_err, ra, dec, cov_re_ar, cov_re_ar_mag = \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands)), \
        np.empty((number_galaxies_in_stamp, number_wavebands))

    return x, x_err, y, y_err, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, \
        sky_value_err, sky_x_grad, sky_x_grad_err, sky_y_grad, sky_y_grad_err, ra, dec, cov_re_ar, cov_re_ar_mag


def append_galaxy_property_stamp(galaxy_property, galaxy_property_error, dictionary, target_name, waveband,
                                 index, psf_image_type, sigma_image_type, background_estimate_method,
                                 number_of_galaxies_index):
    """

    :param galaxy_property:
    :param galaxy_property_error:
    :param dictionary:
    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param number_of_galaxies_index:
    :return:
    """

    try:
        galaxy_property.append(
            float(dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                        sigma_image_type,
                                                        background_estimate_method)]
                  [number_of_galaxies_index, 0]))
        galaxy_property_error.append(float(dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index,
                                                                                 psf_image_type, sigma_image_type,
                                                                                 background_estimate_method)]
                                           [number_of_galaxies_index, 1]))
    except Exception as e:
        logger.info(e)
        logger.info('Missing {}_{}_{}_{}_{}_{} key in dictionary'.format(target_name, waveband, index,
                                                                         psf_image_type,
                                                                         sigma_image_type,
                                                                         background_estimate_method))
        galaxy_property.append(np.nan)
        galaxy_property_error.append(np.nan)

    return galaxy_property, galaxy_property_error


def append_galaxy_property_fullimage(galaxy_property, galaxy_property_error, dictionary, target_name, waveband,
                                     index, psf_image_type, sigma_image_type, background_estimate_method,
                                     number_of_galaxies_index):
    """

    :param galaxy_property:
    :param galaxy_property_error:
    :param dictionary:
    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param number_of_galaxies_index:
    :return:
    """

    try:
        galaxy_property.append(
            float(dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                     sigma_image_type,
                                                     background_estimate_method)]
                  [number_of_galaxies_index, 0]))
        galaxy_property_error.append(float(dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband,
                                                                              psf_image_type, sigma_image_type,
                                                                              background_estimate_method)]
                                           [number_of_galaxies_index, 1]))
    except Exception as e:
        logger.info(e)
        logger.info('Missing {}_{}_{}_{}_{} key in dictionary'.format(target_name, waveband,
                                                                      psf_image_type,
                                                                      sigma_image_type,
                                                                      background_estimate_method))
        galaxy_property.append(np.nan)
        galaxy_property_error.append(np.nan)

    return galaxy_property, galaxy_property_error


def append_galaxy_properties(galaxy_property, galaxy_property_error, dictionary, target_name, waveband, psf_image_type,
                             sigma_image_type, background_estimate_method, number_of_galaxies_index,
                             kind='stamp', index=None):
    """

    :param galaxy_property:
    :param galaxy_property_error:
    :param dictionary:
    :param target_name:
    :param waveband:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param number_of_galaxies_index:
    :param kind:
    :param index:
    :return:
    """

    append_galaxy_property_switcher = {'stamp': append_galaxy_property_stamp,
                                       'region': append_galaxy_property_stamp,
                                       'fullimage': append_galaxy_property_fullimage}

    append_galaxy_property_function = append_galaxy_property_switcher.get(kind, lambda: 'Not implemented...')

    galaxy_property, galaxy_property_error = append_galaxy_property_function(galaxy_property, galaxy_property_error,
                                                                             dictionary, target_name, waveband,
                                                                             index, psf_image_type, sigma_image_type,
                                                                             background_estimate_method,
                                                                             number_of_galaxies_index)

    return galaxy_property, galaxy_property_error


def append_background_property_stamp(galaxy_property, galaxy_property_error, dictionary, target_name, waveband,
                                     index, psf_image_type, sigma_image_type, background_estimate_method):
    """

    :param galaxy_property:
    :param galaxy_property_error:
    :param dictionary:
    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :return:
    """

    try:
        galaxy_property.append(
            float(dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                        sigma_image_type,
                                                        background_estimate_method)][0]))
        galaxy_property_error.append(float(dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index,
                                                                                 psf_image_type, sigma_image_type,
                                                                                 background_estimate_method)][1]))
    except Exception as e:
        logger.info(e)
        logger.info('Missing {}_{}_{}_{}_{}_{} key in dictionary'.format(target_name, waveband, index,
                                                                         psf_image_type,
                                                                         sigma_image_type,
                                                                         background_estimate_method))
        galaxy_property.append(np.nan)
        galaxy_property_error.append(np.nan)

    return galaxy_property, galaxy_property_error


def append_background_property_fullimage(galaxy_property, galaxy_property_error, dictionary, target_name, waveband,
                                         index, psf_image_type, sigma_image_type, background_estimate_method):
    """

    :param galaxy_property:
    :param galaxy_property_error:
    :param dictionary:
    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :return:
    """

    try:
        galaxy_property.append(
            float(dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                     sigma_image_type,
                                                     background_estimate_method)][0]))
        galaxy_property_error.append(float(dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband,
                                                                              psf_image_type, sigma_image_type,
                                                                              background_estimate_method)][1]))
    except Exception as e:
        logger.info(e)
        logger.info('Missing {}_{}_{}_{}_{} key in dictionary'.format(target_name, waveband,
                                                                      psf_image_type,
                                                                      sigma_image_type,
                                                                      background_estimate_method))
        galaxy_property.append(np.nan)
        galaxy_property_error.append(np.nan)

    return galaxy_property, galaxy_property_error


def append_background_properties(galaxy_property, galaxy_property_error, dictionary, target_name, waveband,
                                 psf_image_type, sigma_image_type, background_estimate_method,
                                 kind='stamp', index=None):
    """

    :param galaxy_property:
    :param galaxy_property_error:
    :param dictionary:
    :param target_name:
    :param waveband:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param kind:
    :param index:
    :return:
    """

    append_background_property_switcher = {'stamp': append_background_property_stamp,
                                           'region': append_background_property_stamp,
                                           'fullimage': append_background_property_fullimage}

    append_background_property_function = append_background_property_switcher.get(kind, lambda: 'Not implemented...')

    galaxy_property, galaxy_property_error = append_background_property_function(galaxy_property,
                                                                                 galaxy_property_error,
                                                                                 dictionary, target_name, waveband,
                                                                                 index, psf_image_type,
                                                                                 sigma_image_type,
                                                                                 background_estimate_method)

    return galaxy_property, galaxy_property_error


def append_ra_dec_property_stamp(ral, decl, ra_dictionary, dec_dictionary, target_name, waveband, index, psf_image_type,
                                 sigma_image_type, background_estimate_method, number_of_galaxies_index):
    """

    :param ral:
    :param decl:
    :param ra_dictionary:
    :param dec_dictionary:
    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param number_of_galaxies_index:
    :return:
    """

    try:
        ral.append(
            float(ra_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index, psf_image_type,
                                                           sigma_image_type, background_estimate_method)]
                  [number_of_galaxies_index]))
        decl.append(float(dec_dictionary['{}_{}_{}_{}_{}_{}'.format(target_name, waveband, index,
                                                                    psf_image_type, sigma_image_type,
                                                                    background_estimate_method)]
                          [number_of_galaxies_index]))
    except Exception as e:
        logger.info(e)
        logger.info('Missing {}_{}_{}_{}_{}_{} key in dictionary'.format(target_name, waveband, index,
                                                                         psf_image_type,
                                                                         sigma_image_type,
                                                                         background_estimate_method))
        ral.append(np.nan)
        decl.append(np.nan)

    return ral, decl


def append_ra_dec_property_fullimage(ral, decl, ra_dictionary, dec_dictionary, target_name, waveband, index,
                                     psf_image_type, sigma_image_type, background_estimate_method,
                                     number_of_galaxies_index):
    """

    :param ral:
    :param decl:
    :param ra_dictionary:
    :param dec_dictionary:
    :param target_name:
    :param waveband:
    :param index:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param number_of_galaxies_index:
    :return:
    """

    try:
        ral.append(
            float(ra_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband, psf_image_type,
                                                        sigma_image_type, background_estimate_method)]
                  [number_of_galaxies_index]))
        decl.append(float(dec_dictionary['{}_{}_{}_{}_{}'.format(target_name, waveband,
                                                                 psf_image_type, sigma_image_type,
                                                                 background_estimate_method)]
                          [number_of_galaxies_index]))
    except Exception as e:
        logger.info(e)
        logger.info('Missing {}_{}_{}_{}_{} key in dictionary'.format(target_name, waveband,
                                                                      psf_image_type,
                                                                      sigma_image_type,
                                                                      background_estimate_method))
        ral.append(np.nan)
        decl.append(np.nan)

    return ral, decl


def append_ra_dec_properties(ral, decl, ra_dictionary, dec_dictionary, target_name, waveband,
                             psf_image_type, sigma_image_type, background_estimate_method,
                             number_of_galaxies_index, kind='stamp', index=None):
    """

    :param ral:
    :param decl:
    :param ra_dictionary:
    :param dec_dictionary:
    :param target_name:
    :param waveband:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param number_of_galaxies_index:
    :param kind:
    :param index:
    :return:
    """

    append_ra_dec_property_switcher = {'stamp': append_ra_dec_property_stamp,
                                       'region': append_ra_dec_property_stamp,
                                       'fullimage': append_ra_dec_property_fullimage}

    append_ra_dec_property_function = append_ra_dec_property_switcher.get(kind, lambda: 'Not implemented...')

    ral, decl = append_ra_dec_property_function(ral, decl, ra_dictionary, dec_dictionary,
                                                target_name, waveband, index,
                                                psf_image_type, sigma_image_type,
                                                background_estimate_method,
                                                number_of_galaxies_index)

    return ral, decl


def compute_weighted_statistic(property, property_errors):
    """

    :param property:
    :param property_errors:
    :return:
    """

    w = (np.isnan(property)) | (np.isnan(property_errors) | (np.array(property_errors) == 0))
    property = np.array(property)[~w]
    property_errors = np.array(property_errors)[~w]
    weights = [1 / property_error ** 2 for property_error in property_errors]
    weighted_stats = DescrStatsW(property, weights=weights, ddof=0)
    mean = weighted_stats.mean
    # std = weighted_stats.std
    std = weighted_stats.std_mean

    return mean, std


def compute_outrej_statistic(property, property_errors):
    """

    :param property:
    :param property_errors:
    :return:
    """

    w = (np.isnan(property)) | (np.isnan(property_errors) | (np.array(property_errors) == 0))
    property = np.array(property)[~w]
    property_errors = np.array(property_errors)[~w]
    weights = [1 / property_error ** 2 for property_error in property_errors]
    weighted_stats = DescrStatsW(property, weights=weights, ddof=0)
    mean = weighted_stats.mean
    std = weighted_stats.std
    z_score = np.abs(property - mean) / std
    outlier_rejected_property = property[z_score < 1]
    # median_value = np.median(outlier_rejected_property)
    std = np.std(outlier_rejected_property)

    return mean, std


def compute_statistic(property, property_errors):
    """

    :param property:
    :param property_errors:
    :return:
    """

    w = (np.isnan(property)) | (np.isnan(property_errors) | (np.array(property_errors) == 0))

    if np.all(w):
        w = np.isnan(property)
        property_nonans = np.array(property)[~w]
        mean = np.median(property_nonans)
        std = 0.1

    else:
        property_nonans = np.array(property)[~w]
        property_nonans_errors = np.array(property_errors)[~w]

        if len(property_nonans) == 1:
            mean = property_nonans[0]
            std = property_nonans_errors[0]

        else:
            # change with formula in https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
            # mean = sum(property_nonans / property_nonans_errors ** 2) / sum(1 / property_nonans_errors ** 2)
            # stand_err = np.sqrt(1 / sum(1 / property_nonans_errors ** 2))
            # chi_square = (1 / (len(property_nonans) - 1)) * sum((property_nonans - mean) ** 2 /
            #                                                     property_nonans_errors ** 2)
            # std = np.sqrt(stand_err ** 2 * chi_square)

            weights = 1 / (property_nonans_errors ** 2)
            mean = sum(property_nonans * weights) / sum(weights)
            unbiased_weighted_estimator_sample_variance = (sum(weights) / (sum(weights)**2 - sum(weights**2))) * \
                (sum(weights * (property_nonans - mean)**2))
            std = np.sqrt(unbiased_weighted_estimator_sample_variance)

    return mean, std


def compute_covariance_measurements(first_property, second_property, property_errors=None):
    """

    :param first_property:
    :param second_property:
    :param property_errors
    :return:
    """

    if property_errors is not None:
        w = (np.isnan(first_property)) | (np.isnan(second_property)) | (np.isnan(property_errors) |
                                                                        (np.array(property_errors) == 0))
        first_property = np.array(first_property)[~w]
        second_property = np.array(second_property)[~w]
        property_errors = np.array(property_errors)[~w]
        property = np.empty((len(first_property), 2))
        property[:, 0] = first_property
        property[:, 1] = second_property
        weights_properties = [1 / property_error ** 2 for property_error in property_errors]
        weights = np.empty((len(weights_properties), 1))
        weights[:, 0] = weights_properties
        weighted_stats = DescrStatsW(property, weights=weights, ddof=0)
    else:
        w = (np.isnan(first_property)) | (np.isnan(second_property))
        first_property = np.array(first_property)[~w]
        second_property = np.array(second_property)[~w]
        property = np.empty((len(first_property), 2))
        property[:, 0] = first_property
        property[:, 1] = second_property
        weighted_stats = DescrStatsW(property, ddof=0)

    cov = weighted_stats.cov[0][1]

    return cov


def get_median_property_arrays(xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg, bkge,
                               bkgx, bkgxe, bkgy, bkgye, ral, decl):
    """
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    :param xl:
    :param xel:
    :param yl:
    :param yel:
    :param ml:
    :param mel:
    :param rl:
    :param rel:
    :param nl:
    :param nel:
    :param al:
    :param ael:
    :param pl:
    :param pel:
    :param bkg:
    :param bkge:
    :param bkgx:
    :param bkgxe:
    :param bkgy:
    :param bkgye:
    :param ral:
    :param decl:
    :return:
    """

    # return np.nanmedian(xl), np.nanstd(xl), np.nanmedian(yl), np.nanstd(yl), np.nanmedian(ml), \
    #     np.nanstd(ml), np.nanmedian(rl), np.nanstd(rl), np.nanmedian(nl), np.nanstd(nl), \
    #     np.nanmedian(al), np.nanstd(al), np.nanmedian(pl), np.nanstd(pl), np.nanmedian(bkg), \
    #     np.nanstd(bkg), np.nanmedian(bkgx), np.nanstd(bkgx), np.nanmedian(bkgy), np.nanstd(bkgy), np.nanmedian(ral), \
    #     np.nanmedian(decl)

    mean_xl, std_xl = compute_statistic(xl, xel)  # compute_weighted_statistic(xl, xel)
    mean_yl, std_yl = compute_statistic(yl, yel)  # compute_weighted_statistic(yl, yel)
    mean_ml, std_ml = compute_statistic(ml, mel)  # compute_weighted_statistic(ml, mel)
    mean_rl, std_rl = compute_statistic(rl, rel)  # compute_weighted_statistic(rl, rel)
    mean_nl, std_nl = compute_statistic(nl, nel)  # compute_weighted_statistic(nl, nel)
    mean_al, std_al = compute_statistic(al, ael)  # compute_weighted_statistic(al, ael)
    mean_pl, std_pl = compute_statistic(pl, pel)  # compute_weighted_statistic(pl, pel)
    mean_bkg, std_bkg = compute_statistic(bkg, bkge)  # compute_weighted_statistic(bkg, bkge)
    mean_bkgx, std_bkgx = compute_statistic(bkgx, bkgxe)  # compute_weighted_statistic(bkgx, bkgxe)
    mean_bkgy, std_bkgy = compute_statistic(bkgy, bkgye)  # compute_weighted_statistic(bkgy, bkgye)
    median_ral = np.nanmedian(ral)
    median_decl = np.nanmedian(decl)

    cov_rl_al = compute_covariance_measurements(rl, al, [a + b for a, b in zip(rel, ael)])
    rl_sqrt_al = rl * np.sqrt(al)
    rl_sqrt_al_err = [np.sqrt((a * re ** 2) + ((r ** 2 / (4 * a)) * ae ** 2) + (r * cov_rl_al)) for a, ae, r, re in
                      zip(al, ael, rl, rel)]
    cov_ml_rl_al = compute_covariance_measurements(ml, rl_sqrt_al, [a + b for a, b in zip(rl_sqrt_al_err, mel)])

    if np.isnan(cov_rl_al):
        cov_rl_al = 0
    if np.isnan(cov_ml_rl_al):
        cov_ml_rl_al = 0

    return mean_xl, std_xl, mean_yl, std_yl, mean_ml, std_ml, mean_rl, std_rl, mean_nl, std_nl, mean_al, std_al, \
        mean_pl, std_pl, mean_bkg, std_bkg, mean_bkgx, std_bkgx, mean_bkgy, std_bkgy, median_ral, median_decl, \
        cov_rl_al, cov_ml_rl_al


def append_properties(target_name, waveband, psf_image_type, background_estimate_method, sigma_image_type,
                      number_of_galaxies_index, x_dictionary, y_dictionary, mag_dictionary,
                      re_dictionary, n_dictionary, ar_dictionary,
                      pa_dictionary, background_value_dictionary, background_x_gradient_dictionary,
                      background_y_gradient_dictionary, ra_dictionary, dec_dictionary,
                      xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl,
                      pel, bkg, bkge, bkgx, bkgxe, bkgy, bkgye, ral, decl, kind='stamp', index=None):
    """

    :param target_name:
    :param waveband:
    :param psf_image_type:
    :param background_estimate_method:
    :param sigma_image_type:
    :param number_of_galaxies_index:
    :param x_dictionary:
    :param y_dictionary:
    :param mag_dictionary:
    :param re_dictionary:
    :param n_dictionary:
    :param ar_dictionary:
    :param pa_dictionary:
    :param background_value_dictionary:
    :param background_x_gradient_dictionary:
    :param background_y_gradient_dictionary:
    :param ra_dictionary:
    :param dec_dictionary:
    :param xl:
    :param xel:
    :param yl:
    :param yel:
    :param ml:
    :param mel:
    :param rl:
    :param rel:
    :param nl:
    :param nel:
    :param al:
    :param ael:
    :param pl:
    :param pel:
    :param bkg:
    :param bkge:
    :param bkgx:
    :param bkgxe:
    :param bkgy:
    :param bkgye:
    :param ral:
    :param decl:
    :param kind:
    :param index:
    :return:
    """

    xl, xel = append_galaxy_properties(xl, xel, x_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    yl, yel = append_galaxy_properties(yl, yel, y_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    ml, mel = append_galaxy_properties(ml, mel, mag_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    rl, rel = append_galaxy_properties(rl, rel, re_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    nl, nel = append_galaxy_properties(nl, nel, n_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    al, ael = append_galaxy_properties(al, ael, ar_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    pl, pel = append_galaxy_properties(pl, pel, pa_dictionary, target_name, waveband,
                                       psf_image_type, sigma_image_type, background_estimate_method,
                                       number_of_galaxies_index, kind=kind, index=index)
    bkg, bkge = append_background_properties(bkg, bkge, background_value_dictionary, target_name,
                                             waveband, psf_image_type, sigma_image_type,
                                             background_estimate_method, kind=kind, index=index)
    bkgx, bkgxe = append_background_properties(bkgx, bkgxe, background_x_gradient_dictionary,
                                               target_name, waveband, psf_image_type,
                                               sigma_image_type, background_estimate_method, kind=kind, index=index)
    bkgy, bkgye = append_background_properties(bkgy, bkgye, background_y_gradient_dictionary,
                                               target_name, waveband, psf_image_type,
                                               sigma_image_type, background_estimate_method, kind=kind, index=index)
    ral, decl = append_ra_dec_properties(ral, decl, ra_dictionary, dec_dictionary, target_name, waveband,
                                         psf_image_type, sigma_image_type, background_estimate_method,
                                         number_of_galaxies_index, kind=kind, index=index)

    return xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg, bkge, bkgx, bkgxe, bkgy, bkgye, ral, decl


def get_median_properties(target_name, wavebands, psf_image_types, background_estimate_methods, sigma_image_types,
                          number_of_galaxies, x_dictionary, y_dictionary,
                          ra_dictionary, dec_dictionary, mag_dictionary, re_dictionary, n_dictionary,
                          ar_dictionary, pa_dictionary, background_value_dictionary,
                          background_x_gradient_dictionary, background_y_gradient_dictionary,
                          kind='stamp', index=None):
    """

    :param target_name:
    :param wavebands:
    :param psf_image_types:
    :param background_estimate_methods:
    :param sigma_image_types:
    :param number_of_galaxies:
    :param x_dictionary:
    :param y_dictionary:
    :param ra_dictionary:
    :param dec_dictionary:
    :param mag_dictionary:
    :param re_dictionary:
    :param n_dictionary:
    :param ar_dictionary:
    :param pa_dictionary:
    :param background_value_dictionary:
    :param background_x_gradient_dictionary:
    :param background_y_gradient_dictionary:
    :param kind:
    :param index:
    :return:
    """

    x_positions, x_position_errors, y_positions, y_position_errors, total_magnitudes, total_magnitude_errors, \
        effective_radii, effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios, axis_ratio_errors, \
        position_angles, position_angle_errors, background_values, background_value_errors, background_x_gradients, \
        background_x_gradient_errors, background_y_gradients, background_y_gradient_errors, ra, dec, \
        covariance_effective_radii_position_angles, covariance_effective_radii_position_angles_magnitudes = \
        create_empty_property_arrays(number_of_galaxies, len(wavebands))

    combinations = [['{}'.format(x), '{}'.format(y), '{}'.format(z)]
                    for x, y, z in itertools.product(psf_image_types, background_estimate_methods,
                                                     sigma_image_types)]

    for n_gal_idx in range(number_of_galaxies):
        for wave_idx in range(len(wavebands)):
            xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg, \
                bkge, bkgx, bkgxe, bkgy, bkgye, ral, decl = [], [], [], [], [], [], [], [], [], [], \
                [], [], [], [], [], [], [], [], [], [], [], []
            for k in range(len(combinations)):
                xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg, bkge, bkgx, bkgxe, \
                    bkgy, bkgye, ral, decl = append_properties(target_name, wavebands[wave_idx], combinations[k][0],
                                                               combinations[k][1], combinations[k][2],
                                                               n_gal_idx, x_dictionary, y_dictionary,
                                                               mag_dictionary,
                                                               re_dictionary, n_dictionary, ar_dictionary,
                                                               pa_dictionary, background_value_dictionary,
                                                               background_x_gradient_dictionary,
                                                               background_y_gradient_dictionary,
                                                               ra_dictionary, dec_dictionary,
                                                               xl, xel, yl,
                                                               yel, ml, mel,
                                                               rl, rel, nl, nel, al, ael, pl, pel, bkg, bkge,
                                                               bkgx, bkgxe,
                                                               bkgy, bkgye, ral, decl, kind=kind, index=index)

            x_positions[n_gal_idx][wave_idx], x_position_errors[n_gal_idx][wave_idx], \
                y_positions[n_gal_idx][wave_idx], y_position_errors[n_gal_idx][wave_idx], \
                total_magnitudes[n_gal_idx][wave_idx], total_magnitude_errors[n_gal_idx][wave_idx], \
                effective_radii[n_gal_idx][wave_idx], effective_radius_errors[n_gal_idx][wave_idx], \
                sersic_indices[n_gal_idx][wave_idx], sersic_index_errors[n_gal_idx][wave_idx], \
                axis_ratios[n_gal_idx][wave_idx], axis_ratio_errors[n_gal_idx][wave_idx], \
                position_angles[n_gal_idx][wave_idx], position_angle_errors[n_gal_idx][wave_idx], \
                background_values[n_gal_idx][wave_idx], background_value_errors[n_gal_idx][wave_idx], \
                background_x_gradients[n_gal_idx][wave_idx], \
                background_x_gradient_errors[n_gal_idx][wave_idx], \
                background_y_gradients[n_gal_idx][wave_idx], \
                background_y_gradient_errors[n_gal_idx][wave_idx], ra[n_gal_idx][wave_idx], dec[n_gal_idx][wave_idx], \
                covariance_effective_radii_position_angles[n_gal_idx][wave_idx], \
                covariance_effective_radii_position_angles_magnitudes[n_gal_idx][wave_idx] = \
                get_median_property_arrays(xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg,
                                           bkge, bkgx, bkgxe, bkgy, bkgye, ral, decl)

    return x_positions, x_position_errors, y_positions, y_position_errors, ra, dec, \
        total_magnitudes, total_magnitude_errors, effective_radii, \
        effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios, \
        axis_ratio_errors, position_angles, position_angle_errors, background_values, \
        background_value_errors, background_x_gradients, background_x_gradient_errors, \
        background_y_gradients, background_y_gradient_errors, covariance_effective_radii_position_angles, \
        covariance_effective_radii_position_angles_magnitudes


def create_empty_table(template_table, wavebands, galaxy_ids_key, light_profiles_key, galaxy_components_key,
                       fit_kind='stamps'):
    """
    it needs to contain the following columns: NUMBER, stamp index, component number_*, telescope name, target_field,
    light_profile_*, x_galfit_*, x_galfit_err_*, y_galfit_*, y_galfit_err_*, ra_galfit_*, dec_galfit_*, mag_galfit_*,
    mag_galfit_err_*, re_galfit_*, re_galfit_err_*, n_galfit_*, n_galfit_err_*, ar_galfit_*, ar_galfit_err_*,
    pa_galfit_*, pa_galfit_err_*, bkg_value_*, bkg_value_err_*, bkg_x_grad_*, bkg_x_grad_err_*, bkg_y_grad_*,
    bkg_y_grad_err_*, chi_square_*

    :param template_table:
    :param wavebands:
    :param galaxy_ids_key:
    :param light_profiles_key:
    :param galaxy_components_key:
    :param fit_kind
    :return:
    """

    if fit_kind == 'stamps':
        column_names = [galaxy_ids_key, 'STAMP_INDEX', 'TELESCOPE_NAME', 'TARGET_FIELD_NAME', galaxy_components_key]
    elif fit_kind == 'regions':
        column_names = [galaxy_ids_key, 'REGION_INDEX', 'TELESCOPE_NAME', 'TARGET_FIELD_NAME', galaxy_components_key]
    else:
        column_names = [galaxy_ids_key, 'TELESCOPE_NAME', 'TARGET_FIELD_NAME', galaxy_components_key]

    galfit_column_names = [name for name in template_table.colnames if 'GALFIT' in name]
    for waveband in wavebands:
        column_names.append('{}_{}'.format(light_profiles_key, waveband))
        for galfit_column in galfit_column_names:
            column_names.append('{}_{}'.format(galfit_column, waveband))

    empty_table = Table()
    for column_name in column_names:
        if (column_name == 'TELESCOPE_NAME') | (column_name == 'TARGET_FIELD_NAME') | \
                (light_profiles_key in column_name) | (column_name == 'STAMP_INDEX') | (column_name == 'REGION_INDEX'):
            empty_table.add_column(Column(name=column_name, dtype='U50'))
        elif (column_name == galaxy_ids_key) | (column_name == galaxy_components_key):
            empty_table.add_column(Column(name=column_name, dtype=int))
        else:
            empty_table.add_column(Column(name=column_name))

    return empty_table


def combine_properties(properties_mastercatalogue, wavebands, telescope_name, target_field_name,
                       galaxy_ids_key, light_profiles_key, galaxy_components_key, fit_kind=None, index=None):
    """

    :param properties_mastercatalogue:
    :param wavebands:
    :param telescope_name:
    :param target_field_name:
    :param galaxy_ids_key:
    :param light_profiles_key:
    :param galaxy_components_key:
    :param fit_kind:
    :param index:
    :return:
    """

    table_properties = create_empty_table(properties_mastercatalogue, wavebands, galaxy_ids_key,
                                          light_profiles_key, galaxy_components_key, fit_kind=fit_kind)

    unique_ids = unique(properties_mastercatalogue, keys=galaxy_ids_key)[galaxy_ids_key]

    for i in range(len(unique_ids)):
        id_mask = np.where(properties_mastercatalogue[galaxy_ids_key] == unique_ids[i])
        n_components = len(unique(properties_mastercatalogue[id_mask], keys=galaxy_components_key)
                           [galaxy_components_key])

        for j in range(n_components):
            row_to_add = []
            if (fit_kind == 'stamps') | (fit_kind == 'regions'):
                row_to_add.extend([int(unique_ids[i]), index, telescope_name, target_field_name, int(j)])
            else:
                row_to_add.extend([int(unique_ids[i]), telescope_name, target_field_name, int(j)])

            for k in range(len(wavebands)):
                mask = np.where((properties_mastercatalogue[galaxy_ids_key] == unique_ids[i]) &
                                (properties_mastercatalogue['WAVEBAND'] == wavebands[k]) &
                                (properties_mastercatalogue[galaxy_components_key] == j))

                try:
                    mean_x, std_x = compute_statistic(properties_mastercatalogue[mask]['X_GALFIT'],
                                                      properties_mastercatalogue[mask]['X_GALFIT_ERR'])
                    mean_y, std_y = compute_statistic(properties_mastercatalogue[mask]['Y_GALFIT'],
                                                      properties_mastercatalogue[mask]['Y_GALFIT_ERR'])
                    mean_mag, std_mag = compute_statistic(properties_mastercatalogue[mask]['MAG_GALFIT'],
                                                          properties_mastercatalogue[mask]['MAG_GALFIT_ERR'])
                    mean_re, std_re = compute_statistic(properties_mastercatalogue[mask]['RE_GALFIT'],
                                                        properties_mastercatalogue[mask]['RE_GALFIT_ERR'])
                    mean_n, std_n = compute_statistic(properties_mastercatalogue[mask]['N_GALFIT'],
                                                      properties_mastercatalogue[mask]['N_GALFIT_ERR'])
                    mean_ar, std_ar = compute_statistic(properties_mastercatalogue[mask]['AR_GALFIT'],
                                                        properties_mastercatalogue[mask]['AR_GALFIT_ERR'])
                    mean_pa, std_pa = compute_statistic(properties_mastercatalogue[mask]['PA_GALFIT'],
                                                        properties_mastercatalogue[mask]['PA_GALFIT_ERR'])
                    mean_bkg, std_bkg = compute_statistic(properties_mastercatalogue[mask]['BKG_VALUE_GALFIT'],
                                                          properties_mastercatalogue[mask]['BKG_VALUE_GALFIT_ERR'])
                    mean_bkgx, std_bkgx = compute_statistic(properties_mastercatalogue[mask]['BKG_X_GRAD_GALFIT'],
                                                            properties_mastercatalogue[mask]['BKG_X_GRAD_GALFIT_ERR'])
                    mean_bkgy, std_bkgy = compute_statistic(properties_mastercatalogue[mask]['BKG_Y_GRAD_GALFIT'],
                                                            properties_mastercatalogue[mask]['BKG_Y_GRAD_GALFIT_ERR'])
                    median_ra = np.nanmedian(properties_mastercatalogue[mask]['RA_GALFIT'])
                    median_dec = np.nanmedian(properties_mastercatalogue[mask]['DEC_GALFIT'])
                    median_chisquare = np.nanmedian(properties_mastercatalogue[mask]['CHI_SQUARE_RED_GALFIT'])
                    row_to_add.extend([properties_mastercatalogue[mask][light_profiles_key][0],
                                       mean_x, std_x, mean_y, std_y, median_ra, median_dec, mean_mag, std_mag, mean_re,
                                       std_re, mean_n, std_n, mean_ar, std_ar, mean_pa, std_pa, mean_bkg, std_bkg,
                                       mean_bkgx, std_bkgx, mean_bkgy, std_bkgy, median_chisquare])
                except Exception as e:
                    logger.info(e)
                    row_to_add.extend(['None', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

            table_properties.add_row(row_to_add)

    return table_properties


def find_matched_rows(cat1, cat2, cat_2_ra_key, cat_2_dec_key, wavebands,
                      reference_waveband):
    """
    Matched rows via coordinates.

    :param cat1:
    :param cat2:
    :param cat_2_ra_key:
    :param cat_2_dec_key:
    :param wavebands:
    :param reference_waveband:
    :return cat2_matched[mask]: matched rows.
    """

    w = (np.isfinite(cat1['RA_GALFIT_{}'.format(reference_waveband)])) &\
        (np.isfinite(cat1['DEC_GALFIT_{}'.format(reference_waveband)]))
    if len(cat1[w]) == len(cat1):
        ra = np.array(cat1['RA_GALFIT_{}'.format(reference_waveband)])
        dec = np.array(cat1['DEC_GALFIT_{}'.format(reference_waveband)])
    else:
        ras = np.empty((len(cat1), len(wavebands)))
        decs = np.empty((len(cat1), len(wavebands)))
        for i in range(len(wavebands)):
            ras[:, i] = cat1['RA_GALFIT_{}'.format(wavebands[i])]
            decs[:, i] = cat1['DEC_GALFIT_{}'.format(wavebands[i])]
        ra = np.nanmedian(ras, axis=1)
        dec = np.nanmedian(decs, axis=1)
        tokeep = (np.isfinite(ra)) & (np.isfinite(dec))
        ra = ra[tokeep]
        dec = dec[tokeep]

    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    catalog = SkyCoord(ra=cat2['{}_{}'.format(cat_2_ra_key, reference_waveband)],
                       dec=cat2['{}_{}'.format(cat_2_dec_key, reference_waveband)])
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    separation = Column(d2d.arcsecond, name='d2d')
    cat2_matched = cat2[idx]
    cat2_matched.add_column(separation)
    mask = (cat2_matched['d2d'] < 1.0)

    return cat2_matched[mask]


def delete_columns(table, cols_to_delete, wavebands, waveband_key):
    """
    Delete single columns.

    :param table:
    :param cols_to_delete:
    :param wavebands:
    :param waveband_key:
    :return table: table with removed columns.
    """

    single_columns = ['XWIN_IMAGE', 'YWIN_IMAGE', 'ALPHAWIN_J2000', 'DELTAWIN_J2000']
    for col in cols_to_delete:
        if col in single_columns:
            try:
                wl = list.copy(wavebands)
                wl.remove('{}'.format(waveband_key))
                for band in wl:
                    table.remove_column('{}_{}'.format(col, band))
            except Exception as e:
                logger.info(e)
                pass
        else:
            for band in wavebands:
                table.remove_column('{}_{}'.format(col, band))
    table.remove_column('d2d')
    return table


def match_galfit_table_with_zcat(galfit_params_table_matched_cat, cluster_zcat, waveband_key='f814w'):
    """
     This function matches the best fitting GALFIT table with the spectroscopic catalogues.

    :param galfit_params_table_matched_cat:
    :param cluster_zcat:
    :param waveband_key:
    :return cat1cat2matched: matched catalogue.
    """

    c = SkyCoord(ra=galfit_params_table_matched_cat['ALPHAWIN_J2000_{}'.format(waveband_key)],
                 dec=galfit_params_table_matched_cat['DELTAWIN_J2000_{}'.format(waveband_key)])
    catalog = SkyCoord(ra=cluster_zcat['ALPHAWIN_J2000_{}'.format(waveband_key)],
                       dec=cluster_zcat['DELTAWIN_J2000_{}'.format(waveband_key)])
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    separation = Column(d2d.arcsecond, name='d2d')
    cluster_zcat_matched = cluster_zcat[idx]
    cluster_zcat_matched.add_column(separation)
    cat1cat2matched = hstack([galfit_params_table_matched_cat, cluster_zcat_matched])
    w = np.where(cat1cat2matched['d2d'] > 1.0)
    for idx in w[0]:
        cat1cat2matched['ID'][idx] = 99.
        cat1cat2matched['RA'][idx] = 99.
        cat1cat2matched['DEC'][idx] = 99.
        cat1cat2matched['z'][idx] = 99.
        cat1cat2matched['z_quality'][idx] = 99.
        cat1cat2matched['multiplicity'][idx] = 99.
        cat1cat2matched['sigma_z'][idx] = 99.
        cat1cat2matched['Kron_R'][idx] = 99.
        cat1cat2matched['root_name'][idx] = 99.
        cat1cat2matched['z_ref'][idx] = 99

    return cat1cat2matched


def create_galfit_params_fits_table(wavebands, x_positions, x_position_errors, y_positions, y_position_errors, ra, dec,
                                    total_magnitudes, total_magnitude_errors, effective_radii,
                                    effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios,
                                    axis_ratio_errors, position_angles, position_angle_errors, background_values,
                                    background_value_errors, background_x_gradients, background_x_gradient_errors,
                                    background_y_gradients, background_y_gradient_errors,
                                    covariance_effective_radii_position_angles,
                                    covariance_effective_radii_position_angles_magnitudes):
    """
    This function creates a table with best fitting GALFIT parameters.

    :param wavebands:
    :param x_positions:
    :param x_position_errors:
    :param y_positions:
    :param y_position_errors:
    :param ra:
    :param dec:
    :param total_magnitudes:
    :param total_magnitude_errors:
    :param effective_radii:
    :param effective_radius_errors:
    :param sersic_indices:
    :param sersic_index_errors:
    :param axis_ratios:
    :param axis_ratio_errors:
    :param position_angles:
    :param position_angle_errors:
    :param background_values:
    :param background_value_errors:
    :param background_x_gradients:
    :param background_x_gradient_errors:
    :param background_y_gradients:
    :param background_y_gradient_errors:
    :param covariance_effective_radii_position_angles:
    :param covariance_effective_radii_position_angles_magnitudes:
    :return galfit_params_table: table
    """

    galfit_params_table = Table()
    for band in wavebands:
        idx_band = wavebands.index(band)
        galfit_params_table['x_galfit_{}'.format(band)] = x_positions[:, idx_band]
        galfit_params_table['x_galfit_err_{}'.format(band)] = x_position_errors[:, idx_band]
        galfit_params_table['y_galfit_{}'.format(band)] = y_positions[:, idx_band]
        galfit_params_table['y_galfit_err_{}'.format(band)] = y_position_errors[:, idx_band]
        galfit_params_table['mag_galfit_{}'.format(band)] = total_magnitudes[:, idx_band]
        galfit_params_table['mag_galfit_err_{}'.format(band)] = total_magnitude_errors[:, idx_band]
        galfit_params_table['re_galfit_{}'.format(band)] = effective_radii[:, idx_band]
        galfit_params_table['re_galfit_err_{}'.format(band)] = effective_radius_errors[:, idx_band]
        galfit_params_table['n_galfit_{}'.format(band)] = sersic_indices[:, idx_band]
        galfit_params_table['n_galfit_err_{}'.format(band)] = sersic_index_errors[:, idx_band]
        galfit_params_table['ar_galfit_{}'.format(band)] = axis_ratios[:, idx_band]
        galfit_params_table['ar_galfit_err_{}'.format(band)] = axis_ratio_errors[:, idx_band]
        galfit_params_table['pa_galfit_{}'.format(band)] = position_angles[:, idx_band]
        galfit_params_table['pa_galfit_err_{}'.format(band)] = position_angle_errors[:, idx_band]
        galfit_params_table['ra_galfit_{}'.format(band)] = ra[:, idx_band]
        galfit_params_table['dec_galfit_{}'.format(band)] = dec[:, idx_band]
        galfit_params_table['cov_galfit_re_ar_{}'.format(band)] = \
            covariance_effective_radii_position_angles[:, idx_band]
        galfit_params_table['cov_galfit_re_ar_mag_{}'.format(band)] = \
            covariance_effective_radii_position_angles_magnitudes[:, idx_band]
    galfit_params_table['bkg_value_galfit'] = background_values[:, 0]
    galfit_params_table['bkg_value_galfit_err'] = background_value_errors[:, 0]
    galfit_params_table['bkg_x_grad_galfit'] = background_x_gradients[:, 0]
    galfit_params_table['bkg_x_grad_galfit_err'] = background_x_gradient_errors[:, 0]
    galfit_params_table['bkg_y_grad_galfit'] = background_y_gradients[:, 0]
    galfit_params_table['bkg_y_grad_galfit_err'] = background_y_gradient_errors[:, 0]

    return galfit_params_table


def create_fixed_image_table(wavebands, x_positions, x_position_errors, y_positions, y_position_errors, ra, dec,
                             total_magnitudes, total_magnitude_errors, effective_radii,
                             effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios,
                             axis_ratio_errors, position_angles, position_angle_errors, background_values,
                             background_value_errors, background_x_gradients, background_x_gradient_errors,
                             background_y_gradients, background_y_gradient_errors,
                             covariance_effective_radii_position_angles,
                             covariance_effective_radii_position_angles_magnitudes):
    """

    :param wavebands:
    :param x_positions:
    :param x_position_errors:
    :param y_positions:
    :param y_position_errors:
    :param ra:
    :param dec:
    :param total_magnitudes:
    :param total_magnitude_errors:
    :param effective_radii:
    :param effective_radius_errors:
    :param sersic_indices:
    :param sersic_index_errors:
    :param axis_ratios:
    :param axis_ratio_errors:
    :param position_angles:
    :param position_angle_errors:
    :param background_values:
    :param background_value_errors:
    :param background_x_gradients:
    :param background_x_gradient_errors:
    :param background_y_gradients:
    :param background_y_gradient_errors:
    :param covariance_effective_radii_position_angles:
    :param covariance_effective_radii_position_angles_magnitudes:
    :return:
    """

    galfit_best_fit_properties_table = \
        create_galfit_params_fits_table(wavebands, x_positions, x_position_errors,
                                        y_positions, y_position_errors, ra, dec,
                                        total_magnitudes, total_magnitude_errors,
                                        effective_radii, effective_radius_errors,
                                        sersic_indices, sersic_index_errors,
                                        axis_ratios, axis_ratio_errors,
                                        position_angles, position_angle_errors,
                                        background_values, background_value_errors,
                                        background_x_gradients,
                                        background_x_gradient_errors,
                                        background_y_gradients,
                                        background_y_gradient_errors,
                                        covariance_effective_radii_position_angles,
                                        covariance_effective_radii_position_angles_magnitudes)

    return galfit_best_fit_properties_table


def match_with_source_galaxies_catalogue(galaxy_properties_table, source_galaxies_catalogue,
                                         source_galaxies_catalogue_id_key):
    """

    :param galaxy_properties_table:
    :param source_galaxies_catalogue:
    :param source_galaxies_catalogue_id_key:
    :return:
    """

    combined_cat = join(galaxy_properties_table, source_galaxies_catalogue, keys=source_galaxies_catalogue_id_key)
    repeating_columns = [name for name in combined_cat.colnames if name[-2:] == '_1']
    for col in repeating_columns:
        combined_cat.rename_column(col, col[:-2])
        combined_cat.remove_column(col[:-2] + '_2')

    return combined_cat


def match_with_target_galaxies_catalogue(galaxy_properties_table, target_galaxies_catalogue, source_galaxies_catalogue,
                                         target_galaxies_catalogue_ra_key, target_galaxies_catalogue_dec_key,
                                         source_galaxies_catalogue_ra_key, source_galaxies_catalogue_dec_key,
                                         wavebands, reference_waveband='f814w'):
    """

    :param galaxy_properties_table:
    :param target_galaxies_catalogue:
    :param source_galaxies_catalogue:
    :param target_galaxies_catalogue_ra_key:
    :param target_galaxies_catalogue_dec_key:
    :param source_galaxies_catalogue_ra_key:
    :param source_galaxies_catalogue_dec_key:
    :param wavebands:
    :param reference_waveband:
    :return:
    """

    matched_target_galaxies_catalogue = find_matched_rows(galaxy_properties_table, target_galaxies_catalogue,
                                                          target_galaxies_catalogue_ra_key,
                                                          target_galaxies_catalogue_dec_key,
                                                          wavebands, reference_waveband)
    matched_source_galaxies_catalogue = find_matched_rows(galaxy_properties_table, source_galaxies_catalogue,
                                                          source_galaxies_catalogue_ra_key,
                                                          source_galaxies_catalogue_dec_key,
                                                          wavebands, reference_waveband)

    # cols_to_delete_target = [target_galaxies_catalogue_ra_key, target_galaxies_catalogue_dec_key]
    # matched_target_galaxies_catalogue = delete_columns(matched_target_galaxies_catalogue, cols_to_delete_target,
    #                                                    wavebands, reference_waveband)
    # cols_to_delete_source = [source_galaxies_catalogue_ra_key, source_galaxies_catalogue_dec_key]
    # matched_source_galaxies_catalogue = delete_columns(matched_source_galaxies_catalogue, cols_to_delete_source,
    #                                                    wavebands, reference_waveband)

    if len(matched_source_galaxies_catalogue) != 0:
        combined_full_target_cat = hstack([matched_target_galaxies_catalogue, matched_source_galaxies_catalogue])
        matched_galaxy_properties_table = hstack([galaxy_properties_table, combined_full_target_cat])
    else:
        matched_galaxy_properties_table = hstack([galaxy_properties_table, matched_target_galaxies_catalogue])

    repeating_columns = [name for name in matched_galaxy_properties_table.colnames if name[-2:] == '_1']
    for col in repeating_columns:
        try:
            matched_galaxy_properties_table.rename_column(col, col[:-2])
            try:
                nans = np.isnan(matched_galaxy_properties_table[col[:-2]])
                matched_galaxy_properties_table[col[:-2]][nans.mask] = matched_galaxy_properties_table[col[:-2] +
                                                                                                       '_2'][nans.mask]
            except Exception as e:
                logger.info(e)
        except KeyError:
            matched_galaxy_properties_table.remove_column(col)

        matched_galaxy_properties_table.remove_column(col[:-2] + '_2')

    return matched_galaxy_properties_table


def delete_repeating_sources(table, wavebands):
    """
    In order to delete the repeating sources due to the stamps procedure, we do the following steps:
    1) we group the table by the keyword 'NUMBER'
    2) if the source is unique, i.e. one single row for a given 'NUMBER', then we simply add that row to a newly
    created empty table.
    3) if the source is not unique, i.e. multiple rows for a given 'NUMBER', then we look for the table indices where
    the reduced chi square is the lowest.
    4) we add the remaining columns.

    :param table:
    :param wavebands:
    :return:
    """

    final_table = table.copy()
    final_table.remove_rows(slice(0, -1))
    final_table.remove_row(0)

    group_by_number = table.group_by('NUMBER')

    for key, group in zip(group_by_number.groups.keys, group_by_number.groups):

        if len(group) == 1:
            final_table.add_row(group[0])

        elif len(group) > 1:

            galfit_col_names = [name for name in group.colnames if 'GALFIT' in name]
            col_names = [name for name in group.colnames if 'GALFIT' not in name]
            params_values = []

            for col in col_names[0:4]:
                params_values.append(group[col][0])

            for waveband in wavebands:
                non_zero_chi = np.where(group['CHI_SQUARE_RED_GALFIT_{}'.format(waveband)] != 0.0)

                for col in galfit_col_names:
                    if (waveband in col) & (len(non_zero_chi[0]) != 0):
                        # pick closer to 1, but with penalty for those smaller than 1, keep errors on
                        # params into consideration, collect all indices of smaller idxmin and use the index that occurs
                        # the most
                        if (all(x >= 1 for x in group['CHI_SQUARE_RED_GALFIT_{}'.format(waveband)][non_zero_chi])) | \
                                (all(x < 1 for x in group['CHI_SQUARE_RED_GALFIT_{}'.format(waveband)][non_zero_chi])):
                            idx_min_chi = np.argmin(abs(1 - group['CHI_SQUARE_RED_GALFIT_{}'
                                                        .format(waveband)][non_zero_chi]))
                        else:
                            idx_min_chi = np.argmax(1 / (group['CHI_SQUARE_RED_GALFIT_{}'
                                                         .format(waveband)][non_zero_chi] - 1))
                        idx_min_mag = np.argmin(group['MAG_GALFIT_ERR_{}'.format(waveband)][non_zero_chi])
                        idx_min_re = np.argmin(group['RE_GALFIT_ERR_{}'.format(waveband)][non_zero_chi])
                        idx_min_n = np.argmin(group['N_GALFIT_ERR_{}'.format(waveband)][non_zero_chi])
                        idx_min_ar = np.argmin(group['AR_GALFIT_ERR_{}'.format(waveband)][non_zero_chi])
                        idx_min_pa = np.argmin(group['PA_GALFIT_ERR_{}'.format(waveband)][non_zero_chi])
                        idxs = np.array([idx_min_chi, idx_min_mag, idx_min_re, idx_min_n, idx_min_ar, idx_min_pa])
                        idx_min = np.argmax(np.bincount(idxs))
                        params_values.append(group[col][non_zero_chi][idx_min])
                    elif (waveband in col) & (len(non_zero_chi[0]) == 0):
                        params_values.append(group[col][0])
                    else:
                        pass

            for col in col_names[4:]:
                params_values.append(group[col][0])

            final_table.add_row(params_values)
        else:
            raise ValueError

    return final_table


def delete_repeating_sources_deprecated(table, wavebands):
    """
    In order to delete the repeating sources due to the stamps procedure, we do the following steps:
    1) we group the table by the keyword 'NUMBER'
    2) if the source is unique, i.e. one single row for a given 'NUMBER', then we simply add that row to a newly
    created empty table.
    3) if the source is not unique, i.e. multiple rows for a given 'NUMBER', then we look for the table indices where
    the galfit magnitudes and effective radii are the closest to the SExtractor MAG_ISO_CORR and
    FLUX_RADIUS respectively. If the table indices agree, then we add the specific row to the newly created empty table.
    If they do not agree, we give preference to the row where re_galfit and FLUX_RADIUS are the closest.
    4) we add the background and sextractor columns.

    :param table:
    :param wavebands:
    :return:
    """

    w = (table['FLUX_RADIUS_{}'.format(wavebands[0])] > 0) & (table['MAG_AUTO_{}'.format(wavebands[0])] != 99)
    for wave in wavebands[1:]:
        w *= (table['FLUX_RADIUS_{}'.format(wave)] > 0) & (table['MAG_AUTO_{}'.format(wave)] != 99)
    table = table[w]

    final_table = table.copy()
    final_table.remove_rows(slice(0, -1))
    final_table.remove_row(0)

    group_by_number = table.group_by('NUMBER')

    for key, group in zip(group_by_number.groups.keys, group_by_number.groups):

        if len(group) == 1:
            final_table.add_row(group[0])

        elif len(group) > 1:
            galfit_col_names = [name for name in group.colnames if 'galfit' in name]
            col_names = [name for name in group.colnames if 'galfit' not in name]
            params_values = []
            for j in range(len(wavebands)):
                mag_dist = abs(
                    group['mag_galfit_{}'.format(wavebands[j])] - group['MAG_ISO_CORR_{}'.format(wavebands[j])])
                re_dist = abs(group['re_galfit_{}'.format(wavebands[j])] - group['FLUX_RADIUS_{}'.format(wavebands[j])])
                try:
                    idx_min_mag_dist = np.nanargmin(mag_dist)
                    idx_min_re_dist = np.nanargmin(re_dist)
                    if idx_min_mag_dist == idx_min_re_dist:
                        for col in galfit_col_names:
                            if wavebands[j] in col:
                                params_values.append(group[col][idx_min_mag_dist])
                    else:
                        for col in galfit_col_names:
                            if wavebands[j] in col:
                                params_values.append(group[col][idx_min_re_dist])
                except ValueError:
                    for col in galfit_col_names:
                        if wavebands[j] in col:
                            params_values.append(group[col][0])
            for col in galfit_col_names:
                if 'bkg' in col:
                    params_values.append(group[col][0])
            for col in col_names:
                params_values.append(group[col][0])
            final_table.add_row(params_values)

        else:
            raise ValueError

    return final_table


def check_parameters_for_next_fitting(source_catalogue, waveband, magnitude_keyword='MAG_AUTO',
                                      size_keyword='FLUX_RADIUS',
                                      minor_axis_keyword='BWIN_IMAGE',
                                      major_axis_keyword='AWIN_IMAGE',
                                      position_angle_keyword='THETAWIN_SKY',
                                      magnitude_error_limit=0.1, magnitude_upper_limit=30,
                                      size_error_limit=1, size_upper_limit=200,
                                      sersic_index_error_limit=1,
                                      sersic_index_upper_limit=8, sersic_index_lower_limit=0.3,
                                      axis_ratio_error_limit=0.1, axis_ratio_lower_limit=0.02,
                                      position_angle_error_limit=20):
    """
    This function checks the parameters from the previous iteration.
    For magnitudes, it flags as bad measurements those that have mag_err > magnitude_error_limit or
    mag > magnitude_upper_limit or mag = nan or mag <=0.
    For sizes, it flags as bad measurements those that have re > size_error_limit or re > size_upper_limit or
    or re = nan or re <=0.
    For sersic indices, it flags as bad measurements those that have n > sersic_error_limit or
    n > sersic_upper_limit or n < sersic_lower_limit or n = nan or n<=0.
    For axis ratio, it flags as bad measurements those that have ar > axisratio_error_limit or
    ar < axisratio_lower_limit or ar = nan or ar <=0.
    For position angle, it flags as bad measurements those that have pa > pa_error_limit or pa = nan or pa = 0.

    :param source_catalogue:
    :param waveband:
    :param magnitude_keyword:
    :param size_keyword:
    :param minor_axis_keyword:
    :param major_axis_keyword:
    :param position_angle_keyword:
    :param magnitude_error_limit:
    :param magnitude_upper_limit:
    :param size_error_limit:
    :param size_upper_limit:
    :param sersic_index_error_limit:
    :param sersic_index_upper_limit:
    :param sersic_index_lower_limit:
    :param axis_ratio_error_limit:
    :param axis_ratio_lower_limit:
    :param position_angle_error_limit:
    :return:
    """

    bad_mag = np.where((source_catalogue['mag_galfit_err_{}'.format(waveband)] > magnitude_error_limit) |
                       (source_catalogue['mag_galfit_{}'.format(waveband)] > magnitude_upper_limit) |
                       (np.isnan(source_catalogue['mag_galfit_{}'.format(waveband)])) |
                       (source_catalogue['mag_galfit_{}'.format(waveband)] <= 0))
    source_catalogue['mag_galfit_{}'.format(waveband)][bad_mag] = source_catalogue['{}_{}'.format(magnitude_keyword,
                                                                                                  waveband)][bad_mag]
    source_catalogue['mag_galfit_err_{}'.format(waveband)][bad_mag] = 0.

    bad_size = np.where((source_catalogue['re_galfit_err_{}'.format(waveband)] > size_error_limit) |
                        (source_catalogue['re_galfit_{}'.format(waveband)] > size_upper_limit) |
                        (np.isnan(source_catalogue['re_galfit_{}'.format(waveband)])) |
                        (source_catalogue['re_galfit_{}'.format(waveband)] <= 0))
    source_catalogue['re_galfit_{}'.format(waveband)][bad_size] = source_catalogue['{}_{}'.format(size_keyword,
                                                                                                  waveband)][bad_size]
    source_catalogue['re_galfit_err_{}'.format(waveband)][bad_size] = 0.
    source_catalogue['n_galfit_{}'.format(waveband)][bad_size] = 2.5

    bad_sersic_index = np.where((source_catalogue['n_galfit_err_{}'.format(waveband)] > sersic_index_error_limit) |
                                (source_catalogue['n_galfit_{}'.format(waveband)] > sersic_index_upper_limit) |
                                (source_catalogue['n_galfit_{}'.format(waveband)] < sersic_index_lower_limit) |
                                (np.isnan(source_catalogue['n_galfit_{}'.format(waveband)])) |
                                (source_catalogue['n_galfit_{}'.format(waveband)] <= 0))
    source_catalogue['n_galfit_{}'.format(waveband)][bad_sersic_index] = 2.5

    bad_axis_ratio = np.where((source_catalogue['ar_galfit_err_{}'.format(waveband)] > axis_ratio_error_limit) |
                              (source_catalogue['ar_galfit_{}'.format(waveband)] < axis_ratio_lower_limit) |
                              (np.isnan(source_catalogue['ar_galfit_{}'.format(waveband)])) |
                              (source_catalogue['ar_galfit_{}'.format(waveband)] <= 0))
    source_catalogue['ar_galfit_{}'.format(waveband)][bad_axis_ratio] = (source_catalogue['{}_{}'
                                                                         .format(minor_axis_keyword,
                                                                                 waveband)][bad_axis_ratio] /
                                                                         source_catalogue['{}_{}'
                                                                         .format(major_axis_keyword,
                                                                                 waveband)][bad_axis_ratio]) ** 2

    bad_position_angle = np.where((source_catalogue['pa_galfit_err_{}'.format(waveband)] > position_angle_error_limit) |
                                  (np.isnan(source_catalogue['pa_galfit_{}'.format(waveband)])) |
                                  (source_catalogue['pa_galfit_{}'.format(waveband)] == 0))
    source_catalogue['pa_galfit_{}'.format(waveband)][bad_position_angle] = \
        source_catalogue['{}_{}'.format(position_angle_keyword, waveband)][bad_position_angle]

    return source_catalogue


def check_parameters_for_next_iteration(source_catalogue, waveband, magnitude_keyword='MAG_AUTO',
                                        size_keyword='FLUX_RADIUS',
                                        minor_axis_keyword='BWIN_IMAGE',
                                        major_axis_keyword='AWIN_IMAGE',
                                        position_angle_keyword='THETAWIN_SKY',
                                        magnitude_error_limit=0.1, magnitude_upper_limit=30,
                                        size_error_limit=1, size_upper_limit=30,
                                        sersic_index_error_limit=0.1,
                                        sersic_index_upper_limit=8, sersic_index_lower_limit=0.3,
                                        key='error_value'):
    """
    This function checks the parameters from the previous iteration.
    For magnitudes, it flags as bad measurements those that have mag_err > magnitude_error_limit or
    mag > magnitude_upper_limit or mag = nan or mag <=0.
    For sizes, it flags as bad measurements those that have re > size_error_limit or re > size_upper_limit or
    or re = nan or re <=0.
    For sersic indices, it flags as bad measurements those that have n > sersic_error_limit or
    n > sersic_upper_limit or n < sersic_lower_limit or n = nan or n<=0.
    All the flagged parameters are then returned to their initial SExtractor values.

    :param source_catalogue:
    :param waveband:
    :param magnitude_keyword:
    :param size_keyword:
    :param minor_axis_keyword:
    :param major_axis_keyword:
    :param position_angle_keyword:
    :param magnitude_error_limit:
    :param magnitude_upper_limit:
    :param size_error_limit:
    :param size_upper_limit:
    :param sersic_index_error_limit:
    :param sersic_index_upper_limit:
    :param sersic_index_lower_limit:
    :param key:
    :return:
    """

    if key == 'mean_error':
        bad_measurements = np.where((source_catalogue['mag_galfit_err_{}'.format(waveband)] >
                                     np.nanmean(source_catalogue['mag_galfit_err_{}'.format(waveband)])) |
                                    (source_catalogue['mag_galfit_{}'.format(waveband)] > magnitude_upper_limit) |
                                    (np.isnan(source_catalogue['mag_galfit_{}'.format(waveband)])) |
                                    (source_catalogue['mag_galfit_{}'.format(waveband)] <= 0) |
                                    (source_catalogue['re_galfit_err_{}'.format(waveband)] >
                                     np.nanmean(source_catalogue['re_galfit_err_{}'.format(waveband)])) |
                                    (source_catalogue['re_galfit_{}'.format(waveband)] > size_upper_limit) |
                                    (np.isnan(source_catalogue['re_galfit_{}'.format(waveband)])) |
                                    (source_catalogue['re_galfit_{}'.format(waveband)] <= 0) |
                                    (source_catalogue['n_galfit_err_{}'.format(waveband)] >
                                     np.nanmean(source_catalogue['re_galfit_err_{}'.format(waveband)])) |
                                    (source_catalogue['n_galfit_{}'.format(waveband)] > sersic_index_upper_limit) |
                                    (source_catalogue['n_galfit_{}'.format(waveband)] < sersic_index_lower_limit) |
                                    (np.isnan(source_catalogue['n_galfit_{}'.format(waveband)])) |
                                    (source_catalogue['n_galfit_{}'.format(waveband)] <= 0))
    else:
        bad_measurements = np.where((source_catalogue['mag_galfit_err_{}'.format(waveband)] > magnitude_error_limit) |
                                    (source_catalogue['mag_galfit_{}'.format(waveband)] > magnitude_upper_limit) |
                                    (np.isnan(source_catalogue['mag_galfit_{}'.format(waveband)])) |
                                    (source_catalogue['mag_galfit_{}'.format(waveband)] <= 0) |
                                    (source_catalogue['re_galfit_err_{}'.format(waveband)] > size_error_limit) |
                                    (source_catalogue['re_galfit_{}'.format(waveband)] > size_upper_limit) |
                                    (np.isnan(source_catalogue['re_galfit_{}'.format(waveband)])) |
                                    (source_catalogue['re_galfit_{}'.format(waveband)] <= 0) |
                                    (source_catalogue['n_galfit_err_{}'.format(waveband)] > sersic_index_error_limit) |
                                    (source_catalogue['n_galfit_{}'.format(waveband)] > sersic_index_upper_limit) |
                                    (source_catalogue['n_galfit_{}'.format(waveband)] < sersic_index_lower_limit) |
                                    (np.isnan(source_catalogue['n_galfit_{}'.format(waveband)])) |
                                    (source_catalogue['n_galfit_{}'.format(waveband)] <= 0))

    source_catalogue['mag_galfit_{}'.format(waveband)][bad_measurements] = \
        source_catalogue['{}_{}'.format(magnitude_keyword, waveband)][bad_measurements]
    source_catalogue['mag_galfit_err_{}'.format(waveband)][bad_measurements] = 0.
    source_catalogue['re_galfit_{}'.format(waveband)][bad_measurements] = \
        source_catalogue['{}_{}'.format(size_keyword, waveband)][bad_measurements]
    source_catalogue['re_galfit_err_{}'.format(waveband)][bad_measurements] = 0.
    source_catalogue['n_galfit_{}'.format(waveband)][bad_measurements] = 2.5
    source_catalogue['ar_galfit_{}'.format(waveband)][bad_measurements] = \
        (source_catalogue['{}_{}'.format(minor_axis_keyword, waveband)][bad_measurements] /
         source_catalogue['{}_{}'.format(major_axis_keyword, waveband)][bad_measurements]) ** 2
    source_catalogue['pa_galfit_{}'.format(waveband)][bad_measurements] = \
        source_catalogue['{}_{}'.format(position_angle_keyword, waveband)][bad_measurements]

    return source_catalogue


def check_ucat_sims_parameters_for_next_fitting(source_catalogue, waveband, magnitude_keyword='MAG_AUTO',
                                                size_keyword='FLUX_RADIUS', minor_axis_keyword='BWIN_IMAGE',
                                                major_axis_keyword='AWIN_IMAGE', position_angle_keyword='THETAWIN_SKY',
                                                magnitude_error_limit=0.1, magnitude_upper_limit=30,
                                                size_error_limit=1, size_upper_limit=200, sersic_index_error_limit=1,
                                                sersic_index_lowerlimit=0.3, sersic_index_upper_limit=8,
                                                axis_ratio_error_limit=0.1):
    """
        This function assumes no 99, 0, or nan values in the control catalogue, e.g., SExtractor catalogue.

        :param source_catalogue:
        :param waveband:
        :param magnitude_keyword:
        :param size_keyword:
        :param minor_axis_keyword:
        :param major_axis_keyword:
        :param position_angle_keyword:
        :param magnitude_error_limit:
        :param magnitude_upper_limit:
        :param size_error_limit:
        :param size_upper_limit:
        :param sersic_index_error_limit:
        :param sersic_index_lowerlimit:
        :param sersic_index_upper_limit:
        :param axis_ratio_error_limit:
        :return:
        """

    bad_mag = np.where((source_catalogue['mag_galfit_err_{}'.format(waveband)] > magnitude_error_limit) |
                       (source_catalogue['mag_galfit_{}'.format(waveband)] > magnitude_upper_limit) |
                       (np.isnan(source_catalogue['mag_galfit_{}'.format(waveband)])) |
                       (source_catalogue['mag_galfit_{}'.format(waveband)] <= 0) |
                       (abs(source_catalogue['mag_galfit_{}'.format(waveband)] - source_catalogue[
                           '{}_{}'.format(magnitude_keyword, waveband)]) > 0.1))
    source_catalogue['mag_galfit_{}'.format(waveband)][bad_mag] = source_catalogue['{}_{}'.format(magnitude_keyword,
                                                                                                  waveband)][bad_mag]
    source_catalogue['mag_galfit_err_{}'.format(waveband)][bad_mag] = 0.
    source_catalogue['n_galfit_{}'.format(waveband)][bad_mag] = source_catalogue['sersic_n_{}'.format(waveband)][
        bad_mag]

    bad_size = np.where((source_catalogue['re_galfit_err_{}'.format(waveband)] > size_error_limit) |
                        (source_catalogue['re_galfit_{}'.format(waveband)] <= 0) |
                        (source_catalogue['re_galfit_{}'.format(waveband)] > size_upper_limit) |
                        (np.isnan(source_catalogue['re_galfit_{}'.format(waveband)])) |
                        (abs(source_catalogue['re_galfit_{}'.format(waveband)] - source_catalogue[
                            '{}_{}'.format(size_keyword, waveband)]) > 2))
    source_catalogue['re_galfit_{}'.format(waveband)][bad_size] = source_catalogue['{}_{}'.format(size_keyword,
                                                                                                  waveband)][bad_size]
    source_catalogue['re_galfit_err_{}'.format(waveband)][bad_size] = 0.
    source_catalogue['n_galfit_{}'.format(waveband)][bad_size] = source_catalogue['sersic_n_{}'.format(waveband)][
        bad_size]

    bad_sersic_index = np.where((source_catalogue['n_galfit_err_{}'.format(waveband)] > sersic_index_error_limit) |
                                (source_catalogue['n_galfit_{}'.format(waveband)] < sersic_index_lowerlimit) |
                                (source_catalogue['n_galfit_{}'.format(waveband)] > sersic_index_upper_limit) |
                                (np.isnan(source_catalogue['n_galfit_{}'.format(waveband)])) |
                                (abs(source_catalogue['n_galfit_{}'.format(waveband)] - source_catalogue[
                                    'sersic_n_{}'.format(waveband)]) > 0.25))
    source_catalogue['n_galfit_{}'.format(waveband)][bad_sersic_index] = \
        source_catalogue['sersic_n_{}'.format(waveband)][bad_sersic_index]

    bad_axis_ratio = np.where((source_catalogue['ar_galfit_err_{}'.format(waveband)] > axis_ratio_error_limit) |
                              (np.isnan(source_catalogue['ar_galfit_{}'.format(waveband)])) |
                              (source_catalogue['ar_galfit_{}'.format(waveband)] <= 0.02))
    source_catalogue['ar_galfit_{}'.format(waveband)][bad_axis_ratio] = \
        source_catalogue['{}_{}'.format(minor_axis_keyword, waveband)][bad_axis_ratio] / \
        source_catalogue['{}_{}'.format(major_axis_keyword, waveband)][bad_axis_ratio]

    bad_position_angle = np.where((np.isnan(source_catalogue['pa_galfit_{}'.format(waveband)])) |
                                  (source_catalogue['pa_galfit_{}'.format(waveband)] == 0))
    source_catalogue['pa_galfit_{}'.format(waveband)][bad_position_angle] = \
        source_catalogue['{}_{}'.format(position_angle_keyword, waveband)][bad_position_angle]

    return source_catalogue


def assign_sources_to_region(region_sci_image_filename, sources_catalogue, ra_keyword='ALPHAWIN_J2000',
                             dec_keyword='DELTAWIN_J2000', waveband_keyword='f814w'):
    """

    :param region_sci_image_filename:
    :param sources_catalogue:
    :param ra_keyword:
    :param dec_keyword:
    :param waveband_keyword:
    :return:
    """

    ra = sources_catalogue['{}_{}'.format(ra_keyword, waveband_keyword)]
    dec = sources_catalogue['{}_{}'.format(dec_keyword, waveband_keyword)]

    x, y = ra_dec_2_xy(ra, dec, region_sci_image_filename)
    header = fits.getheader(region_sci_image_filename)
    mask = np.where((x > 0) & (x < header['NAXIS1']) & (y > 0) & (y < header['NAXIS2']))
    region_catalogue = sources_catalogue[mask]
    region_catalogue.write(os.path.splitext(region_sci_image_filename)[0] + '.cat', format='fits', overwrite=True)

    return region_catalogue


def match_galfit_table_with_dacunha(galfit_params_table_matched_cat, dacunha_cat, waveband_key='f814w'):
    """
    This function matches the best fitting GALFIT table with the Da Cunha properties.

    :param galfit_params_table_matched_cat:
    :param dacunha_cat:
    :param waveband_key:
    ::return cat1cat2matched: matched catalogue.
    """

    c = SkyCoord(ra=galfit_params_table_matched_cat['ALPHAWIN_J2000_{}'.format(waveband_key)],
                 dec=galfit_params_table_matched_cat['DELTAWIN_J2000_{}'.format(waveband_key)])
    catalog = SkyCoord(ra=dacunha_cat['RA'] * u.degree, dec=dacunha_cat['DEC'] * u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    separation = Column(d2d.arcsecond, name='d2d')
    dacunha_cat_matched = dacunha_cat[idx]
    dacunha_cat_matched.add_column(separation)
    cat1cat2matched = hstack([galfit_params_table_matched_cat, dacunha_cat_matched])
    cat1cat2matched.rename_column('ID_2', 'ID')
    cat1cat2matched.remove_column('ID_1')
    col_name_list = cat1cat2matched.colnames

    for j in range(len(col_name_list)):
        if col_name_list[j][-2:] == '_1':
            cat1cat2matched.rename_column(col_name_list[j], col_name_list[j][:-2])
        elif col_name_list[j][-2:] == '_2':
            cat1cat2matched.remove_column(col_name_list[j])
        else:
            pass
    w = np.where(cat1cat2matched['d2d'] > 1.0)
    for idx in w[0]:
        cat1cat2matched['logM'][idx] = 99.
        cat1cat2matched['err_logM'][idx] = 99.
        cat1cat2matched['sSFR'][idx] = 99.
        cat1cat2matched['err_sSFR'][idx] = 99.
    cat1cat2matched.remove_column('d2d')

    return cat1cat2matched


def match_galfit_table_with_gobat(galfit_params_table_matched_cat, gobat_cat, waveband_key='f814w'):
    """
    This function matches the best fitting GALFIT table with the spectroscopic catalogues.

    :param galfit_params_table_matched_cat:
    :param gobat_cat:
    :param waveband_key:
    :return cat1cat2matched: matched catalogue.
    """

    c = SkyCoord(ra=galfit_params_table_matched_cat['ALPHAWIN_J2000_{}'.format(waveband_key)],
                 dec=galfit_params_table_matched_cat['DELTAWIN_J2000_{}'.format(waveband_key)])
    catalog = SkyCoord(ra=gobat_cat['RA'] * u.degree, dec=gobat_cat['DEC'] * u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    separation = Column(d2d.arcsecond, name='d2d')
    gobat_cat_matched = gobat_cat[idx]
    gobat_cat_matched.add_column(separation)
    cat1cat2matched = hstack([galfit_params_table_matched_cat, gobat_cat_matched])
    col_name_list = cat1cat2matched.colnames

    for j in range(len(col_name_list)):
        if col_name_list[j][-2:] == '_1':
            cat1cat2matched.rename_column(col_name_list[j], col_name_list[j][:-2])
        elif col_name_list[j][-2:] == '_2':
            cat1cat2matched.remove_column(col_name_list[j])
        else:
            pass
    w = np.where(cat1cat2matched['d2d'] > 1.0)
    for idx in w[0]:
        cat1cat2matched['CHI2'][idx] = 99.
        cat1cat2matched['M*_LOW_Msun'][idx] = 99.
        cat1cat2matched['M*_BEST_Msun'][idx] = 99.
        cat1cat2matched['M*_HIGH_Msun'][idx] = 99.
        cat1cat2matched['T_LOW_Gyr'][idx] = 99.
        cat1cat2matched['T_BEST_Gyr'][idx] = 99.
        cat1cat2matched['T_HIGH_Gyr'][idx] = 99.
        cat1cat2matched['tSF_LOW_Gyr'][idx] = 99.
        cat1cat2matched['tSF_BEST_Gyr'][idx] = 99.
        cat1cat2matched['tSF_HIGH_Gyr'][idx] = 99.
        cat1cat2matched['SFR_LOW_Gyr'][idx] = 99.
        cat1cat2matched['SFR_BEST_Gyr'][idx] = 99.
        cat1cat2matched['SFR_HIGH_Gyr'][idx] = 99.
        cat1cat2matched['MAG_B_LOW'][idx] = 99.
        cat1cat2matched['MAG_B_BEST'][idx] = 99.
        cat1cat2matched['MAG_B_HIGH'][idx] = 99.
        cat1cat2matched['MAG_V_LOW'][idx] = 99.
        cat1cat2matched['MAG_V_BEST'][idx] = 99.
        cat1cat2matched['MAG_V_HIGH'][idx] = 99.
        cat1cat2matched['MAG_R_LOW'][idx] = 99.
        cat1cat2matched['MAG_R_BEST'][idx] = 99.
        cat1cat2matched['MAG_R_HIGH'][idx] = 99.
        cat1cat2matched['MAG_I_LOW'][idx] = 99.
        cat1cat2matched['MAG_I_BEST'][idx] = 99.
        cat1cat2matched['MAG_I_HIGH'][idx] = 99.
        cat1cat2matched['f814_clash'][idx] = 99.
        cat1cat2matched['ef814_clash'][idx] = 99.
        cat1cat2matched['AGE_LOW'][idx] = 99.
        cat1cat2matched['AGE_BEST'][idx] = 99.
        cat1cat2matched['AGE_HIGH'][idx] = 99.
    cat1cat2matched.remove_column('d2d')

    return cat1cat2matched


def match_galfit_table_with_published_params(galfit_params_table_matched_cat, published_cat, waveband_key='f814w'):
    """
    This function matches the best fitting GALFIT table with the published catalogues.

    :param galfit_params_table_matched_cat:
    :param published_cat:
    :param waveband_key:
    :return cat1cat2matched: matched catalogue.
    """

    c = SkyCoord(ra=galfit_params_table_matched_cat['ALPHAWIN_J2000_{}'.format(waveband_key)],
                 dec=galfit_params_table_matched_cat['DELTAWIN_J2000_{}'.format(waveband_key)])
    catalog = SkyCoord(ra=published_cat['RA'] * u.degree, dec=published_cat['DEC'] * u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    separation = Column(d2d.arcsecond, name='d2d')
    published_cat_matched = published_cat[idx]
    published_cat_matched.add_column(separation)
    cat1cat2matched = hstack([galfit_params_table_matched_cat, published_cat_matched])
    col_name_list = cat1cat2matched.colnames

    for j in range(len(col_name_list)):
        if col_name_list[j][-2:] == '_1':
            cat1cat2matched.rename_column(col_name_list[j], col_name_list[j][:-2])
        elif col_name_list[j][-2:] == '_2':
            cat1cat2matched.remove_column(col_name_list[j])
        else:
            pass
    cat1cat2matched.remove_column('d2d')

    return cat1cat2matched
