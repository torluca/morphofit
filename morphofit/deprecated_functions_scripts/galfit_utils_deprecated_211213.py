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


def format_position_for_galfit(source_galaxies_catalogue, ra_key_sources_catalogue, dec_key_sources_catalogue,
                               x_key_sources_catalogue, y_key_sources_catalogue, waveband,
                               sci_image_filename, initial_positions_choice='RADEC'):
    """

    :param source_galaxies_catalogue:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param waveband:
    :param sci_image_filename:
    :param initial_positions_choice:
    :return:
    """

    source_positions = np.empty((len(source_galaxies_catalogue), 4))
    ra = np.empty(len(source_galaxies_catalogue))
    dec = np.empty(len(source_galaxies_catalogue))
    tofit = 1

    for i in range(len(source_galaxies_catalogue)):
        ra[i] = source_galaxies_catalogue['{}_{}'.format(ra_key_sources_catalogue, waveband)][i]
        dec[i] = source_galaxies_catalogue['{}_{}'.format(dec_key_sources_catalogue, waveband)][i]

        if initial_positions_choice == 'RADEC':
            x, y = single_ra_dec_2_xy(ra[i], dec[i], sci_image_filename)
        elif initial_positions_choice == 'XY':
            x = source_galaxies_catalogue['{}_{}'.format(x_key_sources_catalogue, waveband)][i]
            y = source_galaxies_catalogue['{}_{}'.format(y_key_sources_catalogue, waveband)][i]
        else:
            raise ValueError

        source_positions[i, :] = [x, y, tofit, tofit]

    return source_positions, ra, dec


def format_axis_ratio_for_galfit(source_galaxies_catalogue, waveband, axis_ratio_key_sources_catalogue,
                                 minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                 initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param waveband:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param initial_axis_ratios_choice:
    :return:
    """

    axis_ratio = np.empty((len(source_galaxies_catalogue), 2))
    axis_ratio[:, 1] = np.full(len(source_galaxies_catalogue), 1, dtype=int)

    if initial_axis_ratios_choice == 'single_value':
        axis_ratio[:, 0] = [source_galaxies_catalogue[axis_ratio_key_sources_catalogue][i]
                            for i in range(len(source_galaxies_catalogue))]
    elif initial_axis_ratios_choice == 'minormajor':
        axis_ratio[:, 0] = [(source_galaxies_catalogue['{}_{}'.format(minor_axis_key_sources_catalogue, waveband)][i] /
                             source_galaxies_catalogue['{}_{}'.format(major_axis_key_sources_catalogue,
                                                                      waveband)][i])**2
                            for i in range(len(source_galaxies_catalogue))]
    else:
        raise KeyError

    return axis_ratio


def format_common_profile_properties(source_galaxies_catalogue, sci_image_filename, waveband,
                                     ra_key_sources_catalogue, dec_key_sources_catalogue,
                                     x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                                     effective_radius_key_sources_catalogue, axis_ratio_key_sources_catalogue,
                                     minor_axis_key_sources_catalogue,
                                     major_axis_key_sources_catalogue, position_angle_sources_catalogue,
                                     initial_positions_choice='RADEC', initial_axis_ratios_choice='single_value'):

    source_positions, ra, dec = format_position_for_galfit(source_galaxies_catalogue, ra_key_sources_catalogue,
                                                           dec_key_sources_catalogue, x_key_sources_catalogue,
                                                           y_key_sources_catalogue, waveband,
                                                           sci_image_filename, initial_positions_choice)
    axis_ratio = format_axis_ratio_for_galfit(source_galaxies_catalogue, waveband, axis_ratio_key_sources_catalogue,
                                              minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                              initial_axis_ratios_choice)
    total_magnitude = np.empty((len(source_galaxies_catalogue), 2))
    effective_radius = np.empty((len(source_galaxies_catalogue), 2))
    position_angle = np.empty((len(source_galaxies_catalogue), 2))
    tofit = 1

    for i in range(len(source_galaxies_catalogue)):
        total_magnitude[i, :] = [source_galaxies_catalogue['{}_{}'.format(magnitude_key_sources_catalogue,
                                                                          waveband)][i], tofit]
        effective_radius[i, :] = [source_galaxies_catalogue['{}_{}'.format(effective_radius_key_sources_catalogue,
                                                                           waveband)][i], tofit]
        position_angle[i, :] = [source_galaxies_catalogue['{}_{}'.format(position_angle_sources_catalogue,
                                                                         waveband)][i], tofit]

    return source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle


def format_sersic_index_for_galfit_sersic(source_galaxies_catalogue, sersic_index_key_sources_catalogue,
                                          waveband, initial_sersic_indices_choice='fixed_value',
                                          initial_sersic_indices_value=2.5):
    """

    :param source_galaxies_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param waveband:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :return:
    """

    if initial_sersic_indices_choice == 'fixed_value':
        sersic_index = np.full((len(source_galaxies_catalogue), 2), [initial_sersic_indices_value, 1])
    elif initial_sersic_indices_choice == 'from_catalogue':
        sersic_index = np.empty((len(source_galaxies_catalogue), 2))
        sersic_index[:, 0] = source_galaxies_catalogue['{}_{}'.format(sersic_index_key_sources_catalogue, waveband)]
        sersic_index[:, 1] = np.full(len(source_galaxies_catalogue), 1)
    else:
        raise KeyError

    return sersic_index


def format_sersic_index_for_galfit_devauc(source_galaxies_catalogue):
    """

    :param source_galaxies_catalogue:
    :return:
    """

    sersic_index = np.full((len(source_galaxies_catalogue), 2), [4, 0])
    return sersic_index


def format_sersic_index_for_galfit_expdisk(source_galaxies_catalogue):
    """

    :param source_galaxies_catalogue:
    :return:
    """

    sersic_index = np.full((len(source_galaxies_catalogue), 2), [1, 0])
    return sersic_index


def format_sersic_index_for_galfit_double_sersic(source_galaxies_catalogue, sersic_index_key_sources_catalogue,
                                                 waveband, initial_sersic_indices_choice='fixed_value',
                                                 initial_sersic_indices_value=2.5):
    """

    :param source_galaxies_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param waveband:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :return:
    """

    if initial_sersic_indices_choice == 'fixed_value':
        sersic_index = np.full((len(source_galaxies_catalogue), 2), [initial_sersic_indices_value, 1])
        sersic_index = np.vstack((sersic_index, sersic_index))
    elif initial_sersic_indices_choice == 'from_catalogue':
        sersic_index = np.empty((len(source_galaxies_catalogue), 2))
        sersic_index[:, 0] = source_galaxies_catalogue['{}_{}'.format(sersic_index_key_sources_catalogue, waveband)]
        sersic_index[:, 1] = np.full(len(source_galaxies_catalogue), 1)
        sersic_index = np.vstack((sersic_index, sersic_index))
    else:
        raise KeyError

    return sersic_index


def format_sersic_index_for_galfit_triple_sersic(source_galaxies_catalogue, sersic_index_key_sources_catalogue,
                                                 waveband, initial_sersic_indices_choice='fixed_value',
                                                 initial_sersic_indices_value=2.5):
    """

    :param source_galaxies_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param waveband:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :return:
    """

    if initial_sersic_indices_choice == 'fixed_value':
        sersic_index = np.full((len(source_galaxies_catalogue), 2), [initial_sersic_indices_value, 1])
        sersic_index = np.vstack((sersic_index, sersic_index, sersic_index))
    elif initial_sersic_indices_choice == 'from_catalogue':
        sersic_index = np.empty((len(source_galaxies_catalogue), 2))
        sersic_index[:, 0] = source_galaxies_catalogue['{}_{}'.format(sersic_index_key_sources_catalogue, waveband)]
        sersic_index[:, 1] = np.full(len(source_galaxies_catalogue), 1)
        sersic_index = np.vstack((sersic_index, sersic_index, sersic_index))
    else:
        raise KeyError

    return sersic_index


def format_properties_sersic(source_galaxies_catalogue, sci_image_filename, waveband,
                             ra_key_sources_catalogue, dec_key_sources_catalogue,
                             x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                             effective_radius_key_sources_catalogue, sersic_index_key_sources_catalogue,
                             axis_ratio_key_sources_catalogue, minor_axis_key_sources_catalogue,
                             major_axis_key_sources_catalogue, position_angle_sources_catalogue,
                             initial_positions_choice='RADEC', initial_sersic_indices_choice='fixed_value',
                             initial_sersic_indices_value=2.5, initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param sci_image_filename:
    :param waveband:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param magnitude_key_sources_catalogue:
    :param effective_radius_key_sources_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param position_angle_sources_catalogue:
    :param initial_positions_choice:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :param initial_axis_ratios_choice:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties(source_galaxies_catalogue, sci_image_filename, waveband,
                                         ra_key_sources_catalogue, dec_key_sources_catalogue,
                                         x_key_sources_catalogue, y_key_sources_catalogue,
                                         magnitude_key_sources_catalogue,
                                         effective_radius_key_sources_catalogue, axis_ratio_key_sources_catalogue,
                                         minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                         position_angle_sources_catalogue, initial_positions_choice,
                                         initial_axis_ratios_choice)
    light_profiles = np.full(len(source_galaxies_catalogue), 'sersic')
    sersic_index = format_sersic_index_for_galfit_sersic(source_galaxies_catalogue,
                                                         sersic_index_key_sources_catalogue, waveband,
                                                         initial_sersic_indices_choice,
                                                         initial_sersic_indices_value)
    subtract = np.full(len(source_galaxies_catalogue), '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_devauc(source_galaxies_catalogue, sci_image_filename, waveband,
                             ra_key_sources_catalogue, dec_key_sources_catalogue,
                             x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                             effective_radius_key_sources_catalogue, sersic_index_key_sources_catalogue,
                             axis_ratio_key_sources_catalogue, minor_axis_key_sources_catalogue,
                             major_axis_key_sources_catalogue, position_angle_sources_catalogue,
                             initial_positions_choice='RADEC', initial_sersic_indices_choice='fixed_value',
                             initial_sersic_indices_value=2.5, initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param sci_image_filename:
    :param waveband:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param magnitude_key_sources_catalogue:
    :param effective_radius_key_sources_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param position_angle_sources_catalogue:
    :param initial_positions_choice:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :param initial_axis_ratios_choice:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties(source_galaxies_catalogue, sci_image_filename, waveband,
                                         ra_key_sources_catalogue, dec_key_sources_catalogue,
                                         x_key_sources_catalogue, y_key_sources_catalogue,
                                         magnitude_key_sources_catalogue,
                                         effective_radius_key_sources_catalogue, axis_ratio_key_sources_catalogue,
                                         minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                         position_angle_sources_catalogue, initial_positions_choice,
                                         initial_axis_ratios_choice)
    light_profiles = np.full(len(source_galaxies_catalogue), 'devauc')
    sersic_index = format_sersic_index_for_galfit_devauc(source_galaxies_catalogue)
    subtract = np.full(len(source_galaxies_catalogue), '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_expdisk(source_galaxies_catalogue, sci_image_filename, waveband,
                              ra_key_sources_catalogue, dec_key_sources_catalogue,
                              x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                              effective_radius_key_sources_catalogue, sersic_index_key_sources_catalogue,
                              axis_ratio_key_sources_catalogue, minor_axis_key_sources_catalogue,
                              major_axis_key_sources_catalogue, position_angle_sources_catalogue,
                              initial_positions_choice='RADEC', initial_sersic_indices_choice='fixed_value',
                              initial_sersic_indices_value=2.5, initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param sci_image_filename:
    :param waveband:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param magnitude_key_sources_catalogue:
    :param effective_radius_key_sources_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param position_angle_sources_catalogue:
    :param initial_positions_choice:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :param initial_axis_ratios_choice:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties(source_galaxies_catalogue, sci_image_filename, waveband,
                                         ra_key_sources_catalogue, dec_key_sources_catalogue,
                                         x_key_sources_catalogue, y_key_sources_catalogue,
                                         magnitude_key_sources_catalogue,
                                         effective_radius_key_sources_catalogue, axis_ratio_key_sources_catalogue,
                                         minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                         position_angle_sources_catalogue, initial_positions_choice,
                                         initial_axis_ratios_choice)
    light_profiles = np.full(len(source_galaxies_catalogue), 'devauc')
    sersic_index = format_sersic_index_for_galfit_expdisk(source_galaxies_catalogue)
    subtract = np.full(len(source_galaxies_catalogue), '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_double_sersic(source_galaxies_catalogue, sci_image_filename, waveband,
                                    ra_key_sources_catalogue, dec_key_sources_catalogue,
                                    x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                                    effective_radius_key_sources_catalogue, sersic_index_key_sources_catalogue,
                                    axis_ratio_key_sources_catalogue, minor_axis_key_sources_catalogue,
                                    major_axis_key_sources_catalogue, position_angle_sources_catalogue,
                                    initial_positions_choice='RADEC', initial_sersic_indices_choice='fixed_value',
                                    initial_sersic_indices_value=2.5, initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param sci_image_filename:
    :param waveband:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param magnitude_key_sources_catalogue:
    :param effective_radius_key_sources_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param position_angle_sources_catalogue:
    :param initial_positions_choice:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :param initial_axis_ratios_choice:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties(source_galaxies_catalogue, sci_image_filename, waveband,
                                         ra_key_sources_catalogue, dec_key_sources_catalogue,
                                         x_key_sources_catalogue, y_key_sources_catalogue,
                                         magnitude_key_sources_catalogue,
                                         effective_radius_key_sources_catalogue, axis_ratio_key_sources_catalogue,
                                         minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                         position_angle_sources_catalogue, initial_positions_choice,
                                         initial_axis_ratios_choice)
    light_profiles = np.full(len(source_galaxies_catalogue), 'sersic')
    light_profiles = np.hstack((light_profiles, light_profiles))
    source_positions = np.vstack((source_positions, source_positions))
    ra = np.hstack((ra, ra))
    dec = np.hstack((dec, dec))
    total_magnitude = np.vstack((total_magnitude, total_magnitude))
    effective_radius = np.vstack((effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_double_sersic(source_galaxies_catalogue,
                                                                sersic_index_key_sources_catalogue, waveband,
                                                                initial_sersic_indices_choice,
                                                                initial_sersic_indices_value)
    subtract = np.full((len(source_galaxies_catalogue)) * 2, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_triple_sersic(source_galaxies_catalogue, sci_image_filename, waveband,
                                    ra_key_sources_catalogue, dec_key_sources_catalogue,
                                    x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                                    effective_radius_key_sources_catalogue, sersic_index_key_sources_catalogue,
                                    axis_ratio_key_sources_catalogue, minor_axis_key_sources_catalogue,
                                    major_axis_key_sources_catalogue, position_angle_sources_catalogue,
                                    initial_positions_choice='RADEC', initial_sersic_indices_choice='fixed_value',
                                    initial_sersic_indices_value=2.5, initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param sci_image_filename:
    :param waveband:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param magnitude_key_sources_catalogue:
    :param effective_radius_key_sources_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param position_angle_sources_catalogue:
    :param initial_positions_choice:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :param initial_axis_ratios_choice:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties(source_galaxies_catalogue, sci_image_filename, waveband,
                                         ra_key_sources_catalogue, dec_key_sources_catalogue,
                                         x_key_sources_catalogue, y_key_sources_catalogue,
                                         magnitude_key_sources_catalogue,
                                         effective_radius_key_sources_catalogue, axis_ratio_key_sources_catalogue,
                                         minor_axis_key_sources_catalogue, major_axis_key_sources_catalogue,
                                         position_angle_sources_catalogue, initial_positions_choice,
                                         initial_axis_ratios_choice)
    light_profiles = np.full(len(source_galaxies_catalogue), 'sersic')
    light_profiles = np.hstack((light_profiles, light_profiles, light_profiles))
    source_positions = np.vstack((source_positions, source_positions, source_positions))
    ra = np.hstack((ra, ra, ra))
    dec = np.hstack((dec, dec, dec))
    total_magnitude = np.vstack((total_magnitude, total_magnitude, total_magnitude))
    effective_radius = np.vstack((effective_radius, effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_triple_sersic(source_galaxies_catalogue,
                                                                sersic_index_key_sources_catalogue, waveband,
                                                                initial_sersic_indices_choice,
                                                                initial_sersic_indices_value)
    subtract = np.full((len(source_galaxies_catalogue)) * 3, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_for_galfit(source_galaxies_catalogue, sci_image_filename, waveband,
                                 ra_key_sources_catalogue, dec_key_sources_catalogue,
                                 x_key_sources_catalogue, y_key_sources_catalogue, magnitude_key_sources_catalogue,
                                 effective_radius_key_sources_catalogue, sersic_index_key_sources_catalogue,
                                 axis_ratio_key_sources_catalogue, minor_axis_key_sources_catalogue,
                                 major_axis_key_sources_catalogue, position_angle_sources_catalogue, light_profile_key,
                                 initial_positions_choice='RADEC', initial_sersic_indices_choice='fixed_value',
                                 initial_sersic_indices_value=2.5, initial_axis_ratios_choice='single_value'):
    """

    :param source_galaxies_catalogue:
    :param sci_image_filename:
    :param waveband:
    :param ra_key_sources_catalogue:
    :param dec_key_sources_catalogue:
    :param x_key_sources_catalogue:
    :param y_key_sources_catalogue:
    :param magnitude_key_sources_catalogue:
    :param effective_radius_key_sources_catalogue:
    :param sersic_index_key_sources_catalogue:
    :param axis_ratio_key_sources_catalogue:
    :param minor_axis_key_sources_catalogue:
    :param major_axis_key_sources_catalogue:
    :param position_angle_sources_catalogue:
    :param light_profile_key:
    :param initial_positions_choice:
    :param initial_sersic_indices_choice:
    :param initial_sersic_indices_value:
    :param initial_axis_ratios_choice:
    :return:
    """

    format_properties_switcher = {'sersic': format_properties_sersic,
                                  'devauc': format_properties_devauc,
                                  'expdisk': format_properties_expdisk,
                                  'double_sersic': format_properties_double_sersic,
                                  'triple_sersic': format_properties_triple_sersic,
                                  'sersic_expdisk': format_properties_sersic_expdisk,
                                  'devauc_expdisk': format_properties_devauc_expdisk}

    format_properties_function = format_properties_switcher.get(light_profile_key, lambda: 'To be implemented...')
    light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract = format_properties_function(source_galaxies_catalogue, sci_image_filename, waveband,
                                                              ra_key_sources_catalogue, dec_key_sources_catalogue,
                                                              x_key_sources_catalogue, y_key_sources_catalogue,
                                                              magnitude_key_sources_catalogue,
                                                              effective_radius_key_sources_catalogue,
                                                              sersic_index_key_sources_catalogue,
                                                              axis_ratio_key_sources_catalogue,
                                                              minor_axis_key_sources_catalogue,
                                                              major_axis_key_sources_catalogue,
                                                              position_angle_sources_catalogue,
                                                              initial_positions_choice, initial_sersic_indices_choice,
                                                              initial_sersic_indices_value, initial_axis_ratios_choice)

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_for_regions_galfit_single_sersic(region_sci_image_filename, source_catalogue, waveband,
                                                       ra_keyword='ALPHAWIN_J2000', dec_keyword='DELTAWIN_J2000',
                                                       waveband_keyword='f814w', initial_positions_choice='sextractor'):
    """

    :param region_sci_image_filename:
    :param source_catalogue:
    :param waveband:
    :param ra_keyword:
    :param dec_keyword:
    :param waveband_keyword:
    :param initial_positions_choice:
    :return:
    """

    light_profiles = np.full(len(source_catalogue), 'sersic')
    source_positions = np.empty((len(source_catalogue), 4))
    ra = np.empty(len(source_catalogue))
    dec = np.empty(len(source_catalogue))
    total_magnitude = np.empty((len(source_catalogue), 2))
    effective_radius = np.empty((len(source_catalogue), 2))
    sersic_index = np.empty((len(source_catalogue), 2))
    axis_ratio = np.empty((len(source_catalogue), 2))
    position_angle = np.empty((len(source_catalogue), 2))
    subtract = np.full(len(source_catalogue), '0')
    tofit = 1

    for i in range(len(source_catalogue)):
        ra[i] = source_catalogue['{}_{}'.format(ra_keyword, waveband_keyword)][i]
        dec[i] = source_catalogue['{}_{}'.format(dec_keyword, waveband_keyword)][i]

        if initial_positions_choice == 'sextractor':
            x, y = single_ra_dec_2_xy(ra[i], dec[i], region_sci_image_filename)
        elif initial_positions_choice == 'previous_iteration':
            x = source_catalogue['x_galfit_{}'.format(waveband)][i]
            y = source_catalogue['y_galfit_{}'.format(waveband)][i]
        else:
            raise ValueError

        source_positions[i, :] = [x, y, tofit, tofit]
        total_magnitude[i, :] = [source_catalogue['mag_galfit_{}'.format(waveband)][i], tofit]
        effective_radius[i, :] = [source_catalogue['re_galfit_{}'.format(waveband)][i], tofit]
        sersic_index[i, :] = [source_catalogue['n_galfit_{}'.format(waveband)][i], tofit]
        axis_ratio[i, :] = [source_catalogue['ar_galfit_{}'.format(waveband)][i], tofit]
        position_angle[i, :] = [source_catalogue['pa_galfit_{}'.format(waveband)][i], tofit]

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


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


def copy_files_to_working_directory(working_directory, input_galfit_filename, sci_image_filename,
                                    sigma_image_filename, psf_image_filename, bad_pixel_mask_filename,
                                    constraints_file_filename):
    """

    :param working_directory:
    :param input_galfit_filename:
    :param sci_image_filename:
    :param sigma_image_filename:
    :param psf_image_filename:
    :param bad_pixel_mask_filename:
    :param constraints_file_filename:
    :return None:
    """

    files_to_copy = [input_galfit_filename, sci_image_filename, sigma_image_filename, psf_image_filename,
                     bad_pixel_mask_filename, constraints_file_filename]

    subprocess.run(['cp'] + files_to_copy + [working_directory])


def remove_files_from_working_directory(input_galfit_filename, sci_image_filename, output_model_image_filename,
                                        sigma_image_filename, psf_image_filename, bad_pixel_mask_filename,
                                        constraints_file_filename):
    """

    :param input_galfit_filename:
    :param sci_image_filename:
    :param output_model_image_filename:
    :param sigma_image_filename:
    :param psf_image_filename:
    :param bad_pixel_mask_filename:
    :param constraints_file_filename:
    :return None.
    """

    files_to_copy = [os.path.basename(output_model_image_filename), 'fit.log', 'galfit.01']

    subprocess.run(['cp'] + files_to_copy + [os.path.dirname(output_model_image_filename)])

    files_to_remove = ['fit.log', 'galfit.01', os.path.basename(input_galfit_filename),
                       os.path.basename(sci_image_filename), os.path.basename(output_model_image_filename),
                       os.path.basename(sigma_image_filename), os.path.basename(psf_image_filename),
                       os.path.basename(bad_pixel_mask_filename), os.path.basename(constraints_file_filename)]

    subprocess.run(['rm'] + files_to_remove)


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


def run_galfit_on_euler(galfit_binary_file, input_galfit_filename, working_directory):
    """
    This function runs GALFIT from the command line.

    :param galfit_binary_file:
    :param input_galfit_filename:
    :param working_directory:
    :return None.
    """

    subprocess.run(['cp', galfit_binary_file, working_directory])
    subprocess.run([os.path.join(working_directory, os.path.basename(galfit_binary_file)), input_galfit_filename])


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
