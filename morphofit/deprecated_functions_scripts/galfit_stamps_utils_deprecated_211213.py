#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import numpy as np

# morphofit imports
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import fitting_choice
from morphofit.utils import single_ra_dec_2_xy
from morphofit.utils import get_logger

logger = get_logger(__file__)


def format_position_for_galfit_stamps(sci_image_stamp_filename, ra_source, dec_source,
                                      neighbouring_sources_catalogue,
                                      ra_key_neighbouring_sources_catalogue,
                                      dec_key_neighbouring_sources_catalogue, image_size):
    """

    :param sci_image_stamp_filename:
    :param ra_source:
    :param dec_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    source_positions = np.empty((len(neighbouring_sources_catalogue) + 1, 4))
    x_source, y_source = single_ra_dec_2_xy(ra_source, dec_source, sci_image_stamp_filename)
    source_positions[0, :] = [x_source, y_source, 1, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        source_positions[i + 1, :] = [x_neighbouring_galaxy, y_neighbouring_galaxy, tofit, tofit]

    # x, y = single_ra_dec_2_xy(ra[i], dec[i], image_filename)
    # source_positions = np.empty((len(x_neighbouring_galaxies) + 1, 4))
    # # source_positions[0, :] = [x_source, y_source, 1, 1]
    # for i in range(len(x_neighbouring_galaxies)):
    #     tofit = fitting_choice(x_neighbouring_galaxies[i], y_neighbouring_galaxies[i], image_size)
    #     source_positions[i + 1, :] = [x_neighbouring_galaxies[i], y_neighbouring_galaxies[i], tofit, tofit]

    return source_positions


def format_ra_dec_for_galfit_stamps(ra_source, dec_source, neighbouring_sources_catalogue,
                                    ra_key_neighbouring_sources_catalogue,
                                    dec_key_neighbouring_sources_catalogue):
    """

    :param ra_source:
    :param dec_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :return:
    """

    ra = np.empty(len(neighbouring_sources_catalogue) + 1)
    dec = np.empty(len(neighbouring_sources_catalogue) + 1)
    ra[0] = ra_source
    dec[0] = dec_source
    for i in range(len(neighbouring_sources_catalogue)):
        ra[i + 1] = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec[i + 1] = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]

    return ra, dec


def format_magnitude_for_galfit_stamps(sci_image_stamp_filename, mag_source, neighbouring_sources_catalogue,
                                       mag_key_neighbouring_sources_catalogue,
                                       ra_key_neighbouring_sources_catalogue,
                                       dec_key_neighbouring_sources_catalogue,
                                       image_size):
    """

    :param sci_image_stamp_filename:
    :param mag_source:
    :param neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    total_magnitude = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    if mag_source != 99.:
        total_magnitude[0, :] = [mag_source, 1]
    else:
        total_magnitude[0, :] = [27., 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        if neighbouring_sources_catalogue[mag_key_neighbouring_sources_catalogue][i] != 99.:
            total_magnitude[i + 1, :] = [neighbouring_sources_catalogue[mag_key_neighbouring_sources_catalogue][i],
                                         tofit]
        else:
            total_magnitude[i + 1, :] = [27., tofit]

    return total_magnitude


def format_effective_radius_for_galfit_stamps(sci_image_stamp_filename, effective_radius_source,
                                              neighbouring_sources_catalogue,
                                              re_key_neighbouring_sources_catalogue,
                                              ra_key_neighbouring_sources_catalogue,
                                              dec_key_neighbouring_sources_catalogue,
                                              image_size):
    """

    :param sci_image_stamp_filename:
    :param effective_radius_source:
    :param neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    effective_radius = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    if effective_radius_source < 0.:
        effective_radius[0, :] = [5., 1]
    else:
        effective_radius[0, :] = [effective_radius_source, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        if neighbouring_sources_catalogue[re_key_neighbouring_sources_catalogue][i] < 0.:
            effective_radius[i + 1, :] = [5., tofit]
        else:
            effective_radius[i + 1, :] = [neighbouring_sources_catalogue[re_key_neighbouring_sources_catalogue][i],
                                          tofit]

    return effective_radius


def format_sersic_index_for_galfit_stamps_single_sersic(sci_image_stamp_filename, neighbouring_sources_catalogue,
                                                        ra_key_neighbouring_sources_catalogue,
                                                        dec_key_neighbouring_sources_catalogue,
                                                        image_size):
    """

    :param sci_image_stamp_filename:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    sersic_index = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index[0, :] = [2.5, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        sersic_index[i + 1, :] = [2.5, tofit]

    return sersic_index


def format_sersic_index_for_galfit_stamps_devauc(neighbouring_sources_catalogue):
    """

    :param neighbouring_sources_catalogue:
    :return:
    """

    sersic_index = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index[0, :] = [4, 0]
    for i in range(len(neighbouring_sources_catalogue)):
        sersic_index[i + 1, :] = [4, 0]

    return sersic_index


def format_sersic_index_for_galfit_stamps_expdisk(neighbouring_sources_catalogue):
    """

    :param neighbouring_sources_catalogue:
    :return:
    """

    sersic_index = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index[0, :] = [1, 0]
    for i in range(len(neighbouring_sources_catalogue)):
        sersic_index[i + 1, :] = [1, 0]

    return sersic_index


def format_sersic_index_for_galfit_stamps_double_sersic(sci_image_stamp_filename, neighbouring_sources_catalogue,
                                                        ra_key_neighbouring_sources_catalogue,
                                                        dec_key_neighbouring_sources_catalogue,
                                                        image_size):
    """

    :param sci_image_stamp_filename:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    sersic_index_sersic1 = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_sersic2 = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_sersic1[0, :] = [2.5, 1]
    sersic_index_sersic2[0, :] = [2.5, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        sersic_index_sersic1[i + 1, :] = [2.5, tofit]
        sersic_index_sersic2[i + 1, :] = [2.5, tofit]
    sersic_index = np.vstack((sersic_index_sersic1, sersic_index_sersic2))

    return sersic_index


def format_sersic_index_for_galfit_stamps_triple_sersic(sci_image_stamp_filename, neighbouring_sources_catalogue,
                                                        ra_key_neighbouring_sources_catalogue,
                                                        dec_key_neighbouring_sources_catalogue,
                                                        image_size):
    """

    :param sci_image_stamp_filename:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    sersic_index_sersic1 = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_sersic2 = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_sersic3 = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_sersic1[0, :] = [2.5, 1]
    sersic_index_sersic2[0, :] = [2.5, 1]
    sersic_index_sersic3[0, :] = [2.5, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        sersic_index_sersic1[i + 1, :] = [2.5, tofit]
        sersic_index_sersic2[i + 1, :] = [2.5, tofit]
        sersic_index_sersic3[i + 1, :] = [2.5, tofit]
    sersic_index = np.vstack((sersic_index_sersic1, sersic_index_sersic2, sersic_index_sersic3))

    return sersic_index


def format_sersic_index_for_galfit_stamps_sersic_expdisk(sci_image_stamp_filename, neighbouring_sources_catalogue,
                                                         ra_key_neighbouring_sources_catalogue,
                                                         dec_key_neighbouring_sources_catalogue,
                                                         image_size):
    """

    :param sci_image_stamp_filename:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    sersic_index_sersic = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_expdisk = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_sersic[0, :] = [2.5, 1]
    sersic_index_expdisk[0, :] = [1.0, 0]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        sersic_index_sersic[i + 1, :] = [2.5, tofit]
        sersic_index_expdisk[i + 1, :] = [1.0, 0]
    sersic_index = np.vstack((sersic_index_sersic, sersic_index_expdisk))

    return sersic_index


def format_sersic_index_for_galfit_stamps_devauc_expdisk(neighbouring_sources_catalogue):
    """

    :param neighbouring_sources_catalogue:
    :return:
    """

    sersic_index_devauc = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_expdisk = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index_devauc[0, :] = [4.0, 0]
    sersic_index_expdisk[0, :] = [1.0, 0]
    for i in range(len(neighbouring_sources_catalogue)):
        sersic_index_devauc[i + 1, :] = [4.0, 0]
        sersic_index_expdisk[i + 1, :] = [1.0, 0]
    sersic_index = np.vstack((sersic_index_devauc, sersic_index_expdisk))

    return sersic_index


def format_axis_ratio_for_galfit_stamps(sci_image_stamp_filename, minor_axis_source, major_axis_source,
                                        neighbouring_sources_catalogue,
                                        minor_axis_key_neighbouring_sources_catalogue,
                                        major_axis_key_neighbouring_sources_catalogue,
                                        ra_key_neighbouring_sources_catalogue,
                                        dec_key_neighbouring_sources_catalogue,
                                        image_size):
    """

    :param sci_image_stamp_filename:
    :param minor_axis_source:
    :param major_axis_source:
    :param neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    axis_ratio = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    axis_ratio[0, :] = [(minor_axis_source / major_axis_source) ** 2, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        axis_ratio[i + 1, :] = [(neighbouring_sources_catalogue[minor_axis_key_neighbouring_sources_catalogue][i] /
                                neighbouring_sources_catalogue[major_axis_key_neighbouring_sources_catalogue][i]) ** 2,
                                tofit]

    return axis_ratio


def format_position_angle_for_galfit_stamps(sci_image_stamp_filename, position_angle_source,
                                            neighbouring_sources_catalogue,
                                            position_angle_key_neighbouring_sources_catalogue,
                                            ra_key_neighbouring_sources_catalogue,
                                            dec_key_neighbouring_sources_catalogue,
                                            image_size):
    """

    :param sci_image_stamp_filename:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param image_size:
    :return:
    """

    position_angle = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    position_angle[0, :] = [position_angle_source, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        position_angle[i + 1, :] = [neighbouring_sources_catalogue[position_angle_key_neighbouring_sources_catalogue]
                                    [i], tofit]

    return position_angle


def format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                            mag_source, effective_radius_source, minor_axis_source, major_axis_source,
                                            position_angle_source, neighbouring_sources_catalogue,
                                            ra_key_neighbouring_sources_catalogue,
                                            dec_key_neighbouring_sources_catalogue,
                                            mag_key_neighbouring_sources_catalogue,
                                            re_key_neighbouring_sources_catalogue,
                                            minor_axis_key_neighbouring_sources_catalogue,
                                            major_axis_key_neighbouring_sources_catalogue,
                                            position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions = format_position_for_galfit_stamps(sci_image_stamp_filename, ra_source, dec_source,
                                                         neighbouring_sources_catalogue,
                                                         ra_key_neighbouring_sources_catalogue,
                                                         dec_key_neighbouring_sources_catalogue, image_size)
    ra, dec = format_ra_dec_for_galfit_stamps(ra_source, dec_source, neighbouring_sources_catalogue,
                                              ra_key_neighbouring_sources_catalogue,
                                              dec_key_neighbouring_sources_catalogue)
    total_magnitude = format_magnitude_for_galfit_stamps(sci_image_stamp_filename, mag_source,
                                                         neighbouring_sources_catalogue,
                                                         mag_key_neighbouring_sources_catalogue + '_' + waveband,
                                                         ra_key_neighbouring_sources_catalogue,
                                                         dec_key_neighbouring_sources_catalogue,
                                                         image_size)
    effective_radius = format_effective_radius_for_galfit_stamps(sci_image_stamp_filename,
                                                                 effective_radius_source,
                                                                 neighbouring_sources_catalogue,
                                                                 re_key_neighbouring_sources_catalogue +
                                                                 '_' + waveband,
                                                                 ra_key_neighbouring_sources_catalogue,
                                                                 dec_key_neighbouring_sources_catalogue,
                                                                 image_size)
    axis_ratio = format_axis_ratio_for_galfit_stamps(sci_image_stamp_filename, minor_axis_source,
                                                     major_axis_source,
                                                     neighbouring_sources_catalogue,
                                                     minor_axis_key_neighbouring_sources_catalogue +
                                                     '_' + waveband,
                                                     major_axis_key_neighbouring_sources_catalogue +
                                                     '_' + waveband,
                                                     ra_key_neighbouring_sources_catalogue,
                                                     dec_key_neighbouring_sources_catalogue,
                                                     image_size)
    position_angle = format_position_angle_for_galfit_stamps(sci_image_stamp_filename, position_angle_source,
                                                             neighbouring_sources_catalogue,
                                                             position_angle_key_neighbouring_sources_catalogue +
                                                             '_' + waveband,
                                                             ra_key_neighbouring_sources_catalogue,
                                                             dec_key_neighbouring_sources_catalogue,
                                                             image_size)

    return source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle


def format_properties_sersic_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source, mag_source,
                                    effective_radius_source, minor_axis_source, major_axis_source,
                                    position_angle_source, neighbouring_sources_catalogue,
                                    ra_key_neighbouring_sources_catalogue,
                                    dec_key_neighbouring_sources_catalogue, mag_key_neighbouring_sources_catalogue,
                                    re_key_neighbouring_sources_catalogue,
                                    minor_axis_key_neighbouring_sources_catalogue,
                                    major_axis_key_neighbouring_sources_catalogue,
                                    position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle =\
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue, ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    sersic_index = format_sersic_index_for_galfit_stamps_single_sersic(sci_image_stamp_filename,
                                                                       neighbouring_sources_catalogue,
                                                                       ra_key_neighbouring_sources_catalogue,
                                                                       dec_key_neighbouring_sources_catalogue,
                                                                       image_size)
    subtract = np.full(len(neighbouring_sources_catalogue) + 1, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_devauc_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source, mag_source,
                                    effective_radius_source, minor_axis_source, major_axis_source,
                                    position_angle_source, neighbouring_sources_catalogue,
                                    ra_key_neighbouring_sources_catalogue,
                                    dec_key_neighbouring_sources_catalogue,
                                    mag_key_neighbouring_sources_catalogue,
                                    re_key_neighbouring_sources_catalogue,
                                    minor_axis_key_neighbouring_sources_catalogue,
                                    major_axis_key_neighbouring_sources_catalogue,
                                    position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue,
                                                ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles = np.full(len(neighbouring_sources_catalogue) + 1, 'devauc')
    sersic_index = format_sersic_index_for_galfit_stamps_devauc(neighbouring_sources_catalogue)
    subtract = np.full(len(neighbouring_sources_catalogue) + 1, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_expdisk_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source, mag_source,
                                     effective_radius_source, minor_axis_source, major_axis_source,
                                     position_angle_source, neighbouring_sources_catalogue,
                                     ra_key_neighbouring_sources_catalogue,
                                     dec_key_neighbouring_sources_catalogue,
                                     mag_key_neighbouring_sources_catalogue,
                                     re_key_neighbouring_sources_catalogue,
                                     minor_axis_key_neighbouring_sources_catalogue,
                                     major_axis_key_neighbouring_sources_catalogue,
                                     position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue,
                                                ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles = np.full(len(neighbouring_sources_catalogue) + 1, 'expdisk')
    sersic_index = format_sersic_index_for_galfit_stamps_expdisk(neighbouring_sources_catalogue)
    subtract = np.full(len(neighbouring_sources_catalogue) + 1, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_double_sersic_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                           mag_source, effective_radius_source, minor_axis_source, major_axis_source,
                                           position_angle_source, neighbouring_sources_catalogue,
                                           ra_key_neighbouring_sources_catalogue,
                                           dec_key_neighbouring_sources_catalogue,
                                           mag_key_neighbouring_sources_catalogue,
                                           re_key_neighbouring_sources_catalogue,
                                           minor_axis_key_neighbouring_sources_catalogue,
                                           major_axis_key_neighbouring_sources_catalogue,
                                           position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue,
                                                ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles_sersic1 = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles_sersic2 = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles = np.hstack((light_profiles_sersic1, light_profiles_sersic2))
    source_positions = np.vstack((source_positions, source_positions))
    ra = np.hstack((ra, ra))
    dec = np.hstack((dec, dec))
    total_magnitude = np.vstack((total_magnitude, total_magnitude))
    effective_radius = np.vstack((effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_stamps_double_sersic(sci_image_stamp_filename,
                                                                       neighbouring_sources_catalogue,
                                                                       ra_key_neighbouring_sources_catalogue,
                                                                       dec_key_neighbouring_sources_catalogue,
                                                                       image_size)
    axis_ratio = np.vstack((axis_ratio, axis_ratio))
    position_angle = np.vstack((position_angle, position_angle))
    subtract = np.full((len(neighbouring_sources_catalogue) + 1) * 2, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_triple_sersic_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                           mag_source, effective_radius_source, minor_axis_source, major_axis_source,
                                           position_angle_source, neighbouring_sources_catalogue,
                                           ra_key_neighbouring_sources_catalogue,
                                           dec_key_neighbouring_sources_catalogue,
                                           mag_key_neighbouring_sources_catalogue,
                                           re_key_neighbouring_sources_catalogue,
                                           minor_axis_key_neighbouring_sources_catalogue,
                                           major_axis_key_neighbouring_sources_catalogue,
                                           position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue,
                                                ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles_sersic1 = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles_sersic2 = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles_sersic3 = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles = np.hstack((light_profiles_sersic1, light_profiles_sersic2, light_profiles_sersic3))
    source_positions = np.vstack((source_positions, source_positions, source_positions))
    ra = np.hstack((ra, ra, ra))
    dec = np.hstack((dec, dec, dec))
    total_magnitude = np.vstack((total_magnitude, total_magnitude, total_magnitude))
    effective_radius = np.vstack((effective_radius, effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_stamps_triple_sersic(sci_image_stamp_filename,
                                                                       neighbouring_sources_catalogue,
                                                                       ra_key_neighbouring_sources_catalogue,
                                                                       dec_key_neighbouring_sources_catalogue,
                                                                       image_size)
    axis_ratio = np.vstack((axis_ratio, axis_ratio, axis_ratio))
    position_angle = np.vstack((position_angle, position_angle, position_angle))
    subtract = np.full((len(neighbouring_sources_catalogue) + 1) * 3, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_sersic_expdisk_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                            mag_source, effective_radius_source, minor_axis_source, major_axis_source,
                                            position_angle_source, neighbouring_sources_catalogue,
                                            ra_key_neighbouring_sources_catalogue,
                                            dec_key_neighbouring_sources_catalogue,
                                            mag_key_neighbouring_sources_catalogue,
                                            re_key_neighbouring_sources_catalogue,
                                            minor_axis_key_neighbouring_sources_catalogue,
                                            major_axis_key_neighbouring_sources_catalogue,
                                            position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue,
                                                ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles_sersic = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles_expdisk = np.full(len(neighbouring_sources_catalogue) + 1, 'expdisk')
    light_profiles = np.hstack((light_profiles_sersic, light_profiles_expdisk))
    source_positions = np.vstack((source_positions, source_positions))
    ra = np.hstack((ra, ra))
    dec = np.hstack((dec, dec))
    total_magnitude = np.vstack((total_magnitude, total_magnitude))
    effective_radius = np.vstack((effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_stamps_sersic_expdisk(sci_image_stamp_filename,
                                                                        neighbouring_sources_catalogue,
                                                                        ra_key_neighbouring_sources_catalogue,
                                                                        dec_key_neighbouring_sources_catalogue,
                                                                        image_size)
    axis_ratio = np.vstack((axis_ratio, axis_ratio))
    position_angle = np.vstack((position_angle, position_angle))
    subtract = np.full((len(neighbouring_sources_catalogue) + 1) * 2, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_devauc_expdisk_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                            mag_source, effective_radius_source, minor_axis_source, major_axis_source,
                                            position_angle_source, neighbouring_sources_catalogue,
                                            ra_key_neighbouring_sources_catalogue,
                                            dec_key_neighbouring_sources_catalogue,
                                            mag_key_neighbouring_sources_catalogue,
                                            re_key_neighbouring_sources_catalogue,
                                            minor_axis_key_neighbouring_sources_catalogue,
                                            major_axis_key_neighbouring_sources_catalogue,
                                            position_angle_key_neighbouring_sources_catalogue):
    """

    :param sci_image_stamp_filename:
    :param image_size:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :return:
    """

    source_positions, ra, dec, total_magnitude, effective_radius, axis_ratio, position_angle = \
        format_common_profile_properties_stamps(sci_image_stamp_filename, image_size, waveband, ra_source, dec_source,
                                                mag_source, effective_radius_source, minor_axis_source,
                                                major_axis_source, position_angle_source,
                                                neighbouring_sources_catalogue,
                                                ra_key_neighbouring_sources_catalogue,
                                                dec_key_neighbouring_sources_catalogue,
                                                mag_key_neighbouring_sources_catalogue,
                                                re_key_neighbouring_sources_catalogue,
                                                minor_axis_key_neighbouring_sources_catalogue,
                                                major_axis_key_neighbouring_sources_catalogue,
                                                position_angle_key_neighbouring_sources_catalogue)
    light_profiles_devauc = np.full(len(neighbouring_sources_catalogue) + 1, 'devauc')
    light_profiles_expdisk = np.full(len(neighbouring_sources_catalogue) + 1, 'expdisk')
    light_profiles = np.hstack((light_profiles_devauc, light_profiles_expdisk))
    source_positions = np.vstack((source_positions, source_positions))
    ra = np.hstack((ra, ra))
    dec = np.hstack((dec, dec))
    total_magnitude = np.vstack((total_magnitude, total_magnitude))
    effective_radius = np.vstack((effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_stamps_devauc_expdisk(neighbouring_sources_catalogue)
    axis_ratio = np.vstack((axis_ratio, axis_ratio))
    position_angle = np.vstack((position_angle, position_angle))
    subtract = np.full((len(neighbouring_sources_catalogue) + 1) * 2, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_for_galfit_on_stamps(sci_image_stamp_filename, waveband, ra_source,
                                           dec_source, mag_source,
                                           effective_radius_source, minor_axis_source, major_axis_source,
                                           position_angle_source, neighbouring_sources_catalogue,
                                           ra_key_neighbouring_sources_catalogue,
                                           dec_key_neighbouring_sources_catalogue,
                                           mag_key_neighbouring_sources_catalogue,
                                           re_key_neighbouring_sources_catalogue,
                                           minor_axis_key_neighbouring_sources_catalogue,
                                           major_axis_key_neighbouring_sources_catalogue,
                                           position_angle_key_neighbouring_sources_catalogue,
                                           enlarging_image_factor, light_profile_key):
    """
    For the bulge+disk decomposition, we assume that the bulge and disk component have the same initial centroids,
    magnitudes and half-light/scale radii from SExtractor MAG_AUTO and FLUX_RADIUS. Same applies for axis ratios and
    position angles. The initial Sersic index for the bulge component is set to 2.5.

    :param sci_image_stamp_filename:
    :param waveband:
    :param ra_source:
    :param dec_source:
    :param mag_source:
    :param effective_radius_source:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :param neighbouring_sources_catalogue:
    :param ra_key_neighbouring_sources_catalogue:
    :param dec_key_neighbouring_sources_catalogue:
    :param mag_key_neighbouring_sources_catalogue:
    :param re_key_neighbouring_sources_catalogue:
    :param minor_axis_key_neighbouring_sources_catalogue:
    :param major_axis_key_neighbouring_sources_catalogue:
    :param position_angle_key_neighbouring_sources_catalogue:
    :param enlarging_image_factor:
    :param light_profile_key:
    :return:
    """

    axis_ratio = minor_axis_source / major_axis_source
    angle = position_angle_source * (2 * np.pi) / 360

    image_size_x = effective_radius_source * enlarging_image_factor * (abs(np.cos(angle)) + axis_ratio *
                                                                       abs(np.sin(angle)))
    image_size_y = effective_radius_source * enlarging_image_factor * (abs(np.sin(angle)) + axis_ratio *
                                                                       abs(np.cos(angle)))
    image_size = [image_size_x, image_size_y]

    format_properties_switcher = {'sersic': format_properties_sersic_stamps,
                                  'devauc': format_properties_devauc_stamps,
                                  'expdisk': format_properties_expdisk_stamps,
                                  'double_sersic': format_properties_double_sersic_stamps,
                                  'triple_sersic': format_properties_triple_sersic_stamps,
                                  'sersic_expdisk': format_properties_sersic_expdisk_stamps,
                                  'devauc_expdisk': format_properties_devauc_expdisk_stamps}

    format_properties_function = format_properties_switcher.get(light_profile_key, lambda: 'To be implemented...')
    light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract = format_properties_function(sci_image_stamp_filename, image_size, waveband, ra_source,
                                                              dec_source, mag_source, effective_radius_source,
                                                              minor_axis_source, major_axis_source,
                                                              position_angle_source, neighbouring_sources_catalogue,
                                                              ra_key_neighbouring_sources_catalogue,
                                                              dec_key_neighbouring_sources_catalogue,
                                                              mag_key_neighbouring_sources_catalogue,
                                                              re_key_neighbouring_sources_catalogue,
                                                              minor_axis_key_neighbouring_sources_catalogue,
                                                              major_axis_key_neighbouring_sources_catalogue,
                                                              position_angle_key_neighbouring_sources_catalogue)

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def get_number_fitted_components_stamps(source_galaxies_catalogue, light_profile_key):
    """

    :param source_galaxies_catalogue:
    :param light_profile_key:
    :return:
    """

    light_profile_key_switcher = {'sersic': len(source_galaxies_catalogue) + 1,
                                  'devauc': len(source_galaxies_catalogue) + 1,
                                  'expdisk': len(source_galaxies_catalogue) + 1,
                                  'double_sersic': (len(source_galaxies_catalogue) + 1) * 2,
                                  'triple_sersic': (len(source_galaxies_catalogue) + 1) * 3,
                                  'sersic_expdisk': (len(source_galaxies_catalogue) + 1) * 2,
                                  'devauc_expdisk': (len(source_galaxies_catalogue) + 1) * 2}

    n_fitted_components = light_profile_key_switcher.get(light_profile_key, lambda: 'To be implemented...')

    return n_fitted_components


def format_ucat_sims_properties_for_stamps_galfit_single_sersic(sci_image_stamp_filename, waveband, ra_source,
                                                                dec_source, mag_source,
                                                                effective_radius_source, minor_axis_source,
                                                                major_axis_source,
                                                                position_angle_source, neighbouring_sources_catalogue,
                                                                ra_key_neighbouring_sources_catalogue,
                                                                dec_key_neighbouring_sources_catalogue,
                                                                mag_key_neighbouring_sources_catalogue,
                                                                re_key_neighbouring_sources_catalogue,
                                                                minor_axis_key_neighbouring_sources_catalogue,
                                                                major_axis_key_neighbouring_sources_catalogue,
                                                                position_angle_key_neighbouring_sources_catalogue,
                                                                enlarging_image_factor, sources_catalogue):
    """

        :param sci_image_stamp_filename:
        :param waveband:
        :param ra_source:
        :param dec_source:
        :param mag_source:
        :param effective_radius_source:
        :param minor_axis_source:
        :param major_axis_source:
        :param position_angle_source:
        :param neighbouring_sources_catalogue:
        :param ra_key_neighbouring_sources_catalogue:
        :param dec_key_neighbouring_sources_catalogue:
        :param mag_key_neighbouring_sources_catalogue:
        :param re_key_neighbouring_sources_catalogue:
        :param minor_axis_key_neighbouring_sources_catalogue:
        :param major_axis_key_neighbouring_sources_catalogue:
        :param position_angle_key_neighbouring_sources_catalogue:
        :param enlarging_image_factor:
        :param sources_catalogue:
        :return:
        """

    axis_ratio = minor_axis_source / major_axis_source
    angle = position_angle_source * (2 * np.pi) / 360

    image_size_x = effective_radius_source * enlarging_image_factor * (abs(np.cos(angle)) + axis_ratio *
                                                                       abs(np.sin(angle)))
    image_size_y = effective_radius_source * enlarging_image_factor * (abs(np.sin(angle)) + axis_ratio *
                                                                       abs(np.cos(angle)))
    image_size = [image_size_x, image_size_y]

    light_profiles = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    source_positions = format_position_for_galfit_stamps(sci_image_stamp_filename, ra_source, dec_source,
                                                         neighbouring_sources_catalogue,
                                                         ra_key_neighbouring_sources_catalogue,
                                                         dec_key_neighbouring_sources_catalogue, image_size)
    ra, dec = format_ra_dec_for_galfit_stamps(ra_source, dec_source, neighbouring_sources_catalogue,
                                              ra_key_neighbouring_sources_catalogue,
                                              dec_key_neighbouring_sources_catalogue)
    total_magnitude = format_magnitude_for_galfit_stamps(sci_image_stamp_filename, mag_source,
                                                         neighbouring_sources_catalogue,
                                                         mag_key_neighbouring_sources_catalogue + '_' + waveband,
                                                         ra_key_neighbouring_sources_catalogue,
                                                         dec_key_neighbouring_sources_catalogue,
                                                         image_size)
    effective_radius = format_effective_radius_for_galfit_stamps(sci_image_stamp_filename,
                                                                 effective_radius_source,
                                                                 neighbouring_sources_catalogue,
                                                                 re_key_neighbouring_sources_catalogue +
                                                                 '_' + waveband,
                                                                 ra_key_neighbouring_sources_catalogue,
                                                                 dec_key_neighbouring_sources_catalogue,
                                                                 image_size)
    sersic_index_mask = np.where(sources_catalogue['mag_{}'.format(waveband)] == mag_source)
    sersic_index_source = sources_catalogue['sersic_n_f814w'][sersic_index_mask][0]
    sersic_index = format_ucat_sims_sersic_index_for_galfit(sci_image_stamp_filename,
                                                            neighbouring_sources_catalogue,
                                                            ra_key_neighbouring_sources_catalogue,
                                                            dec_key_neighbouring_sources_catalogue,
                                                            image_size, sersic_index_source)
    axis_ratio = format_axis_ratio_for_galfit_stamps(sci_image_stamp_filename, minor_axis_source,
                                                     major_axis_source,
                                                     neighbouring_sources_catalogue,
                                                     minor_axis_key_neighbouring_sources_catalogue +
                                                     '_' + waveband,
                                                     major_axis_key_neighbouring_sources_catalogue +
                                                     '_' + waveband,
                                                     ra_key_neighbouring_sources_catalogue,
                                                     dec_key_neighbouring_sources_catalogue,
                                                     image_size)
    position_angle = format_position_angle_for_galfit_stamps(sci_image_stamp_filename, position_angle_source,
                                                             neighbouring_sources_catalogue,
                                                             position_angle_key_neighbouring_sources_catalogue +
                                                             '_' + waveband,
                                                             ra_key_neighbouring_sources_catalogue,
                                                             dec_key_neighbouring_sources_catalogue,
                                                             image_size)
    subtract = np.full(len(neighbouring_sources_catalogue) + 1, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_ucat_sims_sersic_index_for_galfit(sci_image_stamp_filename, neighbouring_sources_catalogue,
                                             ra_key_neighbouring_sources_catalogue,
                                             dec_key_neighbouring_sources_catalogue,
                                             image_size, sersic_index_source):
    """

        :param sci_image_stamp_filename:
        :param neighbouring_sources_catalogue:
        :param ra_key_neighbouring_sources_catalogue:
        :param dec_key_neighbouring_sources_catalogue:
        :param image_size:
        :param sersic_index_source:
        :return:
        """

    sersic_index = np.empty((len(neighbouring_sources_catalogue) + 1, 2))
    sersic_index[0, :] = [sersic_index_source, 1]
    for i in range(len(neighbouring_sources_catalogue)):
        ra_neighbouring_galaxy = neighbouring_sources_catalogue[ra_key_neighbouring_sources_catalogue][i]
        dec_neighbouring_galaxy = neighbouring_sources_catalogue[dec_key_neighbouring_sources_catalogue][i]
        x_neighbouring_galaxy, y_neighbouring_galaxy = single_ra_dec_2_xy(ra_neighbouring_galaxy,
                                                                          dec_neighbouring_galaxy,
                                                                          sci_image_stamp_filename)
        tofit = fitting_choice(x_neighbouring_galaxy, y_neighbouring_galaxy, image_size)
        sersic_index[i + 1, :] = [neighbouring_sources_catalogue['sersic_n_f814w'][i], tofit]

    return sersic_index
