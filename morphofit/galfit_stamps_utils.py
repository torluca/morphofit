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


def format_positions_stamps(sci_image_filename, target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                            target_galaxies_keys, neighb_galaxies_keys, waveband, image_size):
    """
    tofitext 1 tofit 1 -> 1
    tofitext 1 tofit 0 -> 0
    tofitext 0 tofit 1 -> 0
    tofitext 0 tofit 0 -> 0

    source whose center falls outside image stamp are kept fixed and not fitted

    :param sci_image_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :param image_size:
    :return:
    """

    positions_target_galaxies = np.empty((len(target_galaxies_catalogue), 4))
    for i in range(len(target_galaxies_catalogue)):
        x_target, y_target = single_ra_dec_2_xy(target_galaxies_catalogue[target_galaxies_keys[3]][i],
                                                target_galaxies_catalogue[target_galaxies_keys[4]][i],
                                                sci_image_filename)
        positions_target_galaxies[i, :] = [x_target, y_target,
                                           target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                    waveband)][i],
                                           target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                    waveband)][i]]

    positions_neigh_galaxies = np.empty((len(neighbouring_source_galaxies_catalogue), 4))
    for i in range(len(neighbouring_source_galaxies_catalogue)):
        x_neigh, y_neigh = single_ra_dec_2_xy(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[1]][i],
                                              neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[2]][i],
                                              sci_image_filename)
        tofit = fitting_choice(x_neigh, y_neigh, image_size)
        if (neighbouring_source_galaxies_catalogue['{}_{}'.format(neighb_galaxies_keys[9],
                                                                  waveband)][i] == 1) & (tofit == 1):
            positions_neigh_galaxies[i, :] = [x_neigh, y_neigh, 1, 1]
        else:
            positions_neigh_galaxies[i, :] = [x_neigh, y_neigh, 0, 0]

    source_positions = np.vstack([positions_target_galaxies, positions_neigh_galaxies])

    return source_positions


def format_ra_dec_stamps(target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                         target_galaxies_keys, neighb_galaxies_keys):
    """

    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :return:
    """

    ra_target_galaxies = np.array(target_galaxies_catalogue[target_galaxies_keys[3]])
    dec_target_galaxies = np.array(target_galaxies_catalogue[target_galaxies_keys[4]])
    ra_neigh_galaxies = np.array(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[1]])
    dec_neigh_galaxies = np.array(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[2]])

    ra = np.hstack([ra_target_galaxies, ra_neigh_galaxies])
    dec = np.hstack([dec_target_galaxies, dec_neigh_galaxies])

    return ra, dec


def format_magnitudes_stamps(sci_image_filename, target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                             target_galaxies_keys, neighb_galaxies_keys, waveband, image_size):
    """

    :param sci_image_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :param image_size:
    :return:
    """

    magnitudes_target_galaxies = np.empty((len(target_galaxies_catalogue), 2))
    magnitudes_target_galaxies[:, 0] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[5],
                                                                                         waveband)])
    magnitudes_target_galaxies[:, 1] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                         waveband)])

    magnitudes_neigh_galaxies = np.empty((len(neighbouring_source_galaxies_catalogue), 2))
    for i in range(len(neighbouring_source_galaxies_catalogue)):
        x_neigh, y_neigh = single_ra_dec_2_xy(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[1]][i],
                                              neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[2]][i],
                                              sci_image_filename)
        tofit = fitting_choice(x_neigh, y_neigh, image_size)
        if (neighbouring_source_galaxies_catalogue['{}_{}'.format(neighb_galaxies_keys[9],
                                                                  waveband)][i] == 1) & (tofit == 1):
            magnitudes_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue['{}_{}'
                                                                                      .format(neighb_galaxies_keys[3],
                                                                                              waveband)][i], 1]
        else:
            magnitudes_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue['{}_{}'
                                                                                      .format(neighb_galaxies_keys[3],
                                                                                              waveband)][i], 0]

    total_magnitudes = np.vstack([magnitudes_target_galaxies, magnitudes_neigh_galaxies])

    return total_magnitudes


def format_effective_radii_stamps(sci_image_filename, target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                  target_galaxies_keys, neighb_galaxies_keys, waveband, image_size):
    """

    :param sci_image_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :param image_size:
    :return:
    """

    eff_radii_target_galaxies = np.empty((len(target_galaxies_catalogue), 2))
    eff_radii_target_galaxies[:, 0] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[6],
                                                                                        waveband)])
    eff_radii_target_galaxies[:, 1] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                        waveband)])

    eff_radii_neigh_galaxies = np.empty((len(neighbouring_source_galaxies_catalogue), 2))
    for i in range(len(neighbouring_source_galaxies_catalogue)):
        x_neigh, y_neigh = single_ra_dec_2_xy(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[1]][i],
                                              neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[2]][i],
                                              sci_image_filename)
        tofit = fitting_choice(x_neigh, y_neigh, image_size)
        if (neighbouring_source_galaxies_catalogue['{}_{}'.format(neighb_galaxies_keys[9],
                                                                  waveband)][i] == 1) & (tofit == 1):
            eff_radii_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue['{}_{}'
                                                                                     .format(neighb_galaxies_keys[4],
                                                                                             waveband)][i], 1]
        else:
            eff_radii_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue['{}_{}'
                                                                                     .format(neighb_galaxies_keys[4],
                                                                                             waveband)][i], 0]

    effective_radii = np.vstack([eff_radii_target_galaxies, eff_radii_neigh_galaxies])

    return effective_radii


def format_axis_ratios_stamps(sci_image_filename, target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                              target_galaxies_keys, neighb_galaxies_keys, waveband, image_size):
    """

    :param sci_image_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :param image_size:
    :return:
    """

    axis_ratios_target_galaxies = np.empty((len(target_galaxies_catalogue), 2))
    if target_galaxies_keys[9] == target_galaxies_keys[10]:
        axis_ratios_target_galaxies[:, 0] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[9],
                                                                                              waveband)])
    else:
        axis_ratios_target_galaxies[:, 0] = (np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[9],
                                                                                               waveband)]) /
                                             np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[10],
                                                                                               waveband)]))**2
    axis_ratios_target_galaxies[:, 1] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                          waveband)])

    axis_ratios_neigh_galaxies = np.empty((len(neighbouring_source_galaxies_catalogue), 2))
    for i in range(len(neighbouring_source_galaxies_catalogue)):
        x_neigh, y_neigh = single_ra_dec_2_xy(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[1]][i],
                                              neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[2]][i],
                                              sci_image_filename)
        tofit = fitting_choice(x_neigh, y_neigh, image_size)
        if (neighbouring_source_galaxies_catalogue['{}_{}'.format(neighb_galaxies_keys[9], waveband)][i] == 1) &\
                (tofit == 1):
            if neighb_galaxies_keys[6] == neighb_galaxies_keys[7]:
                axis_ratios_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue[
                                                    '{}_{}'.format(neighb_galaxies_keys[6], waveband)][i], 1]
            else:
                axis_ratios_neigh_galaxies[i, :] = [(neighbouring_source_galaxies_catalogue[
                                                    '{}_{}'.format(neighb_galaxies_keys[6], waveband)][i] /
                                                     neighbouring_source_galaxies_catalogue[
                                                    '{}_{}'.format(neighb_galaxies_keys[7], waveband)][i])**2, 1]
        else:
            if neighb_galaxies_keys[6] == neighb_galaxies_keys[7]:
                axis_ratios_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue[
                                                    '{}_{}'.format(neighb_galaxies_keys[6], waveband)][i], 0]
            else:
                axis_ratios_neigh_galaxies[i, :] = [(neighbouring_source_galaxies_catalogue[
                                                    '{}_{}'.format(neighb_galaxies_keys[6], waveband)][i] /
                                                     neighbouring_source_galaxies_catalogue[
                                                    '{}_{}'.format(neighb_galaxies_keys[7], waveband)][i]) ** 2, 0]

    axis_ratios = np.vstack([axis_ratios_target_galaxies, axis_ratios_neigh_galaxies])

    return axis_ratios


def format_position_angles_stamps(sci_image_filename, target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                  target_galaxies_keys, neighb_galaxies_keys, waveband, image_size):
    """

    :param sci_image_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :param image_size:
    :return:
    """

    pos_angles_target_galaxies = np.empty((len(target_galaxies_catalogue), 2))
    pos_angles_target_galaxies[:, 0] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[11],
                                                                                         waveband)])
    pos_angles_target_galaxies[:, 1] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                         waveband)])

    pos_angles_neigh_galaxies = np.empty((len(neighbouring_source_galaxies_catalogue), 2))
    for i in range(len(neighbouring_source_galaxies_catalogue)):
        x_neigh, y_neigh = single_ra_dec_2_xy(neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[1]][i],
                                              neighbouring_source_galaxies_catalogue[neighb_galaxies_keys[2]][i],
                                              sci_image_filename)
        tofit = fitting_choice(x_neigh, y_neigh, image_size)
        if (neighbouring_source_galaxies_catalogue['{}_{}'.format(neighb_galaxies_keys[9],
                                                                  waveband)][i] == 1) & (tofit == 1):
            pos_angles_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue['{}_{}'
                                                                                      .format(neighb_galaxies_keys[8],
                                                                                              waveband)][i], 1]
        else:
            pos_angles_neigh_galaxies[i, :] = [neighbouring_source_galaxies_catalogue['{}_{}'
                                                                                      .format(neighb_galaxies_keys[8],
                                                                                              waveband)][i], 0]

    position_angles = np.vstack([pos_angles_target_galaxies, pos_angles_neigh_galaxies])

    return position_angles


def format_light_profiles_stamps(target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                 target_galaxies_keys, neighb_galaxies_keys, waveband):
    """

    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :return:
    """

    light_profiles_target_galaxies = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[-1],
                                                                                       waveband)], dtype='U20')
    light_profiles_neigh_galaxies = np.array(neighbouring_source_galaxies_catalogue['{}_{}'
                                             .format(neighb_galaxies_keys[-1], waveband)], dtype='U20')
    light_profiles = np.hstack([light_profiles_target_galaxies, light_profiles_neigh_galaxies])

    return light_profiles


def format_sersic_indices_stamps(target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                 target_galaxies_keys, neighb_galaxies_keys, waveband):
    """

    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :return:
    """

    sersic_indices_target_galaxies = np.empty((len(target_galaxies_catalogue), 2))
    sersic_indices_target_galaxies[:, 0] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[8],
                                                                                             waveband)])
    sersic_indices_target_galaxies[:, 1] = np.array(target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[12],
                                                                                             waveband)])

    sersic_indices_neigh_galaxies = np.empty((len(neighbouring_source_galaxies_catalogue), 2))
    sersic_indices_neigh_galaxies[:, 0] = np.array(neighbouring_source_galaxies_catalogue['{}_{}'
                                                   .format(neighb_galaxies_keys[5], waveband)])
    sersic_indices_neigh_galaxies[:, 1] = np.array(neighbouring_source_galaxies_catalogue['{}_{}'
                                                   .format(neighb_galaxies_keys[9], waveband)])
    sersic_indices = np.vstack([sersic_indices_target_galaxies, sersic_indices_neigh_galaxies])

    return sersic_indices


def format_properties_for_galfit_on_stamps(sci_image_filename, target_galaxies_catalogue,
                                           neighbouring_source_galaxies_catalogue, target_galaxies_keys,
                                           neighb_galaxies_keys, waveband, enlarging_image_factor):
    """

    :param sci_image_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_source_galaxies_catalogue:
    :param target_galaxies_keys:
    :param neighb_galaxies_keys:
    :param waveband:
    :param enlarging_image_factor:
    :return:
    """

    axis_ratio = target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[9], waveband)][0] / \
        target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[10], waveband)][0]
    angle = target_galaxies_catalogue['{}_{}'.format(target_galaxies_keys[11], waveband)][0] * (2 * np.pi) / 360

    image_size_x = target_galaxies_catalogue[target_galaxies_keys[7]][0] * enlarging_image_factor * \
        (abs(np.cos(angle)) + axis_ratio * abs(np.sin(angle)))
    image_size_y = target_galaxies_catalogue[target_galaxies_keys[7]][0] * enlarging_image_factor * \
        (abs(np.sin(angle)) + axis_ratio * abs(np.cos(angle)))
    image_size = [image_size_x, image_size_y]

    source_positions = format_positions_stamps(sci_image_filename, target_galaxies_catalogue,
                                               neighbouring_source_galaxies_catalogue, target_galaxies_keys,
                                               neighb_galaxies_keys, waveband, image_size)

    ra, dec = format_ra_dec_stamps(target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                   target_galaxies_keys, neighb_galaxies_keys)

    total_magnitudes = format_magnitudes_stamps(sci_image_filename, target_galaxies_catalogue,
                                                neighbouring_source_galaxies_catalogue, target_galaxies_keys,
                                                neighb_galaxies_keys, waveband, image_size)

    effective_radii = format_effective_radii_stamps(sci_image_filename, target_galaxies_catalogue,
                                                    neighbouring_source_galaxies_catalogue,
                                                    target_galaxies_keys, neighb_galaxies_keys, waveband, image_size)

    axis_ratios = format_axis_ratios_stamps(sci_image_filename, target_galaxies_catalogue,
                                            neighbouring_source_galaxies_catalogue, target_galaxies_keys,
                                            neighb_galaxies_keys, waveband, image_size)

    position_angles = format_position_angles_stamps(sci_image_filename, target_galaxies_catalogue,
                                                    neighbouring_source_galaxies_catalogue, target_galaxies_keys,
                                                    neighb_galaxies_keys, waveband, image_size)

    light_profiles = format_light_profiles_stamps(target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                                  target_galaxies_keys, neighb_galaxies_keys, waveband)

    sersic_indices = format_sersic_indices_stamps(target_galaxies_catalogue, neighbouring_source_galaxies_catalogue,
                                                  target_galaxies_keys, neighb_galaxies_keys, waveband)

    subtract = np.full(len(target_galaxies_catalogue) + len(neighbouring_source_galaxies_catalogue), '0')

    return light_profiles, source_positions, ra, dec, total_magnitudes, effective_radii, sersic_indices, axis_ratios, \
        position_angles, subtract
