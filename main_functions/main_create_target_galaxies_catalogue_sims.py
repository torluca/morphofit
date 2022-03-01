#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import numpy as np
from astropy.table import Table, join, Column
import glob
from astropy.io import fits
import argparse

# morphofit imports
from morphofit.utils import get_logger

logger = get_logger(__file__)


def match(x_in, y_in, mag_in, x_out, y_out, mag_out, size_x, size_y):

    matching_cells = 20
    max_radius = 2.5
    mag_diff = 3

    indices_link = np.full_like(x_out, -200, dtype=np.int)
    indices_in = np.arange(x_in.size)
    indices_out = np.arange(x_out.size)

    gridsize_x = size_x // matching_cells
    gridsize_y = size_y // matching_cells
    overlap = max_radius + 1

    max_radius = max_radius ** 2

    for i in range(0, size_x, gridsize_x):
        for j in range(0, size_y, gridsize_y):

            mask_out = (x_out >= i) & (x_out < i + gridsize_x) & (y_out >= j) & (y_out < j + gridsize_y)

            mask_in = (x_in > i - overlap) & (x_in < i + gridsize_x + overlap) & \
                      (y_in > j - overlap) & (y_in < j + gridsize_y + overlap)

            x_in_masked = x_in[mask_in]
            y_in_masked = y_in[mask_in]
            mag_in_masked = mag_in[mask_in]
            indices_in_masked = indices_in[mask_in]

            x_out_masked = x_out[mask_out]
            y_out_masked = y_out[mask_out]
            mag_out_masked = mag_out[mask_out]
            indices_out_masked = indices_out[mask_out]

            for k in range(x_out_masked.size):
                mask_mag = np.fabs(mag_in_masked - mag_out_masked[k]) < mag_diff
                dist = (x_in_masked[mask_mag] - x_out_masked[k]) ** 2 + (y_in_masked[mask_mag] - y_out_masked[k]) ** 2

                try:
                    min_dist_index = np.argmin(dist)

                    if dist[min_dist_index] < max_radius:
                        index_out = indices_out_masked[k]
                        indices_link[index_out] = indices_in_masked[mask_mag][min_dist_index]

                except ValueError:
                    pass

    return indices_link


def match_with_ucat(forced_catalogue, ucat_gal_cat, ucat_star_cat, size_x, size_y, x_ref, y_ref):
    # ucat_gal_cat = Table.read(ucat_gal_cat, format='fits', memmap=True)
    # ucat_star_cat = Table.read(ucat_star_cat, format='fits', memmap=True)
    x_in = np.array([], np.float)
    y_in = np.array([], np.float)
    mag_in = np.array([], np.float)

    x_in = np.append(x_in, ucat_gal_cat['x_f814w'])
    y_in = np.append(y_in, ucat_gal_cat['y_f814w'])
    mag_in = np.append(mag_in, ucat_gal_cat['mag_f814w'])

    x_in = np.append(x_in, ucat_star_cat['x_f814w'])
    y_in = np.append(y_in, ucat_star_cat['y_f814w'])
    mag_in = np.append(mag_in, ucat_star_cat['mag_f814w'])

    x_in += 0.5
    y_in += 0.5

    x_out = x_ref  # forced_catalogue['XWIN_IMAGE']
    y_out = y_ref  # forced_catalogue['YWIN_IMAGE']
    mag_out = forced_catalogue['MAG_AUTO_f814w']

    size_x = size_x
    size_y = size_y

    indices_link = match(x_in, y_in, mag_in, x_out, y_out, mag_out, size_x, size_y)

    galaxy_mask = (indices_link < len(ucat_gal_cat)) & (indices_link >= 0)
    star_mask = indices_link >= len(ucat_gal_cat)

    star_gal_sep_array = np.full_like(indices_link, 99.)
    star_gal_sep_array[galaxy_mask] = 0
    star_gal_sep_array[star_mask] = 1

    aa = Column(star_gal_sep_array, name='star/gal')
    forced_catalogue.add_column(aa)

    col_names = set(ucat_gal_cat.colnames) | set(ucat_star_cat.colnames)

    cat_len = len(forced_catalogue)
    galaxy_indices = indices_link[galaxy_mask]
    star_indices = indices_link[star_mask] - len(ucat_gal_cat)

    for col_name in col_names:
        if col_name in ucat_gal_cat.colnames:
            col_galaxies = ucat_gal_cat[col_name][galaxy_indices]
            col_shape = list(col_galaxies.shape)
            col_shape[0] = cat_len
            col_array = np.full(col_shape, 99., dtype=col_galaxies.dtype)
            col_array[galaxy_mask] = col_galaxies
            if col_name in ucat_star_cat.colnames:
                col_array[star_mask] = ucat_star_cat[col_name][star_indices]

        else:
            col_stars = ucat_star_cat[col_name][star_indices]
            col_shape = list(col_stars.shape)
            col_shape[0] = cat_len
            col_array = np.full(col_shape, 99., dtype=col_stars.dtype)
            col_array[star_mask] = col_stars

        aa = Column(col_array, name=col_name)
        forced_catalogue.add_column(aa)

    return forced_catalogue


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    wavebands = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
    targets = ['abells1063', 'macs1149']
    root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/sims/'

    for index in indices:

        logger.info('=============================== running on index={}'.format(index))

        # create master ucat gal cat

        ucat_gal_cats = glob.glob(root + '{}/*drz_crop.gal.cat'.format(targets[index]))
        ucat_gal_cats.sort()

        ucat_gal_cat = Table.read(ucat_gal_cats[0], format='fits')
        col_name_list = ucat_gal_cat.colnames
        for j in range(len(col_name_list)):
            if col_name_list[j] != 'id':
                ucat_gal_cat.rename_column(col_name_list[j], col_name_list[j] + '_%s' % wavebands[0])

        for i in range(1, len(wavebands)):
            try:
                ucat_gal_cat.rename_column('NUMBER_1', 'NUMBER')
            except KeyError:
                pass
            new_ucat_gal_cat = Table.read(ucat_gal_cats[i], format='fits')
            col_name_list = new_ucat_gal_cat.colnames
            for j in range(len(col_name_list)):
                if col_name_list[j] != 'id':
                    new_ucat_gal_cat.rename_column(col_name_list[j], col_name_list[j] + '_%s' % wavebands[i])
            ucat_gal_cat = join(ucat_gal_cat, new_ucat_gal_cat, keys='id')

        # create master ucat star cat

        ucat_star_cats = glob.glob(root + '{}/*drz_crop.star.cat'.format(targets[index]))
        ucat_star_cats.sort()
        ucat_star_cat = Table.read(ucat_star_cats[0], format='fits')
        col_name_list = ucat_star_cat.colnames
        for j in range(len(col_name_list)):
            if col_name_list[j] != 'id':
                ucat_star_cat.rename_column(col_name_list[j], col_name_list[j] + '_%s' % wavebands[0])

        for i in range(1, len(wavebands)):
            try:
                ucat_star_cat.rename_column('NUMBER_1', 'NUMBER')
            except KeyError:
                pass
            new_ucat_star_cat = Table.read(ucat_star_cats[i], format='fits')
            col_name_list = new_ucat_star_cat.colnames
            for j in range(len(col_name_list)):
                if col_name_list[j] != 'id':
                    new_ucat_star_cat.rename_column(col_name_list[j], col_name_list[j] + '_%s' % wavebands[i])
            ucat_star_cat = join(ucat_star_cat, new_ucat_star_cat, keys='id')

        # match master cat with master ucat gal cat

        mastercat = Table.read(root + '{}/HST_{}_multiband.forced.sexcat'.format(targets[index], targets[index]),
                               format='fits')
        if targets[index] == 'macs1149':
            header = fits.getheader(root +
                                    'macs1149/hlsp_frontier_hst_acs-30mas-selfcal_macs1149_f814w_v1.0-epoch2_drz_crop.sim.fits')
        else:
            header = fits.getheader(root +
                                    'abells1063/hlsp_frontier_hst_acs-30mas-selfcal_abells1063_f814w_v1.0-epoch1_drz_crop.sim.fits')
        size_x = header['NAXIS1']
        size_y = header['NAXIS2']
        x_ref = mastercat['XWIN_IMAGE_f814w']
        y_ref = mastercat['YWIN_IMAGE_f814w']

        matchedcat = match_with_ucat(mastercat, ucat_gal_cat, ucat_star_cat, size_x, size_y, x_ref, y_ref)

        matchedcat.write(root + '{}/HST_{}_matcheducat_multiband.forced.sexcat'.format(targets[index], targets[index]),
                         format='fits', overwrite=True)

        # selection of target galaxies with 0.3 <= z <= 0.5 and MAG_AUTO_f814w <= 22.5

        cluster_like_mask = np.where((matchedcat['MAG_AUTO_f814w'] <= 22.5) & (matchedcat['z_f814w'] != 99))
        targetcat = matchedcat[cluster_like_mask]

        targetcat.write(root + '{}/HST_{}_target_multiband.forced.sexcat'.format(targets[index], targets[index]),
                        format='fits', overwrite=True)


def setup(args):
    """

    :param args:
    :return:
    """

    description = "Create target galaxies catalogue for sims"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    args = parser.parse_args(args)
