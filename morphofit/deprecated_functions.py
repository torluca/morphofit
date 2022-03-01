#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
from astropy.table import Column, hstack, Table, unique, vstack
import numpy as np
from astropy.io import fits
import itertools
from scipy.ndimage import gaussian_filter
import os

# morphofit imports
from morphofit.catalogue_managing import find_matched_rows, delete_columns
from morphofit.catalogue_managing import match_galfit_table_with_zcat
from morphofit.utils import ra_dec_2_xy, single_ra_dec_2_xy
from morphofit.catalogue_managing import create_empty_property_arrays, append_properties, get_median_property_arrays
from morphofit.catalogue_managing import manage_crashed_galfit, delete_star_character, get_sersic_parameters_from_header
from morphofit.catalogue_managing import get_expdisk_parameters_from_header, get_background_parameters_from_header
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_position_for_galfit_single_sersic, format_ra_dec_for_galfit_single_sersic
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_magnitude_for_galfit_single_sersic, format_effective_radius_for_galfit_single_sersic
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_sersic_index_for_galfit_sersic_expdisk, format_axis_ratio_for_galfit_single_sersic
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import format_position_angle_for_galfit_single_sersic, format_sersic_index_for_galfit_single_sersic
from morphofit.deprecated_functions_scripts.galfit_utils_deprecated_211213 import copy_files_to_working_directory, remove_files_from_working_directory
from morphofit.utils import get_logger

logger = get_logger(__file__)


def create_params_tuple(params_names, wavebands):
    params = []
    for name in params_names:
        for band in wavebands:
            params.append('{}_{}'.format(name, band))
    params = tuple(params)
    return params


def create_galfit_params_fits_table(params_names, wavebands, x_positions, x_position_errors, y_positions,
                                    y_position_errors, ra, dec,
                                    total_magnitudes, total_magnitude_errors, effective_radii,
                                    effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios,
                                    axis_ratio_errors, position_angles, position_angle_errors, background_values,
                                    background_value_errors, background_x_gradients, background_x_gradient_errors,
                                    background_y_gradients, background_y_gradient_errors):
    """
    This function creates a table with best fitting GALFIT parameters.

    :return galfit_params_table: table
    """

    params_names_table = create_params_tuple(params_names, wavebands)
    galfit_params_table = Table()
    for band in wavebands:
        idx_band = wavebands.index(band)
        galfit_params_table['x_{}'.format(band)] = x_positions[:, idx_band]
        galfit_params_table['x_err_{}'.format(band)] = x_position_errors[:, idx_band]
        galfit_params_table['y_{}'.format(band)] = y_positions[:, idx_band]
        galfit_params_table['y_err_{}'.format(band)] = y_position_errors[:, idx_band]
        galfit_params_table['mag_{}'.format(band)] = total_magnitudes[:, idx_band]
        galfit_params_table['mag_err_{}'.format(band)] = total_magnitude_errors[:, idx_band]
        galfit_params_table['re_{}'.format(band)] = effective_radii[:, idx_band]
        galfit_params_table['re_err_{}'.format(band)] = effective_radius_errors[:, idx_band]
        galfit_params_table['n_{}'.format(band)] = sersic_indices[:, idx_band]
        galfit_params_table['n_err_{}'.format(band)] = sersic_index_errors[:, idx_band]
        galfit_params_table['ar_{}'.format(band)] = axis_ratios[:, idx_band]
        galfit_params_table['ar_err_{}'.format(band)] = axis_ratio_errors[:, idx_band]
        galfit_params_table['pa_{}'.format(band)] = position_angles[:, idx_band]
        galfit_params_table['pa_err_{}'.format(band)] = position_angle_errors[:, idx_band]
    galfit_params_table['ra'] = ra[:]
    galfit_params_table['dec'] = dec[:]
    galfit_params_table['bkg_value'] = background_values[:, 0]
    galfit_params_table['bkg_value_err'] = background_value_errors[:, 0]
    galfit_params_table['bkg_x_grad'] = background_x_gradients[:, 0]
    galfit_params_table['bkg_x_grad_err'] = background_x_gradient_errors[:, 0]
    galfit_params_table['bkg_y_grad'] = background_y_gradients[:, 0]
    galfit_params_table['bkg_y_grad_err'] = background_y_gradient_errors[:, 0]

    return galfit_params_table


def create_fixed_cluster_stamp_table(cluster_name, wavebands, cluster_sexcat, cluster_zcat,
                                     x, x_err, y, y_err, ra, dec, mag,
                                     mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err,
                                     sky_x_grad,
                                     sky_x_grad_err, sky_y_grad, sky_y_grad_err):
    """
    This function creates tables at fixed stamp.

    :return galfit_params_stamps_table: table
    """

    params_names = ['x', 'x_err', 'y', 'y_err', 'mag', 'mag_err', 're', 're_err', 'n', 'n_err', 'ar', 'ar_err', 'pa',
                    'pa_err', 'ra', 'dec', 'sky_value', 'sky_value_err', 'sky_x_grad', 'sky_x_grad_err', 'sky_y_grad',
                    'sky_y_grad_err']

    galfit_params_table = create_galfit_params_fits_table(params_names, wavebands, x, x_err, y, y_err, mag,
                                                          mag_err, re, re_err, n,
                                                          n_err, ar, ar_err, pa, pa_err, ra, dec,
                                                          sky_value, sky_value_err, sky_x_grad, sky_x_grad_err,
                                                          sky_y_grad, sky_y_grad_err)

    matched_cluster_sexcat = find_matched_rows(ra, dec, cluster_sexcat)
    matched_cluster_zcat = find_matched_rows(ra, dec, cluster_zcat)

    cols_to_delete = ['KRON_RADIUS', 'PETRO_RADIUS',
                      'BACKGROUND', 'XWIN_IMAGE', 'YWIN_IMAGE', 'ALPHAWIN_J2000', 'DELTAWIN_J2000', 'X2WIN_IMAGE',
                      'Y2WIN_IMAGE', 'XYWIN_IMAGE']
    matched_cluster_sexcat = delete_columns(matched_cluster_sexcat, cols_to_delete, wavebands)
    matched_cluster_zcat = delete_columns(matched_cluster_zcat, cols_to_delete, wavebands)

    galfit_params_table_matched_cluster_sexcat = hstack([galfit_params_table, matched_cluster_sexcat])

    try:
        galfit_params_stamps_table = match_galfit_table_with_zcat(galfit_params_table_matched_cluster_sexcat,
                                                                  matched_cluster_zcat)
        galfit_params_stamps_table.remove_column('d2d')
        col_names = galfit_params_stamps_table.colnames
        for col_name in col_names:
            if col_name[-2:] == '_2':
                galfit_params_stamps_table.remove_column(col_name)
            elif col_name[-2:] == '_1':
                galfit_params_stamps_table.rename_column(col_name, col_name[:-2])
            else:
                pass
    except Exception as e:
        print(e)
        galfit_params_stamps_table = galfit_params_table_matched_cluster_sexcat
        values = np.full(len(galfit_params_stamps_table), 99.)
        i_d = Column(values, name='ID')
        galfit_params_stamps_table.add_column(i_d)
        ra = Column(values, name='RA')
        galfit_params_stamps_table.add_column(ra)
        dec = Column(values, name='DEC')
        galfit_params_stamps_table.add_column(dec)
        z = Column(values, name='z')
        galfit_params_stamps_table.add_column(z)
        z_ref = Column(values, name='z_ref')
        galfit_params_stamps_table.add_column(z_ref)
        if cluster_name == 'abells1063':
            z_quality = Column(values, name='z_quality')
            galfit_params_stamps_table.add_column(z_quality)
            multiplicity = Column(values, name='multiplicity')
            galfit_params_stamps_table.add_column(multiplicity)
            sigma_z = Column(values, name='sigma_z')
            galfit_params_stamps_table.add_column(sigma_z)
            Kron_R = Column(values, name='Kron_R')
            galfit_params_stamps_table.add_column(Kron_R)
            root_name = Column(values, name='root_name')
            galfit_params_stamps_table.add_column(root_name)
        else:
            Kron_R = Column(values, name='Kron_f814w')
            galfit_params_stamps_table.add_column(Kron_R)
            err_Kron_R = Column(values, name='err_Kron_f814w')
            galfit_params_stamps_table.add_column(err_Kron_R)

    return galfit_params_stamps_table


def get_median_properties(cluster_name, wavebands, psf_types, background_estimate_methods,
                          sigma_image_types, n_galaxies, stamp_number, x_dict, y_dict, ra_dict, dec_dict, mag_dict,
                          re_dict, n_dict, ar_dict, pa_dict,
                          sky_value_dict, sky_x_grad_dict, sky_y_grad_dict):
    """
    This function computes the median of the best fitting properties from different combinations of PSF, sigma image
    and background estimate type.
    :param cluster_name:
    :param wavebands:
    :param psf_types:
    :param background_estimate_methods:
    :param sigma_image_types:
    :param n_galaxies:
    :param stamp_number:
    :param x_dict:
    :param y_dict:
    :param ra_dict:
    :param dec_dict:
    :param mag_dict:
    :param re_dict:
    :param n_dict:
    :param ar_dict:
    :param pa_dict:
    :param sky_value_dict:
    :param sky_x_grad_dict:
    :param sky_y_grad_dict:
    :return x,x_err,y,y_err,ra,dec,mag,mag_err,re,re_err,n,n_err,ar,ar_err,pa,pa_err,sky_value,sky_value_err,sky_x_grad,\
           sky_x_grad_err,sky_y_grad,sky_y_grad_err: median of the properties.
    """

    x, x_err, y, y_err, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err, sky_x_grad, \
    sky_x_grad_err, sky_y_grad, sky_y_grad_err = np.empty((n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands))), np.empty(
        (n_galaxies, len(wavebands))), \
                                                 np.empty((n_galaxies, len(wavebands)))
    ra = np.empty(n_galaxies)
    dec = np.empty(n_galaxies)

    for i in range(n_galaxies):
        for j in range(len(wavebands)):
            xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, svl, svel, sgxl, sgxel, sgyl, sgyel = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            for psf in psf_types:
                for back in background_estimate_methods:
                    for sigma in sigma_image_types:
                        xl.append(float(x_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        xel.append(float(x_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        yl.append(float(y_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        yel.append(float(y_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        ml.append(float(mag_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        mel.append(float(mag_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        rl.append(float(re_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        rel.append(float(re_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        nl.append(float(n_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        nel.append(float(n_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        al.append(float(ar_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        ael.append(float(ar_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        pl.append(float(pa_dict[
                                            '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, stamp_number)][0][i, 0]))
                        pel.append(float(pa_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][i, 1]))
                        svl.append(float(sky_value_dict[
                                             '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, stamp_number)][0][0]))
                        svel.append(float(sky_value_dict[
                                              '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, stamp_number)][0][1]))
                        sgxl.append(float(sky_x_grad_dict[
                                              '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, stamp_number)][0][0]))
                        sgxel.append(float(sky_x_grad_dict[
                                               '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf,
                                                                               back, sigma, stamp_number)][0][1]))
                        sgyl.append(float(sky_y_grad_dict[
                                              '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, stamp_number)][0][0]))
                        sgyel.append(float(sky_y_grad_dict[
                                               '{}_{}_{}_{}_{}_stamp{}'.format(cluster_name, wavebands[j], psf,
                                                                               back, sigma, stamp_number)][0][1]))
            x[i][j], x_err[i][j], y[i][j], y_err[i][j], mag[i][j], mag_err[i][j], re[i][j], re_err[i][j], n[i][j], \
            n_err[i][j], \
            ar[i][j], ar_err[i][j], pa[i][j], pa_err[i][j], sky_value[i][j], sky_value_err[i][j], sky_x_grad[i][j], \
            sky_x_grad_err[i][j], sky_y_grad[i][j], sky_y_grad_err[i][j] = np.nanmedian(xl), np.nanmedian(
                xel), np.nanmedian(yl), \
                                                                           np.nanmedian(yel), np.nanmedian(
                ml), np.nanmedian(mel), np.nanmedian(rl), np.nanmedian(rel), \
                                                                           np.nanmedian(nl), np.nanmedian(
                nel), np.nanmedian(al), np.nanmedian(ael), np.nanmedian(pl), \
                                                                           np.nanmedian(pel), np.nanmedian(
                svl), np.nanmedian(svel), np.nanmedian(sgxl), np.nanmedian(sgxel), \
                                                                           np.nanmedian(sgyl), np.nanmedian(sgyel)
        ra[i] = \
        ra_dict['{}_f814w_{}_{}_{}_stamp{}'.format(cluster_name, psf, back, sigma, stamp_number)][0][i]
        dec[i] = \
        dec_dict['{}_f814w_{}_{}_{}_stamp{}'.format(cluster_name, psf, back, sigma, stamp_number)][0][i]

    return x, x_err, y, y_err, ra, dec, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err, sky_x_grad, \
           sky_x_grad_err, sky_y_grad, sky_y_grad_err


def cut_muse_fov(root_input, image_name, seg_image_name, rms_image_name, root_output, sources_catalogue):
    data, head = fits.getdata(root_input + image_name, header=True)
    seg_data, seg_head = fits.getdata(root_input + seg_image_name, header=True)
    rms_data, rms_head = fits.getdata(root_input + rms_image_name, header=True)
    x = sources_catalogue['XWIN_IMAGE_f814w']
    y = sources_catalogue['YWIN_IMAGE_f814w']
    crpix1 = head['CRPIX1']
    crpix2 = head['CRPIX2']
    enlarge = 100  # pixels
    deltaxy = (int(round(max(x))) - int(round(min(x)))) - (int(round(max(y))) - int(round(min(y))))
    xmax = int(round(max(x))) + enlarge
    ymax = int(round(max(y))) + enlarge
    if deltaxy > 0:
        xmin = int(round(min(x))) - enlarge
        ymin = int(round(min(y))) - (enlarge + deltaxy)
        head['CRPIX1'] = crpix1 - (round(min(x)) - enlarge)
        head['CRPIX2'] = crpix2 - (round(min(y)) - (enlarge + deltaxy))
        seg_head['CRPIX1'] = crpix1 - (round(min(x)) - enlarge)
        seg_head['CRPIX2'] = crpix2 - (round(min(y)) - (enlarge + deltaxy))
        rms_head['CRPIX1'] = crpix1 - (round(min(x)) - enlarge)
        rms_head['CRPIX2'] = crpix2 - (round(min(y)) - (enlarge + deltaxy))
    else:
        xmin = int(round(min(x))) - (enlarge + deltaxy)
        ymin = int(round(min(y))) - enlarge
        head['CRPIX1'] = crpix1 - (round(min(x)) - (enlarge + deltaxy))
        head['CRPIX2'] = crpix2 - (round(min(y)) - enlarge)
        seg_head['CRPIX1'] = crpix1 - (round(min(x)) - (enlarge + deltaxy))
        seg_head['CRPIX2'] = crpix2 - (round(min(y)) - enlarge)
        rms_head['CRPIX1'] = crpix1 - (round(min(x)) - (enlarge + deltaxy))
        rms_head['CRPIX2'] = crpix2 - (round(min(y)) - enlarge)

    newdata = data[ymin:ymax, xmin:xmax]
    newsegdata = seg_data[ymin:ymax, xmin:xmax]
    newrmsdata = rms_data[ymin:ymax, xmin:xmax]
    muse_fov_image = '{}{}_muse_drz.fits'.format(root_output, image_name.split('_drz.fits')[0])
    muse_fov_seg_image = '{}{}_muse_drz_forced_seg.fits'.format(root_output,
                                                                seg_image_name.split('_drz_forced_seg.fits')[0])
    muse_fov_rms_image = '{}{}_muse_rms.fits'.format(root_output, rms_image_name.split('_rms.fits')[0])
    fits.writeto(muse_fov_image, newdata, head, overwrite=True)
    fits.writeto(muse_fov_seg_image, newsegdata, seg_head, overwrite=True)
    fits.writeto(muse_fov_rms_image, newrmsdata, rms_head, overwrite=True)

    return muse_fov_image, muse_fov_seg_image, muse_fov_rms_image


def cut_regions(root_input, image_name, seg_image_name, rms_image_name, root_output, N):
    data, head = fits.getdata(root_input + image_name, header=True)
    seg_data, seg_head = fits.getdata(root_input + seg_image_name, header=True)
    rms_data, rms_head = fits.getdata(root_input + rms_image_name, header=True)
    size = head['NAXIS1'] / N
    crpix1 = head['CRPIX1']
    crpix2 = head['CRPIX2']
    reg_image_filenames = []
    reg_seg_image_filenames = []
    reg_rms_image_filenames = []
    for i in range(0, N):
        for j in range(0, N):
            newdata = data[int(i*size):int(size*i+size), int(j*size):int(size*j+size)]
            newsegdata = seg_data[int(i*size):int(size*i+size), int(j*size):int(size*j+size)]
            newrmsdata = rms_data[int(i*size):int(size*i+size), int(j*size):int(size*j+size)]
            head['CRPIX1'] = crpix1 - int(j*size)
            head['CRPIX2'] = crpix2 - int(i*size)
            seg_head['CRPIX1'] = crpix1 - int(j*size)
            seg_head['CRPIX2'] = crpix2 - int(i*size)
            rms_head['CRPIX1'] = crpix1 - int(j*size)
            rms_head['CRPIX2'] = crpix2 - int(i*size)
            fits.writeto('{}{}_reg{}_drz.fits'.format(root_output, image_name.split('_drz.fits')[0], (i * N) + j),
                         newdata, head, overwrite=True)
            fits.writeto('{}{}_reg{}_drz_forced_seg.fits'.format(root_output,
                                                                   seg_image_name.split('_drz_forced_seg.fits')[0],
                                                                   (i * N) + j),
                         newsegdata, seg_head, overwrite=True)
            fits.writeto('{}{}_reg{}_rms.fits'.format(root_output, rms_image_name.split('_rms.fits')[0], (i * N) + j),
                         newrmsdata, rms_head, overwrite=True)
            reg_image_filenames.append('{}{}_reg{}_drz.fits'.format(root_output,
                                                                      image_name.split('_drz.fits')[0], (i * N) + j))
            reg_seg_image_filenames.append('{}{}_reg{}_drz_forced_seg.fits'.format(root_output,
                                                                                     seg_image_name.split('_drz_forced_seg.fits')[0],
                                                                                     (i * N) + j))
            reg_rms_image_filenames.append('{}{}_reg{}_rms.fits'.format(root_output, rms_image_name.split('_rms.fits')[0],
                                                                          (i * N) + j))

    return reg_image_filenames, reg_seg_image_filenames, reg_rms_image_filenames


def assign_sources_to_regions(reg_image_filenames, stamps_mastercatalogue, waveband_key):
    """

    :param reg_image_filenames:
    :param stamps_mastercatalogue:
    :param waveband_key:
    :return:
    """

    ra = stamps_mastercatalogue['ALPHAWIN_J2000_{}'.format(waveband_key)]
    dec = stamps_mastercatalogue['DELTAWIN_J2000_{}'.format(waveband_key)]
    region_cats = []
    for i in range(len(reg_image_filenames)):
        x, y = ra_dec_2_xy(ra, dec, reg_image_filenames[i])
        h = fits.getheader(reg_image_filenames[i])
        mask = np.where((x > 0) & (x < h['NAXIS1']) & (y > 0) & (y < h['NAXIS2']))
        region_cat = stamps_mastercatalogue[mask]
        region_cat.write(reg_image_filenames[i][:-4] + 'sexcat', format='fits', overwrite=True)
        region_cats.append(reg_image_filenames[i][:-4] + 'sexcat')

    return region_cats


def format_properties_for_regions_galfit(image_filename, source_catalogue_filename, band):
    """

    :param image_filename:
    :param source_catalogue_filename:
    :param band:
    :return:
    """

    source_catalogue = Table.read(source_catalogue_filename, format='fits')
    # size = fits.getheader(image_filename)['NAXIS1']
    profile = np.full(len(source_catalogue), 'sersic')
    position = np.empty((len(source_catalogue), 4))
    ra = np.empty(len(source_catalogue))
    dec = np.empty(len(source_catalogue))
    tot_magnitude = np.empty((len(source_catalogue), 2))
    eff_radius = np.empty((len(source_catalogue), 2))
    sersic_index = np.empty((len(source_catalogue), 2))
    axis_ratio = np.empty((len(source_catalogue), 2))
    pa_angle = np.empty((len(source_catalogue), 2))
    subtract = np.full(len(source_catalogue), '0')
    tofit = 1

    for i in range(len(source_catalogue)):
        ra[i] = source_catalogue['ALPHAWIN_J2000_f814w'][i]
        dec[i] = source_catalogue['DELTAWIN_J2000_f814w'][i]
        x, y = single_ra_dec_2_xy(ra[i], dec[i], image_filename)
        position[i, :] = [x, y, tofit, tofit]
        tot_magnitude[i, :] = [source_catalogue['mag_{}'.format(band)][i], tofit]
        eff_radius[i, :] = [source_catalogue['re_{}'.format(band)][i], tofit]
        sersic_index[i, :] = [source_catalogue['n_{}'.format(band)][i], tofit]
        axis_ratio[i, :] = [source_catalogue['ar_{}'.format(band)][i], tofit]
        pa_angle[i, :] = [source_catalogue['pa_{}'.format(band)][i], tofit]

    return profile, position, ra, dec, tot_magnitude, eff_radius, sersic_index, axis_ratio, pa_angle, subtract


def create_bad_pixel_region_mask(source_catalogue_filename, seg_image_filename):
    """

    :param source_catalogue_filename:
    :param seg_image_filename:
    :return:
    """

    seg_image, seg_head = fits.getdata(seg_image_filename, header=True)
    source_catalogue = Table.read(source_catalogue_filename, format='fits')
    for i in range(len(source_catalogue)):
        w = np.where(seg_image == source_catalogue['NUMBER'][i])
        seg_image[w] = 0
    bad_pixel_mask_name = seg_image_filename[:-5] + '_badpixel.fits'
    fits.writeto(bad_pixel_mask_name, seg_image, seg_head, overwrite=True)

    return bad_pixel_mask_name


def get_stamps_median_properties(target_name, wavebands, psf_image_types, background_estimate_methods,
                                 sigma_image_types, number_galaxies_in_stamp, stamp_number, x_dictionary, y_dictionary,
                                 ra_dictionary, dec_dictionary, mag_dictionary, re_dictionary, n_dictionary,
                                 ar_dictionary, pa_dictionary, background_value_dictionary,
                                 background_x_gradient_dictionary, background_y_gradient_dictionary,
                                 waveband_key='f814w'):
    """
    This function computes the median of the best fitting properties from different combinations of PSF, sigma image
    and background estimate type.

    :param target_name:
    :param wavebands:
    :param psf_image_types:
    :param background_estimate_methods:
    :param sigma_image_types:
    :param number_galaxies_in_stamp:
    :param stamp_number:
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
    :param waveband_key:
    :return : median of the properties.
    """

    x_positions, x_position_errors, y_positions, y_position_errors, total_magnitudes, total_magnitude_errors, \
        effective_radii, effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios, axis_ratio_errors, \
        position_angles, position_angle_errors, background_values, background_value_errors, background_x_gradients, \
        background_x_gradient_errors, background_y_gradients, background_y_gradient_errors = \
        create_empty_property_arrays(number_galaxies_in_stamp, len(wavebands))
    ra = np.empty(number_galaxies_in_stamp)
    dec = np.empty(number_galaxies_in_stamp)

    for n_gal_stamp_idx in range(number_galaxies_in_stamp):
        for wave_idx in range(len(wavebands)):
            xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg, \
                bkge, bkgx, bkgxe, bkgy, bkgye = [], [], [], [], [], [], [], [], [], [], \
                                                 [], [], [], [], [], [], [], [], [], []
            combinations = [['{}'.format(x), '{}'.format(y), '{}'.format(z)]
                            for x, y, z in itertools.product(psf_image_types, background_estimate_methods,
                                                             sigma_image_types)]
            for k in range(len(combinations)):
                try:
                    xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg, bkge, bkgx, bkgxe, \
                        bkgy, bkgye = append_properties(target_name, wavebands[wave_idx], combinations[k][0],
                                                        combinations[k][1], combinations[k][2], stamp_number,
                                                        n_gal_stamp_idx, x_dictionary, y_dictionary, mag_dictionary,
                                                        re_dictionary, n_dictionary, ar_dictionary,
                                                        pa_dictionary, background_value_dictionary,
                                                        background_x_gradient_dictionary,
                                                        background_y_gradient_dictionary, xl, xel, yl, yel, ml, mel,
                                                        rl, rel, nl, nel, al, ael, pl, pel, bkg, bkge, bkgx, bkgxe,
                                                        bkgy, bkgye)
                except Exception as e:
                    print(e)
                    continue

            x_positions[n_gal_stamp_idx][wave_idx], x_position_errors[n_gal_stamp_idx][wave_idx], \
                y_positions[n_gal_stamp_idx][wave_idx], y_position_errors[n_gal_stamp_idx][wave_idx], \
                total_magnitudes[n_gal_stamp_idx][wave_idx], total_magnitude_errors[n_gal_stamp_idx][wave_idx], \
                effective_radii[n_gal_stamp_idx][wave_idx], effective_radius_errors[n_gal_stamp_idx][wave_idx], \
                sersic_indices[n_gal_stamp_idx][wave_idx], sersic_index_errors[n_gal_stamp_idx][wave_idx], \
                axis_ratios[n_gal_stamp_idx][wave_idx], axis_ratio_errors[n_gal_stamp_idx][wave_idx], \
                position_angles[n_gal_stamp_idx][wave_idx], position_angle_errors[n_gal_stamp_idx][wave_idx], \
                background_values[n_gal_stamp_idx][wave_idx], background_value_errors[n_gal_stamp_idx][wave_idx], \
                background_x_gradients[n_gal_stamp_idx][wave_idx], \
                background_x_gradient_errors[n_gal_stamp_idx][wave_idx], \
                background_y_gradients[n_gal_stamp_idx][wave_idx], \
                background_y_gradient_errors[n_gal_stamp_idx][wave_idx] = \
                get_median_property_arrays(xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, bkg,
                                           bkge, bkgx, bkgxe, bkgy, bkgye)

        try:
            ra[n_gal_stamp_idx] = ra_dictionary['{}_{}_{}_{}_{}_{}'
                                                .format(target_name, waveband_key, stamp_number,
                                                        psf_image_types[0], sigma_image_types[0],
                                                        background_estimate_methods[0])][n_gal_stamp_idx]
            dec[n_gal_stamp_idx] = dec_dictionary['{}_{}_{}_{}_{}_{}'
                                                  .format(target_name, waveband_key, stamp_number, psf_image_types[0],
                                                          sigma_image_types[0],
                                                          background_estimate_methods[0])][n_gal_stamp_idx]
        except Exception as e:
            print(e)
            pass

    return x_positions, x_position_errors, y_positions, y_position_errors, ra, dec, \
        total_magnitudes, total_magnitude_errors, effective_radii, \
        effective_radius_errors, sersic_indices, sersic_index_errors, axis_ratios, \
        axis_ratio_errors, position_angles, position_angle_errors, background_values, \
        background_value_errors, background_x_gradients, background_x_gradient_errors, \
        background_y_gradients, background_y_gradient_errors


def get_median_properties_region(cluster_name, wavebands, psf_types, background_estimate_methods,
                                 sigma_image_types, n_galaxies, reg_number, x_dict, y_dict, ra_dict,
                                 dec_dict, mag_dict,
                                 re_dict, n_dict, ar_dict, pa_dict,
                                 sky_value_dict, sky_x_grad_dict, sky_y_grad_dict):
    """

    :param cluster_name:
    :param wavebands:
    :param psf_types:
    :param background_estimate_methods:
    :param sigma_image_types:
    :param n_galaxies:
    :param reg_number:
    :param x_dict:
    :param y_dict:
    :param ra_dict:
    :param dec_dict:
    :param mag_dict:
    :param re_dict:
    :param n_dict:
    :param ar_dict:
    :param pa_dict:
    :param sky_value_dict:
    :param sky_x_grad_dict:
    :param sky_y_grad_dict:
    :return:
    """

    x, x_err, y, y_err, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, sky_value_err, \
        sky_x_grad, sky_x_grad_err, sky_y_grad, sky_y_grad_err = \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands))), \
        np.empty((n_galaxies, len(wavebands))), np.empty((n_galaxies, len(wavebands)))
    ra = np.empty(n_galaxies)
    dec = np.empty(n_galaxies)

    for i in range(n_galaxies):
        for j in range(len(wavebands)):
            xl, xel, yl, yel, ml, mel, rl, rel, nl, nel, al, ael, pl, pel, svl, svel, sgxl, sgxel, sgyl, sgyel =\
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            for psf in psf_types:
                for back in background_estimate_methods:
                    for sigma in sigma_image_types:
                        xl.append(float(x_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, reg_number)][0][i, 0]))
                        xel.append(float(x_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, reg_number)][0][i, 1]))
                        yl.append(float(y_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, reg_number)][0][i, 0]))
                        yel.append(float(y_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, reg_number)][0][i, 1]))
                        ml.append(float(mag_dict[
                                            '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                          sigma, reg_number)][0][i, 0]))
                        mel.append(float(mag_dict[
                                             '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                           sigma, reg_number)][0][i, 1]))
                        rl.append(float(re_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, reg_number)][0][i, 0]))
                        rel.append(float(re_dict[
                                             '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                           sigma, reg_number)][0][i, 1]))
                        nl.append(float(n_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, reg_number)][0][i, 0]))
                        nel.append(float(n_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, reg_number)][0][i, 1]))
                        al.append(float(ar_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, reg_number)][0][i, 0]))
                        ael.append(float(ar_dict[
                                             '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                           sigma, reg_number)][0][i, 1]))
                        pl.append(float(pa_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                              sigma, reg_number)][0][i, 0]))
                        pel.append(float(pa_dict[
                                             '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                           sigma, reg_number)][0][i, 1]))
                        svl.append(float(sky_value_dict[
                                             '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                           sigma, reg_number)][0][0]))
                        svel.append(float(sky_value_dict[
                                              '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, reg_number)][0][1]))
                        sgxl.append(float(sky_x_grad_dict[
                                              '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, reg_number)][0][0]))
                        sgxel.append(float(sky_x_grad_dict[
                                               '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, reg_number)][0][1]))
                        sgyl.append(float(sky_y_grad_dict[
                                              '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                            sigma, reg_number)][0][0]))
                        sgyel.append(float(sky_y_grad_dict[
                                               '{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back,
                                                                             sigma, reg_number)][1]))
            x[i][j], x_err[i][j], y[i][j], y_err[i][j], mag[i][j], mag_err[i][j], re[i][j], re_err[i][j], n[i][j], \
                n_err[i][j], ar[i][j], ar_err[i][j], pa[i][j], pa_err[i][j], sky_value[i][j], sky_value_err[i][j],\
                sky_x_grad[i][j], sky_x_grad_err[i][j], sky_y_grad[i][j], sky_y_grad_err[i][j] = \
                np.nanmedian(xl), np.nanmedian(xel), np.nanmedian(yl), np.nanmedian(yel), np.nanmedian(ml),\
                np.nanmedian(mel), np.nanmedian(rl), np.nanmedian(rel), np.nanmedian(nl), np.nanmedian(nel),\
                np.nanmedian(al), np.nanmedian(ael), np.nanmedian(pl), np.nanmedian(pel), np.nanmedian(svl),\
                np.nanmedian(svel), np.nanmedian(sgxl), np.nanmedian(sgxel), np.nanmedian(sgyl), np.nanmedian(sgyel)
        ra[i] = ra_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back, sigma, reg_number)][0][i]
        dec[i] = dec_dict['{}_{}_{}_{}_{}_reg{}'.format(cluster_name, wavebands[j], psf, back, sigma, reg_number)][0][i]

    return x, x_err, y, y_err, ra, dec, mag, mag_err, re, re_err, n, n_err, ar, ar_err, pa, pa_err, sky_value, \
        sky_value_err, sky_x_grad, sky_x_grad_err, sky_y_grad, sky_y_grad_err


def create_cutout_header(header, x_source, y_source, effective_radius_source, enlarging_factor, axis_ratio, angle):
    """

    :param header:
    :param x_source:
    :param y_source:
    :param effective_radius_source:
    :param enlarging_factor:
    :param axis_ratio:
    :param angle:
    :return
    """

    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']
    size_cutout_x = effective_radius_source * enlarging_factor * (abs(np.cos(angle)) + axis_ratio * abs(np.sin(angle)))
    size_cutout_y = effective_radius_source * enlarging_factor * (abs(np.sin(angle)) + axis_ratio * abs(np.cos(angle)))
    cutout_header = header.copy()
    cutout_header['NAXIS1'] = size_cutout_x
    cutout_header['NAXIS2'] = size_cutout_y
    cutout_header['CRPIX1'] = round(crpix1 - x_source + size_cutout_x / 2)
    cutout_header['CRPIX2'] = round(crpix2 - y_source + size_cutout_y / 2)

    return cutout_header


def delete_repeating_sources(table, wavebands):
    """
    In order to delete the repeating sources due to the stamps procedure, we do the following steps:
    1) we select the unique sources according to the SExtractor ALPHAWIN_J2000_f814w and DELTAWIN_J2000_f814w
    2) we select the repeating sources using Pandas and we find the unique ids given by NUMBER for each source.
    3) we create a new empty table called new_table
    4)

    :param table:
    :param wavebands:
    :return:
    """

    table_orig = unique(table, keys=['ra_f814w', 'dec_f814w'], keep='none')
    df = table.to_pandas()
    mask = (df.duplicated(['ra_f814w', 'dec_f814w'], keep=False))
    table_repeat = Table.from_pandas(df[mask])
    unique_idxs = np.unique(table_repeat['NUMBER'])
    new_table = table_orig.copy()
    new_table.remove_rows(slice(0, -1))
    new_table.remove_row(0)
    for i in range(len(unique_idxs)):
        w = np.where(table_repeat['NUMBER'] == unique_idxs[i])
        col_names = table_repeat[w].colnames[14 * len(wavebands):]
        params_values = []
        for j in range(len(wavebands)):
            x_err = table_repeat[w]['x_err_{}'.format(wavebands[j])]
            y_err = table_repeat[w]['y_err_{}'.format(wavebands[j])]
            mag_err = table_repeat[w]['mag_err_{}'.format(wavebands[j])]
            re_err = table_repeat[w]['re_err_{}'.format(wavebands[j])]
            n_err = table_repeat[w]['n_err_{}'.format(wavebands[j])]
            ar_err = table_repeat[w]['ar_err_{}'.format(wavebands[j])]
            pa_err = table_repeat[w]['pa_err_{}'.format(wavebands[j])]
            idx_min_x = np.argmin(x_err)
            idx_min_y = np.argmin(y_err)
            idx_min_mag = np.argmin(mag_err)
            idx_min_re = np.argmin(re_err)
            idx_min_n = np.argmin(n_err)
            idx_min_ar = np.argmin(ar_err)
            idx_min_pa = np.argmin(pa_err)

            params_values.append(table_repeat[w]['x_{}'.format(wavebands[j])][idx_min_x])
            params_values.append(table_repeat[w]['x_err_{}'.format(wavebands[j])][idx_min_x])
            params_values.append(table_repeat[w]['y_{}'.format(wavebands[j])][idx_min_y])
            params_values.append(table_repeat[w]['y_err_{}'.format(wavebands[j])][idx_min_y])
            params_values.append(table_repeat[w]['mag_{}'.format(wavebands[j])][idx_min_mag])
            params_values.append(table_repeat[w]['mag_err_{}'.format(wavebands[j])][idx_min_mag])
            params_values.append(table_repeat[w]['re_{}'.format(wavebands[j])][idx_min_re])
            params_values.append(table_repeat[w]['re_err_{}'.format(wavebands[j])][idx_min_re])
            params_values.append(table_repeat[w]['n_{}'.format(wavebands[j])][idx_min_n])
            params_values.append(table_repeat[w]['n_err_{}'.format(wavebands[j])][idx_min_n])
            params_values.append(table_repeat[w]['ar_{}'.format(wavebands[j])][idx_min_ar])
            params_values.append(table_repeat[w]['ar_err_{}'.format(wavebands[j])][idx_min_ar])
            params_values.append(table_repeat[w]['pa_{}'.format(wavebands[j])][idx_min_pa])
            params_values.append(table_repeat[w]['pa_err_{}'.format(wavebands[j])][idx_min_pa])

        for name in col_names:
            params_values.append(table_repeat[w][name][0])

        new_table.add_row(params_values)

    final_table = vstack([table_orig, new_table], join_type='exact')

    return final_table


def check_parameters_for_next_fitting_deprecated(source_catalogue, waveband, magnitude_keyword='MAG_AUTO',
                                                 size_keyword='FLUX_RADIUS',
                                                 axis_keywords=None,
                                                 position_angle_keyword='THETAWIN_SKY',
                                                 magnitude_error_limit=0.1, magnitude_upper_limit=30,
                                                 max_magnitude_distance=0.5, size_error_limit=1, size_upper_limit=200,
                                                 max_size_distance=2, sersic_index_error_limit=1,
                                                 sersic_index_upper_limit=8, sersic_index_lower_limit=0.3,
                                                 axis_ratio_error_limit=0.1, axis_ratio_lower_limit=0.02,
                                                 max_ar_distance=0.1,
                                                 position_angle_error_limit=20):
    """
    This function checks the parameters from the previous iteration.
    For magnitudes, it flags as bad measurements those that have mag_err > magnitude_error_limit or
    mag > magnitude_upper_limit or mag = nan or mag <=0 or |mag - mag_reference| > reference_value.
    For sizes, it flags as bad measurements those that have re > size_error_limit or re > size_upper_limit or
    or re = nan or re <=0 or |re - re_reference| > reference_value.
    For sersic indices, it flags as bad measurements those that have n > sersic_error_limit or
    n > sersic_upper_limit or n < sersic_lower_limit or n = nan or n<=0.
    For axis ratio, it flags as bad measurements those that have ar > axisratio_error_limit or
    ar < axisratio_lower_limit or ar = nan or ar <=0 or |ar - ar_reference| > reference_value.
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
    :param max_magnitude_distance:
    :param size_error_limit:
    :param size_upper_limit:
    :param max_size_distance:
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
                       (source_catalogue['mag_galfit_{}'.format(waveband)] <= 0) |
                       (abs(source_catalogue['mag_galfit_{}'.format(waveband)] -
                            source_catalogue['{}_{}'.format(magnitude_keyword, waveband)]) > max_magnitude_distance))
    source_catalogue['mag_galfit_{}'.format(waveband)][bad_mag] = source_catalogue['{}_{}'.format(magnitude_keyword,
                                                                                                  waveband)][bad_mag]
    source_catalogue['mag_galfit_err_{}'.format(waveband)][bad_mag] = 0.

    bad_size = np.where((source_catalogue['re_galfit_err_{}'.format(waveband)] > size_error_limit) |
                        (source_catalogue['re_galfit_{}'.format(waveband)] > size_upper_limit) |
                        (np.isnan(source_catalogue['re_galfit_{}'.format(waveband)])) |
                        (source_catalogue['re_galfit_{}'.format(waveband)] <= 0) |
                        (abs(source_catalogue['re_galfit_{}'.format(waveband)] -
                             source_catalogue['{}_{}'.format(size_keyword, waveband)]) > max_size_distance))
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

    if axis_keywords is None:
        axis_keywords = ['BWIN_IMAGE', 'AWIN_IMAGE']
        axis_ratio_reference = (source_catalogue['{}_{}'.format(axis_keywords[0], waveband)] /
                                source_catalogue['{}_{}'.format(axis_keywords[1], waveband)]) ** 2
    else:
        e1 = (source_catalogue['X2WIN_IMAGE_{}'.format(waveband)] -
              source_catalogue['Y2WIN_IMAGE_{}'.format(waveband)]) / \
             (source_catalogue['X2WIN_IMAGE_{}'.format(waveband)] +
              source_catalogue['Y2WIN_IMAGE_{}'.format(waveband)])
        e2 = (2 * source_catalogue['XYWIN_IMAGE_{}'.format(waveband)]) / \
             (source_catalogue['X2WIN_IMAGE_{}'.format(waveband)] +
              source_catalogue['Y2WIN_IMAGE_{}'.format(waveband)])
        axis_ratio_reference = (1 - np.sqrt(e1 ** 2 + e2 ** 2)) ** 2

    bad_axis_ratio = np.where((source_catalogue['ar_galfit_err_{}'.format(waveband)] > axis_ratio_error_limit) |
                              (source_catalogue['ar_galfit_{}'.format(waveband)] < axis_ratio_lower_limit)
                              (np.isnan(source_catalogue['ar_galfit_{}'.format(waveband)])) |
                              (source_catalogue['ar_galfit_{}'.format(waveband)] <= 0)|
                              (abs(source_catalogue['ar_galfit_{}'.format(waveband)] -
                                   axis_ratio_reference) > max_ar_distance) )
    source_catalogue['ar_galfit_{}'.format(waveband)][bad_axis_ratio] = axis_ratio_reference[bad_axis_ratio]

    bad_position_angle = np.where((source_catalogue['pa_galfit_err_{}'.format(waveband)] > position_angle_error_limit) |
                                  (np.isnan(source_catalogue['pa_galfit_{}'.format(waveband)])) |
                                  (source_catalogue['pa_galfit_{}'.format(waveband)] == 0))
    source_catalogue['pa_galfit_{}'.format(waveband)][bad_position_angle] = \
        source_catalogue['{}_{}'.format(position_angle_keyword, waveband)][bad_position_angle]

    # bad_fit = (source_catalogue['mag_err_{}'.format(band)] > 0.1) | \
    #             (source_catalogue['re_err_{}'.format(band)] > 1.) | \
    #             (source_catalogue['re_{}'.format(band)] > 100) | \
    #             (source_catalogue['FLUX_RADIUS_{}'.format(band)] > 30) | \
    #             (source_catalogue['FLUX_RADIUS_{}'.format(band)] < 0)
    # x = np.array(source_catalogue['{}_{}'.format(magnitude_keyword, band)][~bad_fit])
    # y = np.array(source_catalogue['mag_{}'.format(band)][~bad_fit])
    # x = sm.add_constant(x)
    # model = sm.OLS(y, x)
    # results = model.fit()
    # intcpt, slope = results.params
    # err_intcpt, err_slope = np.sqrt(np.diag(results.cov_params()))
    # upper = slope * np.array(source_catalogue['{}_{}'.format(magnitude_keyword,
    # band)][~bad_fit]) + (intcpt + 2 * err_intcpt)
    # lower = slope * np.array(source_catalogue['{}_{}'.format(magnitude_keyword,
    # band)][~bad_fit]) + (intcpt - 2 * err_intcpt)
    # bad_from_mag = np.where((y < lower) | (y > upper))
    # source_catalogue['mag_{}'.format(band)][bad_from_mag] = source_catalogue['{}_{}'.format(magnitude_keyword,
    # band)][
    #     bad_from_mag]
    # x = np.array(source_catalogue['FLUX_RADIUS_{}'.format(band)][~bad_fit])
    # y = np.array(source_catalogue['re_{}'.format(band)][~bad_fit])
    # x = sm.add_constant(x)
    # model = sm.OLS(y, x)
    # results = model.fit()
    # intcpt, slope = results.params
    # err_intcpt, err_slope = np.sqrt(np.diag(results.cov_params()))
    # upper = slope * np.array(source_catalogue['FLUX_RADIUS_{}'.format(band)][~bad_fit]) +
    # (intcpt + 4 * err_intcpt)
    # lower = slope * np.array(source_catalogue['FLUX_RADIUS_{}'.format(band)][~bad_fit]) +
    # (intcpt - 4 * err_intcpt)
    # bad_from_size = np.where((y < lower) | (y > upper))
    # source_catalogue['re_{}'.format(band)][bad_from_size] = source_catalogue['FLUX_RADIUS_{}'.format(band)][
    #     bad_from_size]

    return source_catalogue


def delete_repeating_sources(table, wavebands):
    """
    In order to delete the repeating sources due to the stamps procedure, we do the following steps:
    1) we select the unique sources according to the SExtractor ALPHAWIN_J2000_f814w and DELTAWIN_J2000_f814w
    2) we select the repeating sources using Pandas and we find the unique ids given by NUMBER for each source.
    3) we create a new empty table called new_table
    4) of the different repeating sources we select the one that has the lowest distance in magnitude with respect to
    SExtractor
    5) add this new column to the new_table and vstack it with table_orig

    :param table:
    :param wavebands:
    :return:
    """

    w = (table['FLUX_RADIUS_{}'.format(wavebands[0])] > 0) & (table['MAG_AUTO_{}'.format(wavebands[0])] != 99)
    for wave in wavebands[1:]:
        w *= (table['FLUX_RADIUS_{}'.format(wave)] > 0) & (table['MAG_AUTO_{}'.format(wave)] != 99)
    table = table[w]

    table_orig = unique(table, keys=['ALPHAWIN_J2000_f814w', 'DELTAWIN_J2000_f814w'], keep='none')
    df = table.to_pandas()
    mask = (df.duplicated(['ALPHAWIN_J2000_f814w', 'DELTAWIN_J2000_f814w'], keep=False))
    table_repeat = Table.from_pandas(df[mask])
    unique_idxs = np.unique(table_repeat['NUMBER'])
    new_table = table_orig.copy()
    new_table.remove_rows(slice(0, -1))
    new_table.remove_row(0)
    for i in range(len(unique_idxs)):
        w = np.where(table_repeat['NUMBER'] == unique_idxs[i])
        galfit_col_names = [name for name in table_repeat[w].colnames if 'galfit' in name]
        col_names = [name for name in table_repeat[w].colnames if 'galfit' not in name]
        params_values = []

        for j in range(len(wavebands)):
            mag_dist = abs(table_repeat[w]['mag_galfit_{}'.format(wavebands[j])] - table_repeat[w][
                'MAG_AUTO_{}'.format(wavebands[j])])
            idx_min = np.argmin(mag_dist)
            for col in galfit_col_names:
                if wavebands[j] in col:
                    params_values.append(table_repeat[w][col][idx_min])

        for col in galfit_col_names:
            if 'bkg' in col:
                params_values.append(table_repeat[w][col][0])

        for col in col_names:
            params_values.append(table_repeat[w][col][0])

        new_table.add_row(params_values)

    final_table = vstack([table_orig, new_table], join_type='exact')

    return final_table


def create_custom_sigma_image(telescope_name, sigma_image_filename, sci_image_filename, rms_image_filename,
                              exp_image_filename, background_value, exposure_time):
    """
    Sigma image used up to 2020/12/09 results

    :param telescope_name:
    :param sigma_image_filename:
    :param sci_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param background_value:
    :param exposure_time:
    :return:
    """

    sci_image, sci_header = fits.getdata(sci_image_filename, header=True)
    sci_image_original = sci_image.copy()
    rms_image = fits.getdata(rms_image_filename)
    mask_negative_values_sci_image = np.where(sci_image < 0)
    sci_image[mask_negative_values_sci_image] = 0
    # mask_low_values_rms_image = np.where(rms_image < 10**5)
    # mask_high_values_rms_image = np.where(rms_image > 10 ** 5)
    # rms_image[mask_high_values_rms_image] = np.median(rms_image[mask_low_values_rms_image])
    if exp_image_filename:
        exp_image = fits.getdata(exp_image_filename)
        sigma_image = np.sqrt(rms_image**2 + (sci_image - background_value) / exp_image)
    else:
        sigma_image = np.sqrt(rms_image ** 2 + (sci_image - background_value) / exposure_time)
    if telescope_name == 'HST':
        sci_header['EXPTIME'] = 1.
        sci_header['TEXPTIME'] = 1.
        sci_header['CCDGAIN'] = np.mean([sci_header['ATODGNA'], sci_header['ATODGNB'], sci_header['ATODGNC'],
                                         sci_header['ATODGND']])
        sci_header['RDNOISE'] = np.mean([sci_header['READNSEA'], sci_header['READNSEB'], sci_header['READNSEC'],
                                         sci_header['READNSED']])
    else:
        sci_header['EXPTIME'] = 1.
    smoothed_sigma_image = gaussian_filter(sigma_image, sigma=1.2)
    mask_high_values_rms_image = np.where(rms_image > 10 ** 5)
    smoothed_sigma_image[mask_high_values_rms_image] = np.nan
    fits.writeto(sigma_image_filename, smoothed_sigma_image, sci_header, overwrite=True)
    fits.writeto(sci_image_filename, sci_image_original, sci_header, overwrite=True)


def create_constraints_file_for_galfit(constraints_file_path, n_galaxies):
    """
    Constraints file used up to 2020/12/09 results.

    :param constraints_file_path:
    :param n_galaxies:
    :return:
    """

    if constraints_file_path is not None:
        with open(constraints_file_path, 'w') as f:
            f.write('# Component/    parameter   constraint	Comment \n'
                    '# operation	(see below)   range \n\n')
            for i in range(n_galaxies):
                f.write('{} re 0.2 to 1000 \n'
                        '{} n 0.2 to 10 \n'
                        '{} q 0.02 to 1 \n'.format(i+1, i+1, i+1))
    else:
        pass


def create_internal_generated_sigma_image(telescope_name, sci_image_filename, exp_image_filename,
                                          exposure_time, instrumental_gain,
                                          magnitude_zeropoint, background_value):
    """

    :param telescope_name:
    :param sci_image_filename:
    :param exp_image_filename:
    :param exposure_time:
    :param instrumental_gain:
    :param magnitude_zeropoint
    :param background_value:
    :return:
    """

    sci_image, sci_header = fits.getdata(sci_image_filename, header=True)
    if exp_image_filename is not None:
        exposure_time_image = fits.getdata(exp_image_filename)
        sci_image_adu = (sci_image / instrumental_gain) * exposure_time_image
    else:
        sci_image_adu = (sci_image / instrumental_gain) * exposure_time
    if telescope_name == 'HST':
        sci_header['BUNIT'] = 'ADU'
        # sci_header['NCOMBINE'] = 1 # set to 1 only if NCOMBINE is already factorized in GAIN
    fits.writeto(sci_image_filename, sci_image_adu, sci_header, overwrite=True)
    magnitude_zeropoint = magnitude_zeropoint - 2.5 * np.log10(instrumental_gain)
    # magnitude_zeropoint = magnitude_zeropoint  + 2.5 * np.log10(instrumental_gain) - 2.5 * np.log10(exposure_time)
    background_value = (background_value / instrumental_gain) * exposure_time

    return magnitude_zeropoint, background_value


def get_bulge_disk_best_fit_parameters_from_model_image_deprecated(output_model_image_filename, n_fitted_components,
                                                                   light_profiles):
    """
    This function reads the GALFIT best fit parameters from the imgblock header for a bulge+disk fit.

    :param output_model_image_filename:
    :param n_fitted_components:
    :param light_profiles: same size and ordering of n_fitted_components
    :return best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, \
            best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles,\
            best_fit_background_value, best_fit_background_x_gradient, best_fit_background_y_gradient, \
            reduced_chisquare: best fit parameters.
    """

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
            if light_profiles[i] == 'sersic':
                best_fit_source_x_positions[i, :], best_fit_source_y_positions[i, :], best_fit_total_magnitudes[i, :], \
                    best_fit_effective_radii[i, :], best_fit_sersic_indices[i, :], best_fit_axis_ratios[i, :], \
                    best_fit_position_angles[i, :] = \
                    get_sersic_parameters_from_header(output_model_image_header, i)
            elif light_profiles[i] == 'expdisk':
                best_fit_source_x_positions[i, :], best_fit_source_y_positions[i, :], best_fit_total_magnitudes[i, :], \
                    best_fit_effective_radii[i, :], best_fit_sersic_indices[i, :], best_fit_axis_ratios[i, :], \
                    best_fit_position_angles[i, :] = \
                    get_expdisk_parameters_from_header(output_model_image_header, i)
            else:
                logger.info('not implemented')
                raise ValueError

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


def get_single_sersic_best_fit_parameters_from_model_image_deprecated(output_model_image_filename, n_fitted_components):
    """
    This function reads the GALFIT best fit parameters from the imgblock header.

    :param output_model_image_filename:
    :param n_fitted_components:
    :return best_fit_source_x_positions, best_fit_source_y_positions, best_fit_total_magnitudes, \
            best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios, best_fit_position_angles,\
            best_fit_background_value, best_fit_background_x_gradient, best_fit_background_y_gradient, \
            reduced_chisquare: best fit parameters.
    """

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
            best_fit_source_x_positions[i, :], best_fit_source_y_positions[i, :], best_fit_total_magnitudes[i, :], \
                best_fit_effective_radii[i, :], best_fit_sersic_indices[i, :], best_fit_axis_ratios[i, :], \
                best_fit_position_angles[i, :] = \
                get_sersic_parameters_from_header(output_model_image_header, i)

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


def format_properties_for_stamps_galfit_bulge_disk_deprecated(sci_image_stamp_filename, waveband, ra_source,
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
                                                              enlarging_image_factor):
    """
    For the bulge+disk decomposition, we assume that the bulge and disk component have the same initial centroids,
    magnitudes and half-light/scale radii from SExtractor MAG_AUTO and FLUX_RADIUS. Same applies for axis ratios and
    position angles. The initial Sersic index for the bulge component is set to 4.

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
    :return:
    """

    axis_ratio = minor_axis_source / major_axis_source
    angle = position_angle_source * (2 * np.pi) / 360

    image_size_x = effective_radius_source * enlarging_image_factor * (abs(np.cos(angle)) + axis_ratio *
                                                                       abs(np.sin(angle)))
    image_size_y = effective_radius_source * enlarging_image_factor * (abs(np.sin(angle)) + axis_ratio *
                                                                       abs(np.cos(angle)))
    image_size = [image_size_x, image_size_y]

    light_profiles_bulge = np.full(len(neighbouring_sources_catalogue) + 1, 'sersic')
    light_profiles_disk = np.full(len(neighbouring_sources_catalogue) + 1, 'expdisk')
    light_profiles = np.hstack((light_profiles_bulge, light_profiles_disk))

    source_positions = format_position_for_galfit_single_sersic(sci_image_stamp_filename, ra_source, dec_source,
                                                                neighbouring_sources_catalogue,
                                                                ra_key_neighbouring_sources_catalogue,
                                                                dec_key_neighbouring_sources_catalogue,
                                                                image_size)
    source_positions = np.vstack((source_positions, source_positions))
    ra, dec = format_ra_dec_for_galfit_single_sersic(ra_source, dec_source, neighbouring_sources_catalogue,
                                                     ra_key_neighbouring_sources_catalogue,
                                                     dec_key_neighbouring_sources_catalogue)
    ra = np.hstack((ra, ra))
    dec = np.hstack((dec, dec))
    total_magnitude = format_magnitude_for_galfit_single_sersic(sci_image_stamp_filename, mag_source,
                                                                neighbouring_sources_catalogue,
                                                                mag_key_neighbouring_sources_catalogue + '_' +
                                                                waveband,
                                                                ra_key_neighbouring_sources_catalogue,
                                                                dec_key_neighbouring_sources_catalogue,
                                                                image_size)
    total_magnitude = np.vstack((total_magnitude, total_magnitude))
    effective_radius = format_effective_radius_for_galfit_single_sersic(sci_image_stamp_filename,
                                                                        effective_radius_source,
                                                                        neighbouring_sources_catalogue,
                                                                        re_key_neighbouring_sources_catalogue +
                                                                        '_' + waveband,
                                                                        ra_key_neighbouring_sources_catalogue,
                                                                        dec_key_neighbouring_sources_catalogue,
                                                                        image_size)
    effective_radius = np.vstack((effective_radius, effective_radius))
    sersic_index = format_sersic_index_for_galfit_sersic_expdisk(sci_image_stamp_filename,
                                                                 neighbouring_sources_catalogue,
                                                                 ra_key_neighbouring_sources_catalogue,
                                                                 dec_key_neighbouring_sources_catalogue,
                                                                 image_size)
    axis_ratio = format_axis_ratio_for_galfit_single_sersic(sci_image_stamp_filename, minor_axis_source,
                                                            major_axis_source,
                                                            neighbouring_sources_catalogue,
                                                            minor_axis_key_neighbouring_sources_catalogue +
                                                            '_' + waveband,
                                                            major_axis_key_neighbouring_sources_catalogue +
                                                            '_' + waveband,
                                                            ra_key_neighbouring_sources_catalogue,
                                                            dec_key_neighbouring_sources_catalogue,
                                                            image_size)
    axis_ratio = np.vstack((axis_ratio, axis_ratio))
    position_angle = format_position_angle_for_galfit_single_sersic(sci_image_stamp_filename, position_angle_source,
                                                                    neighbouring_sources_catalogue,
                                                                    position_angle_key_neighbouring_sources_catalogue +
                                                                    '_' + waveband,
                                                                    ra_key_neighbouring_sources_catalogue,
                                                                    dec_key_neighbouring_sources_catalogue,
                                                                    image_size)
    position_angle = np.vstack((position_angle, position_angle))
    subtract = np.full((len(neighbouring_sources_catalogue) + 1) * 2, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio, \
        position_angle, subtract


def format_properties_for_stamps_galfit_single_sersic_deprecated(sci_image_stamp_filename, waveband, ra_source,
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
                                                                 enlarging_image_factor):
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
    source_positions = format_position_for_galfit_single_sersic(sci_image_stamp_filename, ra_source, dec_source,
                                                                neighbouring_sources_catalogue,
                                                                ra_key_neighbouring_sources_catalogue,
                                                                dec_key_neighbouring_sources_catalogue, image_size)
    ra, dec = format_ra_dec_for_galfit_single_sersic(ra_source, dec_source, neighbouring_sources_catalogue,
                                                     ra_key_neighbouring_sources_catalogue,
                                                     dec_key_neighbouring_sources_catalogue)
    total_magnitude = format_magnitude_for_galfit_single_sersic(sci_image_stamp_filename, mag_source,
                                                                neighbouring_sources_catalogue,
                                                                mag_key_neighbouring_sources_catalogue + '_' + waveband,
                                                                ra_key_neighbouring_sources_catalogue,
                                                                dec_key_neighbouring_sources_catalogue,
                                                                image_size)
    effective_radius = format_effective_radius_for_galfit_single_sersic(sci_image_stamp_filename,
                                                                        effective_radius_source,
                                                                        neighbouring_sources_catalogue,
                                                                        re_key_neighbouring_sources_catalogue +
                                                                        '_' + waveband,
                                                                        ra_key_neighbouring_sources_catalogue,
                                                                        dec_key_neighbouring_sources_catalogue,
                                                                        image_size)
    sersic_index = format_sersic_index_for_galfit_single_sersic(sci_image_stamp_filename,
                                                                neighbouring_sources_catalogue,
                                                                ra_key_neighbouring_sources_catalogue,
                                                                dec_key_neighbouring_sources_catalogue,
                                                                image_size)
    axis_ratio = format_axis_ratio_for_galfit_single_sersic(sci_image_stamp_filename, minor_axis_source,
                                                            major_axis_source,
                                                            neighbouring_sources_catalogue,
                                                            minor_axis_key_neighbouring_sources_catalogue +
                                                            '_' + waveband,
                                                            major_axis_key_neighbouring_sources_catalogue +
                                                            '_' + waveband,
                                                            ra_key_neighbouring_sources_catalogue,
                                                            dec_key_neighbouring_sources_catalogue,
                                                            image_size)
    position_angle = format_position_angle_for_galfit_single_sersic(sci_image_stamp_filename, position_angle_source,
                                                                    neighbouring_sources_catalogue,
                                                                    position_angle_key_neighbouring_sources_catalogue +
                                                                    '_' + waveband,
                                                                    ra_key_neighbouring_sources_catalogue,
                                                                    dec_key_neighbouring_sources_catalogue,
                                                                    image_size)
    subtract = np.full(len(neighbouring_sources_catalogue) + 1, '0')

    return light_profiles, source_positions, ra, dec, total_magnitude, effective_radius, sersic_index, axis_ratio,\
        position_angle, subtract


def create_galfit_inputfile_on_euler(input_galfit_filename, sci_image_filename, output_model_image_filename,
                                     sigma_image_filename, psf_image_filename, psf_sampling_factor,
                                     bad_pixel_mask_filename,
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
            if light_profiles[i] == 'sersic':
                f.write('# Object number {} \n'
                        '0) sersic # object type \n'
                        '1) {} {} {} {} #  position x, y \n'
                        '3) {} {} # Integrated magnitude \n'
                        '4) {} {} #  R_e (half-light radius) [pix] \n'
                        '5) {} {} #  Sersic index n (de Vaucouleurs n=4) \n'
                        '9) {} {} #  axis ratio (b/a) \n'
                        '10) {} {} #  position angle (PA) [deg: Up=0, Left=90] \n'
                        'Z) {} #  output option (0 = resid., 1 = Do not subtract)\n\n'
                        .format(str(i + 2),
                                source_positions[i][0],
                                source_positions[i][1],
                                int(source_positions[i][2]),
                                int(source_positions[i][3]),
                                total_magnitudes[i][0],
                                total_magnitudes[i][1],
                                effective_radii[i][0],
                                effective_radii[i][1],
                                sersic_indices[i][0],
                                sersic_indices[i][1],
                                axis_ratios[i][0],
                                axis_ratios[i][1],
                                position_angles[i][0],
                                position_angles[i][1],
                                subtract[i]))
            elif light_profiles[i] == 'devauc':
                f.write('# Object number {} \n'
                        '0) devauc # object type \n'
                        '1) {} {} {} {} #  position x, y \n'
                        '3) {} {} # Integrated magnitude \n'
                        '4) {} {} #  R_e (half-light radius) [pix] \n'
                        '9) {} {} #  axis ratio (b/a) \n'
                        '10) {} {} #  position angle (PA) [deg: Up=0, Left=90] \n'
                        'Z) {} #  output option (0 = resid., 1 = Do not subtract)\n\n'
                        .format(str(i + 2),
                                source_positions[i][0],
                                source_positions[i][1],
                                source_positions[i][2],
                                source_positions[i][3],
                                total_magnitudes[i][0],
                                total_magnitudes[i][1],
                                effective_radii[i][0],
                                effective_radii[i][1],
                                axis_ratios[i][0],
                                axis_ratios[i][1],
                                position_angles[i][0],
                                position_angles[i][1],
                                subtract[i]))
            elif light_profiles[i] == 'expdisk':
                f.write('# Object number {} \n'
                        '0) expdisk # object type \n'
                        '1) {} {} {} {} #  position x, y \n'
                        '3) {} {} # Integrated magnitude \n'
                        '4) {} {} #  Rs (scale radius) [pix] \n'
                        '9) {} {} #  axis ratio (b/a) \n'
                        '10) {} {} #  position angle (PA) [deg: Up=0, Left=90] \n'
                        'Z) {} #  output option (0 = resid., 1 = Do not subtract)\n\n'
                        .format(str(i + 2),
                                source_positions[i][0],
                                source_positions[i][1],
                                int(source_positions[i][2]),
                                int(source_positions[i][3]),
                                total_magnitudes[i][0],
                                total_magnitudes[i][1],
                                effective_radii[i][0],
                                effective_radii[i][1],
                                axis_ratios[i][0],
                                axis_ratios[i][1],
                                position_angles[i][0],
                                position_angles[i][1],
                                subtract[i]))
            else:
                logger.info('not implemented')
                raise ValueError
        f.write('================================================================================\n\n')
        f.close()


def run_galfit_deprecated(galfit_binary_file, input_galfit_filename, sci_image_filename, output_model_image_filename,
                          sigma_image_filename, psf_image_filename, bad_pixel_mask_filename, constraints_file_filename,
                          working_directory):
    """
    This function runs GALFIT from the command line.

    :param galfit_binary_file:
    :param input_galfit_filename:
    :param sci_image_filename:
    :param output_model_image_filename:
    :param sigma_image_filename:
    :param psf_image_filename:
    :param bad_pixel_mask_filename:
    :param constraints_file_filename:
    :param working_directory:
    :return None.
    """

    copy_files_to_working_directory(working_directory, input_galfit_filename, sci_image_filename,
                                    sigma_image_filename, psf_image_filename, bad_pixel_mask_filename,
                                    constraints_file_filename)
    os.system('{} {}'.format(galfit_binary_file, os.path.basename(input_galfit_filename)))
    remove_files_from_working_directory(input_galfit_filename, sci_image_filename, output_model_image_filename,
                                        sigma_image_filename, psf_image_filename, bad_pixel_mask_filename,
                                        constraints_file_filename)


def beta_seeing_evaluation_deprecated(sci_image_filename, sextractor_catalogue, ext_star_cat, pixscale,
                           background_noise_amp, seeing_initial_guess, match_type='sextractor_star_catalogue'):
    """

    :param sci_image_filename:
    :param sextractor_catalogue:
    :param ext_star_cat:
    :param pixscale:
    :param background_noise_amp:
    :param seeing_initial_guess:
    :param match_type:
    :return:
    """

    max_dist_arcsec = 1.0
    data = fits.getdata(sci_image_filename)

    if match_type == 'external_star_catalogue':
        select_in_gaia, not_select_in_gaia = match_sources_with_gaia(sextractor_catalogue, ext_star_cat,
                                                                     max_dist_arcsec)
        with fits.open(ext_star_cat) as f:
            starcat = f[1].data
        ra = np.array(starcat['ra'], dtype=float)[select_in_gaia]
        dec = np.array(starcat['dec'], dtype=float)[select_in_gaia]
        x_stars, y_stars = ra_dec_to_pixels(ra, dec, sci_image_filename)
    elif match_type == 'sextractor_star_catalogue':
        with fits.open(sextractor_catalogue) as f:
            starcat = f[1].data
            # signal_to_noise = starcat['FLUX_AUTO'] / starcat['FLUXERR_AUTO']
            w = np.where((starcat['FLAGS'] == 0) & (starcat['CLASS_STAR'] >= 0.95))
            starcat = starcat[w]
            x_stars = starcat['XWIN_IMAGE']
            y_stars = starcat['YWIN_IMAGE']
    else:
        raise ValueError('%s is not a valid entry.' % match_type)

    try:
        h = fits.getheader(sci_image_filename, ext=0)
        mask = np.where((x_stars > 0) & (x_stars < h['NAXIS1']) & (y_stars > 0) & (y_stars < h['NAXIS2']))
    except Exception as e:
        logger.info(e)
        h = fits.getheader(sci_image_filename, ext=1)
        mask = np.where((x_stars > 0) & (x_stars < h['NAXIS1']) & (y_stars > 0) & (y_stars < h['NAXIS2']))

    if mask[0].size != 0:
        x_stars = x_stars[mask]
        y_stars = y_stars[mask]
    else:
        logger.info('No GAIA stars found, using SExtractor stars...')
        # in this case sextractor_catalogue must contain only stars
        with fits.open(sextractor_catalogue) as g:
            cat = g[1].data
        w = np.where((cat['FLAGS'] == 0) & (cat['CLASS_STAR'] >= 0.95))
        x_stars = cat[w]['XWIN_IMAGE']
        y_stars = cat[w]['YWIN_IMAGE']

    fwhm_array = []
    beta_array = []

    for i in range(len(x_stars)):

        size = 51
        cutout = create_cutout_image(x_stars[i], y_stars[i], size, data)

        if np.isnan(np.min(cutout.data)):  # avoids nan in cutouts
            continue
        else:
            if len(cutout.data) == size & len(cutout.data[0]) == size:  # avoids stars at edges
                starting_point = [background_noise_amp, 1., int(size / 2), int(size / 2),
                                  seeing_initial_guess / pixscale, 3.5]
                try:
                    popt, fwhm, beta = fit_2d_moffat_profile(cutout.data, starting_point, size)
                    fwhm_array.append(fwhm)
                    beta_array.append(beta)
                except Exception as e:
                    logger.info(e)
                    pass
            else:
                continue

    fwhm_array = np.array(fwhm_array)
    beta_array = np.array(beta_array)

    mask_mean = np.where((fwhm_array * pixscale < 1.) & (fwhm_array * pixscale > 0.05))
    mask_beta = np.where((beta_array > 0) & (beta_array < 10))
    fwhm = np.nanmedian(fwhm_array[mask_mean]) * pixscale
    beta = np.nanmedian(beta_array[mask_beta])

    logger.info('Image: {}, FWHM: {}, Beta Moffat: {}'.format(sci_image_filename, fwhm, beta))

    return fwhm, beta
