#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
from astropy.io import fits
from astropy.table import Table, Column, TableReplaceWarning, vstack, hstack
from astropy import wcs
import numpy as np
from astropy.wcs import FITSFixedWarning
from h5py.h5py_warnings import H5pyDeprecationWarning
from sklearn.neighbors import BallTree
import starlink.Ast as Ast
import starlink.Atl as Atl
from PyAstronomy import pyasl
import pickle
import h5py
import os
import logging
import warnings
import sys
import subprocess
import glob

# morphofit imports


def get_hst_saturation(img_name):
    """

    :param img_name:
    :return:
    """

    data = fits.getdata(img_name)
    newdata = data[~np.isnan(data)]
    saturation = max(newdata)

    return saturation


def get_saturations(telescope_name, img_names, wavebands):
    """
    This function computes the maximum pixel value in an image as estimate of the saturation limit.

    :param telescope_name:
    :param img_names:
    :param wavebands:
    :return saturations: dict, dictionary of maximum pixel value in the image.
    """

    saturations_switcher = {'HST': get_hst_saturation}

    saturations = {}
    for name in img_names:
        idx_name = img_names.index(name)
        # data = fits.getdata(name)
        # newdata = data[~np.isnan(data)]
        # saturations[wavebands[idx_name]] = max(newdata)
        saturation_function = saturations_switcher.get(telescope_name, lambda: 'To be implemented...')
        saturations[wavebands[idx_name]] = saturation_function(name)

    return saturations


def get_hst_exposure_time(img_name):
    """

    :param img_name:
    :return:
    """

    h = fits.getheader(img_name, ext=0)
    exptime = h['EXPTIME']

    return exptime


def get_exposure_times(telescope_name, img_names, wavebands):
    """
    This function reads the exposure time from the image HEADER.

    :param telescope_name:
    :param img_names:
    :param wavebands:
    :return exptimes: dict, dictionary of exposure times.
    """

    exptimes_switcher = {'HST': get_hst_exposure_time}

    exptimes = {}
    for name in img_names:
        idx_name = img_names.index(name)
        exptime_function = exptimes_switcher.get(telescope_name, lambda: 'To be implemented...')
        exptimes[wavebands[idx_name]] = exptime_function(name)
        # if telescope_name == 'HST':
        #     h = fits.getheader(name, ext=0)
        #     exptimes[wavebands[idx_name]] = h['EXPTIME']
        # else:
        #     logger.info('To be implemented...')

    return exptimes


def get_hst_gain(img_name):
    """

    :param img_name:
    :return:
    """

    h = fits.getheader(img_name, ext=0)
    effective_gain = h['CCDGAIN'] * h['EXPTIME']
    instrumental_gain = h['CCDGAIN']

    return effective_gain, instrumental_gain


def get_gains(telescope_name, img_names, wavebands):
    """
    This function computes the gains for each HST image.

    :param telescope_name:
    :param img_names:
    :param wavebands:
    :return gains: dict, dictionary of gains in the images.
    """

    gains_switcher = {'HST': get_hst_gain}

    effective_gains = {}
    instrumental_gains = {}
    for name in img_names:
        idx_name = img_names.index(name)
        gain_function = gains_switcher.get(telescope_name, lambda: 'To be implemented...')
        effective_gains[wavebands[idx_name]], instrumental_gains[wavebands[idx_name]] = gain_function(name)
        # if telescope_name == 'HST':
        #     h = fits.getheader(name, ext=0)
        #     effective_gains[wavebands[idx_name]] = h['CCDGAIN'] * h['EXPTIME']
        #     instrumental_gains[wavebands[idx_name]] = h['CCDGAIN']
        # else:
        #     logger.info('To be implemented...')

    return effective_gains, instrumental_gains


def get_sims_gains(telescope_name, img_names, wavebands):
    """
    This function computes the gains for each HST image.

    :param telescope_name:
    :param img_names:
    :param wavebands:
    :return gains: dict, dictionary of gains in the images.
    """

    effective_gains = {}
    instrumental_gains = {}
    for name in img_names:
        idx_name = img_names.index(name)
        if telescope_name == 'HST':
            h = fits.getheader(name, ext=0)
            effective_gains[wavebands[idx_name]] = h['EFF_GAIN']
            instrumental_gains[wavebands[idx_name]] = h['CCDGAIN']
        else:
            logger.info('To be implemented...')

    return effective_gains, instrumental_gains


def get_hst_zeropoint(img_name, target_name=None, waveband=None):
    """

    :param img_name:
    :param target_name:
    :param waveband:
    :return:
    """

    zeropoints_tortorelli2018 = {'abells1063': {'f435w': 25.60744, 'f606w': 26.45534, 'f814w': 25.93702,
                                                'f105w': 26.25827, 'f125w': 26.23810,
                                                'f140w': 26.45705, 'f160w': 25.95015},
                                 'macs0416': {'f435w': 25.48920, 'f606w': 26.37124, 'f814w': 25.88468,
                                              'f105w': 26.22912, 'f125w': 26.21636,
                                              'f140w': 26.43956, 'f160w': 25.93665},
                                 'macs1149': {'f435w': 25.56322, 'f606w': 26.42389, 'f814w': 25.91745,
                                              'f105w': 26.24737, 'f125w': 26.22997,
                                              'f140w': 26.45051, 'f160w': 25.94510}}

    h = fits.getheader(img_name, ext=0)
    zpt = -2.5 * np.log10(h['PHOTFLAM']) - 5 * np.log10(h['PHOTPLAM']) - 2.408

    try:
        zeropoint = zeropoints_tortorelli2018[target_name][waveband]

    except Exception as e:
        logger.info(e)
        zeropoint = zpt

    return zeropoint


def get_zeropoints(telescope_name, target_name, img_names, wavebands):
    """
    This function returns the zeropoint of each image coorrected for the galactic extinction
    from the Schlegel dust maps. First we get A/E(B-V)_SFD from Schlafly and Finkbeiner 2011 for Rv = 3.1 and then we
    multiply it for E(B-V)_SFD taken from the Schlegel map. Alternatively, we read the extinction values from the
    table provided by the CLASH collaboration (`extinction' dictionary).

    :param telescope_name:
    :param target_name:
    :param img_names:
    :param wavebands:
    :return zeropoints: dict, dictionary of extinction corrected zeropoints.
    """

    # A_E_B_V_SFD = {'f435w': 3.610, 'f475w': 3.268, 'f606w': 2.471, 'f625w': 2.219, 'f775w': 1.629,
    #                'f814w': 1.526, 'f850lp': 1.243, 'f105w': 0.969, 'f125w': 0.726, 'f140w': 0.613, 'f160w': 0.512}
    # E_B_V_SFD = {'abell370': 0.032, 'abell2744': 0.013, 'abells1063': 0.012, 'macs0416': 0.041, 'macs0717': 0.077,
    #              'macs1149': 0.023, 'macs1206': 0.065}
    # extinction = {
    #     'abells1063': {'f435w': 0.05035, 'f475w': 0.04555, 'f606w': 0.03582, 'f625w': 0.03267, 'f775w': 0.02468,
    #                    'f814w': 0.02229, 'f850lp': 0.01802, 'f105w': 0.01241, 'f125w': 0.00926, 'f140w': 0.00745,
    #                    'f160w': 0.00575},
    #     'macs1149': {'f435w': 0.09457, 'f475w': 0.08555, 'f606w': 0.06727, 'f625w': 0.06136, 'f775w': 0.04635,
    #                  'f814w': 0.04186, 'f850lp': 0.03384, 'f105w': 0.02331, 'f125w': 0.01739, 'f140w': 0.01399,
    #                  'f160w': 0.01080}}
    # extinction = {'abells1063': {'f435w': 0.05035, 'f606w': 0.03582, 'f814w': 0.02229,
    #                              'f105w': 0.01241, 'f125w': 0.00926, 'f140w': 0.00745, 'f160w': 0.00575},
    #               'macs0416': {'f435w': 0.16859, 'f606w': 0.11992, 'f814w': 0.07463,
    #                            'f105w': 0.04156, 'f125w': 0.03100, 'f140w': 0.02494, 'f160w': 0.01925},
    #               'macs1149': {'f435w': 0.09457, 'f606w': 0.06727, 'f814w': 0.04186,
    #                            'f105w': 0.02331, 'f125w': 0.01739, 'f140w': 0.01399, 'f160w': 0.01080}}

    # zeropoints_tortorelli2018 = {'abells1063': {'f435w': 25.60744, 'f606w': 26.45534, 'f814w': 25.93702,
    #                                             'f105w': 26.25827, 'f125w': 26.23810,
    #                                             'f140w': 26.45705, 'f160w': 25.95015},
    #                              'macs0416': {'f435w': 25.48920, 'f606w': 26.37124, 'f814w': 25.88468,
    #                                           'f105w': 26.22912, 'f125w': 26.21636,
    #                                           'f140w': 26.43956, 'f160w': 25.93665},
    #                              'macs1149': {'f435w': 25.56322, 'f606w': 26.42389, 'f814w': 25.91745,
    #                                           'f105w': 26.24737, 'f125w': 26.22997,
    #                                           'f140w': 26.45051, 'f160w': 25.94510}}

    zeropoints_switcher = {'HST': get_hst_zeropoint}

    zeropoints = {}
    for name in img_names:
        idx_name = img_names.index(name)
        zeropoint_function = zeropoints_switcher.get(telescope_name, lambda: 'To be implemented...')
        zeropoints[wavebands[idx_name]] = zeropoint_function(name, target_name=target_name,
                                                             waveband=wavebands[idx_name])

        # zpt = zeropoints_switcher.get(telescope_name, lambda: 'To be implemented...')
        # zeropoints[wavebands[idx_name]] = zpt[target_name][wavebands[idx_name]]
        # if telescope_name == 'HST':
        #     h = fits.getheader(name, ext=0)
        #     zpt = -2.5 * np.log10(h['PHOTFLAM']) - 5 * np.log10(h['PHOTPLAM']) - 2.408
        #     try:
        #         zeropoints[wavebands[idx_name]] = zeropoints_tortorelli2018[target_name][wavebands[idx_name]]
        #         # zeropoints[wavebands[idx_name]] = zpt - extinction[target_name][wavebands[idx_name]]
        #         # zeropoints[wavebands[idx_name]] = zpt - (A_E_B_V_SFD[wavebands[idx_name]] * E_B_V_SFD[target_name])
        #     except Exception as e:
        #         logger.info(e)
        #         zeropoints[wavebands[idx_name]] = zpt
        # else:
        #     logger.info('To be implemented...')

    return zeropoints


def get_sims_zeropoints(telescope_name, img_names, wavebands):
    """
        This function returns the zeropoint of each simulated image.

        :param telescope_name:
        :param img_names:
        :param wavebands:
        :return zeropoints: dict, dictionary of extinction corrected zeropoints.
        """

    zeropoints = {}
    for name in img_names:
        idx_name = img_names.index(name)
        if telescope_name == 'HST':
            h = fits.getheader(name, ext=0)
            zpt = h['SEXMGZPT']
            zeropoints[wavebands[idx_name]] = zpt
        else:
            logger.info('To be implemented...')

    return zeropoints


def match_sources_with_star_catalogue(catalogue, star_catalogue, catalogue_ra_keyword, catalogue_dec_keyword,
                                      stars_ra_keyword, stars_dec_keyword, max_dist_arcsec, match_type='0'):
    """
    This function matches sources with star catalogue sources.

    :param catalogue: input catalogue.
    :param star_catalogue: GAIA input catalogue.
    :param catalogue_ra_keyword:
    :param catalogue_dec_keyword:
    :param stars_ra_keyword:
    :param stars_dec_keyword
    :param max_dist_arcsec: maximum arcsecond distance for matching.
    :param match_type: int, whether to select sources in the gaia catalogue (`0') or in the Source
                       Extractor catalogue (1)
    :return select_in_starcat: indices of sources matching with GAIA stars.
    :return not_select_in_starcat: indices of sources not matching with GAIA stars.
    """

    with fits.open(star_catalogue) as fh5:
        cat_gaia = fh5[1].data

    with fits.open(catalogue) as f:
        cat = f[1].data

    # w = np.where((cat['FLAGS'] == 0) & (cat['CLASS_STAR'] >= 0.95))
    # starcat = cat[w]
    starcat = cat

    gaia_ang = np.concatenate([np.array(cat_gaia[stars_dec_keyword], dtype=float)[:, np.newaxis] * np.pi / 180.,
                               np.array(cat_gaia[stars_ra_keyword], dtype=float)[:, np.newaxis] * np.pi / 180.], axis=1)
    cat_ang = np.concatenate([starcat[catalogue_dec_keyword][:, np.newaxis] * np.pi / 180.,
                              starcat[catalogue_ra_keyword][:, np.newaxis] * np.pi / 180.], axis=1)

    if match_type == '0':
        ball_tree = BallTree(cat_ang, metric='haversine')
        dist, ind = ball_tree.query(gaia_ang, k=1)
    elif match_type == '1':
        ball_tree = BallTree(gaia_ang, metric='haversine')
        dist, ind = ball_tree.query(cat_ang, k=1)
    else:
        raise ValueError('%s is not a valid entry, it should be either 0 or 1.' % match_type)

    dist_arcsec = dist[:, 0] / np.pi * 180. * 3600.
    select_in_starcat = dist_arcsec < max_dist_arcsec
    not_select_in_starcat = dist_arcsec > max_dist_arcsec

    return select_in_starcat, not_select_in_starcat


def ra_dec_to_pixels(ra_list, dec_list, img):
    """
    This function converts an external list of sky coordinates into pixel coordinates.

    :param ra_list: list of RA coordinates.
    :param dec_list: list of DEC coordinates.
    :param img: image filename.
    :return: x: x pixel coordinates.
    :return: y: y pixel coordinates.
    """

    def column(matrix, idx):
        return [float(row[idx]) for row in matrix]

    with fits.open(img) as f:
        try:
            fitschan = Ast.FitsChan(Atl.PyFITSAdapter(f[0]))
            wcsinfo = fitschan.read()
            wcsinfo.Digits = str(12)
        except Exception as e:
            logger.info(e)
            fitschan = Ast.FitsChan(Atl.PyFITSAdapter(f[1]))
            wcsinfo = fitschan.read()
            wcsinfo.Digits = str(12)

    sexagesimal = []

    for i in range(len(ra_list)):
        sexagesimal.append(pyasl.coordsDegToSexa(ra_list[i], dec_list[i], asString=True))

    ra_string = []
    dec_string = []

    for i in range(len(sexagesimal)):
        pieces = sexagesimal[i].split(' ')
        ra_string.append(pieces[0] + ':' + pieces[1] + ':' + pieces[2])
        dec_string.append(pieces[4] + ':' + pieces[5] + ':' + pieces[6])

    xworld = []
    yworld = []

    for i in range(len(dec_string)):
        if dec_string[i][5:11] == '60.000':
            dec_string[i] = dec_string[i][0:5] + '59.999'

    for (m, n) in zip(ra_string, dec_string):
        xworld.append(wcsinfo.unformat(1, m))
        yworld.append(wcsinfo.unformat(2, n))

    xworld = column(xworld, 1)
    yworld = column(yworld, 1)

    (x, y) = wcsinfo.tran([xworld, yworld], False)

    return x, y


def create_image_params_table(wavebands, saturations, effective_gains, instrumental_gains, exptimes,
                              zeropoints, bkg_amps, bkg_sigmas, fwhms, betas):
    """
    This function creates a fits table containing the measured instrumental parameters.

    :param wavebands:
    :param saturations:
    :param effective_gains:
    :param instrumental_gains:
    :param exptimes:
    :param zeropoints:
    :param bkg_amps:
    :param bkg_sigmas:
    :param fwhms:
    :param betas:
    :return table: fits table
    """
    wavebands = np.array(wavebands)
    saturation = np.array([value[1] for value in saturations.items()])
    effective_gain = np.array([value[1] for value in effective_gains.items()])
    instrumental_gain = np.array([value[1] for value in instrumental_gains.items()])
    exptime = np.array([value[1] for value in exptimes.items()])
    zeropoint = np.array([value[1] for value in zeropoints.items()])
    bkg_amp = np.array([value[1] for value in bkg_amps.items()])
    bkg_sigma = np.array([value[1] for value in bkg_sigmas.items()])
    fwhm = np.array([value[1] for value in fwhms.items()])
    beta = np.array([value[1] for value in betas.items()])
    table = Table([wavebands, saturation, effective_gain, instrumental_gain, exptime, zeropoint, bkg_amp,
                   bkg_sigma, fwhm, beta],
                  names=('wavebands', 'saturations', 'effective_gains', 'instrumental_gains', 'exptimes',
                         'zeropoints', 'bkg_amps', 'bkg_sigmas', 'fwhms', 'betas'))
    return table


def ra_dec_2_xy(ra, dec, image_path):
    """
    This function converts RA and DEC coordinates in pixel coordinates.

    :param ra:
    :param dec:
    :param image_path:
    :return x_new, y_new: x and y coordinates.
    """
    hdulist = fits.open(image_path)
    w = wcs.WCS(hdulist[0].header)
    world = []
    for i in range(len(ra)):
        world.append([ra[i], dec[i]])
    pixcrd2 = w.wcs_world2pix(world, 1)
    x_new = Column(pixcrd2[:, 0], name='x')
    y_new = Column(pixcrd2[:, 1], name='y')

    return x_new, y_new


def save_best_fit_properties_h5table(best_fit_properties_h5table_filename, light_profiles,
                                     psf_image_type, sigma_image_type, background_estimate_method,
                                     best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                                     best_fit_total_magnitudes,
                                     best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios,
                                     best_fit_position_angles, best_fit_background_value,
                                     best_fit_background_x_gradient,
                                     best_fit_background_y_gradient, reduced_chisquare):
    """

    :param best_fit_properties_h5table_filename:
    :param light_profiles:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param best_fit_source_x_positions:
    :param best_fit_source_y_positions:
    :param ra:
    :param dec:
    :param best_fit_total_magnitudes:
    :param best_fit_effective_radii:
    :param best_fit_sersic_indices:
    :param best_fit_axis_ratios:
    :param best_fit_position_angles:
    :param best_fit_background_value:
    :param best_fit_background_x_gradient:
    :param best_fit_background_y_gradient:
    :param reduced_chisquare:
    :return:
    """

    light_profiles = [name.encode('utf8') for name in light_profiles]

    with h5py.File(best_fit_properties_h5table_filename, 'w') as h5table:
        grp = h5table.create_group(name='{}_{}_{}'.format(psf_image_type, sigma_image_type, background_estimate_method))
        grp.create_dataset(name='light_profile', data=light_profiles)
        grp.create_dataset(name='x', data=best_fit_source_x_positions.astype(float))
        grp.create_dataset(name='y', data=best_fit_source_y_positions.astype(float))
        grp.create_dataset(name='ra', data=ra.astype(float))
        grp.create_dataset(name='dec', data=dec.astype(float))
        grp.create_dataset(name='mag', data=best_fit_total_magnitudes.astype(float))
        grp.create_dataset(name='Re', data=best_fit_effective_radii.astype(float))
        grp.create_dataset(name='n', data=best_fit_sersic_indices.astype(float))
        grp.create_dataset(name='ar', data=best_fit_axis_ratios.astype(float))
        grp.create_dataset(name='pa', data=best_fit_position_angles.astype(float))
        grp.create_dataset(name='chi_red', data=float(reduced_chisquare))
        grp.create_dataset(name='bkg_amp', data=best_fit_background_value.astype(float))
        grp.create_dataset(name='bkg_x_grad', data=best_fit_background_x_gradient.astype(float))
        grp.create_dataset(name='bkg_y_grad', data=best_fit_background_y_gradient.astype(float))


def create_best_fit_property_columns(best_fit_property, property_name):
    """

    :param best_fit_property:
    :param property_name:
    :return:
    """

    property_column = Column(np.array(best_fit_property[:, 0], dtype=float), property_name)
    property_err_column = Column(np.array(best_fit_property[:, 1], dtype=float), property_name + '_ERR')

    return property_column, property_err_column


def create_best_fit_single_property_columns(best_fit_property, number_of_sources, property_name):
    """

    :param best_fit_property:
    :param number_of_sources:
    :param property_name:
    :return:
    """

    property_column = Column(np.full(number_of_sources, best_fit_property[0], dtype=float), name=property_name)
    property_err_column = Column(np.full(number_of_sources, best_fit_property[1], dtype=float),
                                 name=property_name + '_ERR')

    return property_column, property_err_column


def create_best_fit_property_column(best_fit_property, property_name):
    """

    :param best_fit_property:
    :param property_name:
    :return:
    """

    property_column = Column(np.array(best_fit_property, dtype=float), property_name)

    return property_column


def create_best_fit_single_property_column(best_fit_property, number_of_sources, property_name):
    """

    :param best_fit_property:
    :param number_of_sources:
    :param property_name:
    :return:
    """

    property_column = Column(np.full(number_of_sources, best_fit_property, dtype=float), property_name)

    return property_column


def create_best_fit_properties_table(best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                                     best_fit_total_magnitudes, best_fit_effective_radii, best_fit_sersic_indices,
                                     best_fit_axis_ratios, best_fit_position_angles, best_fit_background_value,
                                     best_fit_background_x_gradient, best_fit_background_y_gradient, reduced_chisquare):
    """

    :param best_fit_source_x_positions:
    :param best_fit_source_y_positions:
    :param ra:
    :param dec:
    :param best_fit_total_magnitudes:
    :param best_fit_effective_radii:
    :param best_fit_sersic_indices:
    :param best_fit_axis_ratios:
    :param best_fit_position_angles:
    :param best_fit_background_value:
    :param best_fit_background_x_gradient:
    :param best_fit_background_y_gradient:
    :param reduced_chisquare:
    :return:
    """

    best_fit_source_x_positions_column, best_fit_source_x_positions_err_column = \
        create_best_fit_property_columns(best_fit_source_x_positions, 'X_GALFIT')
    best_fit_source_y_positions_column, best_fit_source_y_positions_err_column = \
        create_best_fit_property_columns(best_fit_source_y_positions, 'Y_GALFIT')
    ra_column = create_best_fit_property_column(ra, 'RA_GALFIT')
    dec_column = create_best_fit_property_column(dec, 'DEC_GALFIT')
    best_fit_total_magnitudes_column, best_fit_total_magnitudes_err_column = \
        create_best_fit_property_columns(best_fit_total_magnitudes, 'MAG_GALFIT')
    best_fit_effective_radii_column, best_fit_effective_radii_err_column = \
        create_best_fit_property_columns(best_fit_effective_radii, 'RE_GALFIT')
    best_fit_sersic_indices_column, best_fit_sersic_indices_err_column = \
        create_best_fit_property_columns(best_fit_sersic_indices, 'N_GALFIT')
    best_fit_axis_ratios_column, best_fit_axis_ratios_err_column = \
        create_best_fit_property_columns(best_fit_axis_ratios, 'AR_GALFIT')
    best_fit_position_angles_column, best_fit_position_angles_err_column = \
        create_best_fit_property_columns(best_fit_position_angles, 'PA_GALFIT')
    best_fit_background_value_column, best_fit_background_value_err_column = \
        create_best_fit_single_property_columns(best_fit_background_value, len(ra), 'BKG_VALUE_GALFIT')
    best_fit_background_x_gradient_column, best_fit_background_x_gradient_err_column = \
        create_best_fit_single_property_columns(best_fit_background_x_gradient, len(ra), 'BKG_X_GRAD_GALFIT')
    best_fit_background_y_gradient_column, best_fit_background_y_gradient_err_column = \
        create_best_fit_single_property_columns(best_fit_background_y_gradient, len(ra), 'BKG_Y_GRAD_GALFIT')
    reduced_chisquare_column = create_best_fit_single_property_column(reduced_chisquare, len(ra),
                                                                      'CHI_SQUARE_RED_GALFIT')

    table = hstack([best_fit_source_x_positions_column, best_fit_source_x_positions_err_column,
                    best_fit_source_y_positions_column, best_fit_source_y_positions_err_column,
                    ra_column, dec_column, best_fit_total_magnitudes_column, best_fit_total_magnitudes_err_column,
                    best_fit_effective_radii_column, best_fit_effective_radii_err_column,
                    best_fit_sersic_indices_column, best_fit_sersic_indices_err_column,
                    best_fit_axis_ratios_column, best_fit_axis_ratios_err_column,
                    best_fit_position_angles_column, best_fit_position_angles_err_column,
                    best_fit_background_value_column, best_fit_background_value_err_column,
                    best_fit_background_x_gradient_column, best_fit_background_x_gradient_err_column,
                    best_fit_background_y_gradient_column, best_fit_background_y_gradient_err_column,
                    reduced_chisquare_column])

    return table


def save_best_fit_properties_stamps(best_fit_properties_table_filename, target_galaxies_catalogue,
                                    neighbouring_galaxies_catalogue, target_field_name, stamp_index, waveband,
                                    telescope_name, galaxy_ids_key, component_number_key, light_profiles_key,
                                    light_profiles, psf_image_type, sigma_image_type, background_estimate_method,
                                    best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                                    best_fit_total_magnitudes, best_fit_effective_radii,
                                    best_fit_sersic_indices, best_fit_axis_ratios,
                                    best_fit_position_angles, best_fit_background_value,
                                    best_fit_background_x_gradient,
                                    best_fit_background_y_gradient, reduced_chisquare):
    """

    :param best_fit_properties_table_filename:
    :param target_galaxies_catalogue:
    :param neighbouring_galaxies_catalogue:
    :param target_field_name:
    :param stamp_index:
    :param waveband:
    :param telescope_name:
    :param galaxy_ids_key:
    :param component_number_key:
    :param light_profiles_key:
    :param light_profiles:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param best_fit_source_x_positions:
    :param best_fit_source_y_positions:
    :param ra:
    :param dec:
    :param best_fit_total_magnitudes:
    :param best_fit_effective_radii:
    :param best_fit_sersic_indices:
    :param best_fit_axis_ratios:
    :param best_fit_position_angles:
    :param best_fit_background_value:
    :param best_fit_background_x_gradient:
    :param best_fit_background_y_gradient:
    :param reduced_chisquare:
    :return:
    """

    galaxy_ids = vstack([target_galaxies_catalogue[galaxy_ids_key], neighbouring_galaxies_catalogue[galaxy_ids_key]])
    stamp_index_column = Column(np.full(len(galaxy_ids), stamp_index, dtype='U{}'.format(len(stamp_index))),
                                name='STAMP_INDEX')
    component_number = vstack([target_galaxies_catalogue[component_number_key],
                               neighbouring_galaxies_catalogue[component_number_key]])
    # component_number = vstack([target_galaxies_catalogue['{}_{}'.format(component_number_key, waveband)],
    #                            neighbouring_galaxies_catalogue['{}_{}'.format(component_number_key, waveband)]])
    # component_number.rename_column('{}_{}'.format(component_number_key, waveband), component_number_key)
    telescope_name_column = Column(np.full(len(galaxy_ids), telescope_name, dtype='U{}'.format(len(telescope_name))),
                                   name='TELESCOPE_NAME')
    target_field_name_column = Column(np.full(len(galaxy_ids), target_field_name,
                                              dtype='U{}'.format(len(target_field_name))), name='TARGET_FIELD_NAME')
    waveband_column = Column(np.full(len(galaxy_ids), waveband, dtype='U{}'.format(len(waveband))), name='WAVEBAND')
    sigma_image_type_column = Column(np.full(len(galaxy_ids), sigma_image_type,
                                             dtype='U{}'.format(len(sigma_image_type))), name='SIGMA_IMAGE_TYPE')
    background_estimate_method_column = Column(np.full(len(galaxy_ids), background_estimate_method,
                                                       dtype='U{}'.format(len(background_estimate_method))),
                                               name='BACKGROUND_ESTIMATE_METHOD')
    psf_image_type_column = Column(np.full(len(galaxy_ids), psf_image_type, dtype='U{}'.format(len(psf_image_type))),
                                   name='PSF_IMAGE_TYPE')
    light_profile_column = Column(light_profiles, name=light_profiles_key)

    table_of_indices = hstack([galaxy_ids, stamp_index_column, component_number, telescope_name_column,
                               target_field_name_column, waveband_column, sigma_image_type_column,
                               background_estimate_method_column, psf_image_type_column, light_profile_column])

    table_of_properties = create_best_fit_properties_table(best_fit_source_x_positions, best_fit_source_y_positions,
                                                           ra, dec, best_fit_total_magnitudes, best_fit_effective_radii,
                                                           best_fit_sersic_indices, best_fit_axis_ratios,
                                                           best_fit_position_angles, best_fit_background_value,
                                                           best_fit_background_x_gradient,
                                                           best_fit_background_y_gradient, reduced_chisquare)

    best_fit_properties_table = hstack([table_of_indices, table_of_properties])
    best_fit_properties_table.write(best_fit_properties_table_filename, format='fits', overwrite=True)


def save_best_fit_properties_regions(best_fit_properties_table_filename, source_galaxies_catalogue,
                                     target_field_name, region_index, waveband, telescope_name, galaxy_ids_key,
                                     component_number_key, light_profiles_key, light_profiles,
                                     psf_image_type, sigma_image_type, background_estimate_method,
                                     best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                                     best_fit_total_magnitudes,
                                     best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios,
                                     best_fit_position_angles, best_fit_background_value,
                                     best_fit_background_x_gradient,
                                     best_fit_background_y_gradient, reduced_chisquare):
    """

    :param best_fit_properties_table_filename:
    :param source_galaxies_catalogue:
    :param target_field_name:
    :param region_index:
    :param waveband:
    :param telescope_name:
    :param galaxy_ids_key:
    :param component_number_key:
    :param light_profiles_key:
    :param light_profiles:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param best_fit_source_x_positions:
    :param best_fit_source_y_positions:
    :param ra:
    :param dec:
    :param best_fit_total_magnitudes:
    :param best_fit_effective_radii:
    :param best_fit_sersic_indices:
    :param best_fit_axis_ratios:
    :param best_fit_position_angles:
    :param best_fit_background_value:
    :param best_fit_background_x_gradient:
    :param best_fit_background_y_gradient:
    :param reduced_chisquare:
    :return:
    """

    galaxy_ids = source_galaxies_catalogue[galaxy_ids_key]
    region_index_column = Column(np.full(len(galaxy_ids), region_index, dtype='U{}'.format(len(region_index))),
                                 name='REGION_INDEX')
    component_number = source_galaxies_catalogue[component_number_key]
    # component_number.rename_column('{}_{}'.format(component_number_key, waveband), component_number_key)
    telescope_name_column = Column(np.full(len(galaxy_ids), telescope_name, dtype='U{}'.format(len(telescope_name))),
                                   name='TELESCOPE_NAME')
    target_field_name_column = Column(np.full(len(galaxy_ids), target_field_name,
                                              dtype='U{}'.format(len(target_field_name))), name='TARGET_FIELD_NAME')
    waveband_column = Column(np.full(len(galaxy_ids), waveband, dtype='U{}'.format(len(waveband))), name='WAVEBAND')
    sigma_image_type_column = Column(np.full(len(galaxy_ids), sigma_image_type,
                                             dtype='U{}'.format(len(sigma_image_type))), name='SIGMA_IMAGE_TYPE')
    background_estimate_method_column = Column(np.full(len(galaxy_ids), background_estimate_method,
                                                       dtype='U{}'.format(len(background_estimate_method))),
                                               name='BACKGROUND_ESTIMATE_METHOD')
    psf_image_type_column = Column(np.full(len(galaxy_ids), psf_image_type, dtype='U{}'.format(len(psf_image_type))),
                                   name='PSF_IMAGE_TYPE')
    light_profile_column = Column(light_profiles, name=light_profiles_key)

    table_of_indices = hstack([galaxy_ids, region_index_column, component_number, telescope_name_column,
                               target_field_name_column, waveband_column, sigma_image_type_column,
                               background_estimate_method_column, psf_image_type_column, light_profile_column],
                              join_type='exact')

    table_of_properties = create_best_fit_properties_table(best_fit_source_x_positions, best_fit_source_y_positions,
                                                           ra, dec, best_fit_total_magnitudes, best_fit_effective_radii,
                                                           best_fit_sersic_indices, best_fit_axis_ratios,
                                                           best_fit_position_angles, best_fit_background_value,
                                                           best_fit_background_x_gradient,
                                                           best_fit_background_y_gradient, reduced_chisquare)

    best_fit_properties_table = hstack([table_of_indices, table_of_properties])
    best_fit_properties_table.write(best_fit_properties_table_filename, format='fits', overwrite=True)


def save_best_fit_properties(best_fit_properties_table_filename, source_galaxies_catalogue,
                             target_field_name, waveband, telescope_name, galaxy_ids_key,
                             component_number_key, light_profiles_key, light_profiles,
                             psf_image_type, sigma_image_type, background_estimate_method,
                             best_fit_source_x_positions, best_fit_source_y_positions, ra, dec,
                             best_fit_total_magnitudes,
                             best_fit_effective_radii, best_fit_sersic_indices, best_fit_axis_ratios,
                             best_fit_position_angles, best_fit_background_value,
                             best_fit_background_x_gradient,
                             best_fit_background_y_gradient, reduced_chisquare):
    """

    :param best_fit_properties_table_filename:
    :param source_galaxies_catalogue:
    :param target_field_name:
    :param waveband:
    :param telescope_name:
    :param galaxy_ids_key:
    :param component_number_key:
    :param light_profiles_key:
    :param light_profiles:
    :param psf_image_type:
    :param sigma_image_type:
    :param background_estimate_method:
    :param best_fit_source_x_positions:
    :param best_fit_source_y_positions:
    :param ra:
    :param dec:
    :param best_fit_total_magnitudes:
    :param best_fit_effective_radii:
    :param best_fit_sersic_indices:
    :param best_fit_axis_ratios:
    :param best_fit_position_angles:
    :param best_fit_background_value:
    :param best_fit_background_x_gradient:
    :param best_fit_background_y_gradient:
    :param reduced_chisquare:
    :return:
    """

    galaxy_ids = source_galaxies_catalogue[galaxy_ids_key]
    component_number = source_galaxies_catalogue[component_number_key]
    # component_number.rename_column('{}_{}'.format(component_number_key, waveband), component_number_key)
    telescope_name_column = Column(np.full(len(galaxy_ids), telescope_name, dtype='U{}'.format(len(telescope_name))),
                                   name='TELESCOPE_NAME')
    target_field_name_column = Column(np.full(len(galaxy_ids), target_field_name,
                                              dtype='U{}'.format(len(target_field_name))), name='TARGET_FIELD_NAME')
    waveband_column = Column(np.full(len(galaxy_ids), waveband, dtype='U{}'.format(len(waveband))), name='WAVEBAND')
    sigma_image_type_column = Column(np.full(len(galaxy_ids), sigma_image_type,
                                             dtype='U{}'.format(len(sigma_image_type))), name='SIGMA_IMAGE_TYPE')
    background_estimate_method_column = Column(np.full(len(galaxy_ids), background_estimate_method,
                                                       dtype='U{}'.format(len(background_estimate_method))),
                                               name='BACKGROUND_ESTIMATE_METHOD')
    psf_image_type_column = Column(np.full(len(galaxy_ids), psf_image_type, dtype='U{}'.format(len(psf_image_type))),
                                   name='PSF_IMAGE_TYPE')
    light_profile_column = Column(light_profiles, name=light_profiles_key)

    table_of_indices = hstack([galaxy_ids, component_number, telescope_name_column,
                               target_field_name_column, waveband_column, sigma_image_type_column,
                               background_estimate_method_column, psf_image_type_column, light_profile_column],
                              join_type='exact')

    table_of_properties = create_best_fit_properties_table(best_fit_source_x_positions, best_fit_source_y_positions,
                                                           ra, dec, best_fit_total_magnitudes, best_fit_effective_radii,
                                                           best_fit_sersic_indices, best_fit_axis_ratios,
                                                           best_fit_position_angles, best_fit_background_value,
                                                           best_fit_background_x_gradient,
                                                           best_fit_background_y_gradient, reduced_chisquare)

    best_fit_properties_table = hstack([table_of_indices, table_of_properties])
    best_fit_properties_table.write(best_fit_properties_table_filename, format='fits', overwrite=True)


def get_psf_image(root, psf_type, target_name, waveband):
    """
    This function assigns PSF paths to a variable.
    'psf_pca', 'direct' from stars in the field, 'indirect' from MultiKing

    :param root:
    :param psf_type:
    :param target_name:
    :param waveband:
    :return psf_image: PSF path.
    """

    psf_type_switcher = {'psf_pca': os.path.join(root, '{}/stars/pca_psf_{}_{}.fits'.format(target_name,
                                                                                            target_name, waveband)),
                         'direct': os.path.join(root, '{}/stars/obs_psf_{}_{}.fits'.format(target_name,
                                                                                           target_name, waveband)),
                         'indirect': os.path.join(root, '{}/stars/synth_psf_{}_{}.fits'.format(target_name,
                                                                                               target_name, waveband))}

    psf_image = psf_type_switcher.get(psf_type, lambda: 'To be implemented...')

    return psf_image


def save_stamps_dict_properties(root, dictionary, property_name, target_name, band, psf_type,
                                background_estimate_method, sigma_image_type, i):
    """

    :param root:
    :param dictionary:
    :param property_name:
    :param target_name:
    :param band:
    :param psf_type:
    :param background_estimate_method:
    :param sigma_image_type:
    :param i:
    :return:
    """

    f = open(os.path.join(root, "{}/stamps/pkl_files/{}_dict_{}_{}_{}_{}_{}_stamp{}.pkl"
                          .format(target_name, property_name, target_name, band, psf_type, background_estimate_method,
                                  sigma_image_type, i)), "wb")
    pickle.dump(dictionary, f)
    f.close()


def save_regions_dict_properties(root, dictionary, property_name, target_name, band, psf_type,
                                 background_estimate_method, sigma_image_type, i):
    """

    :param root:
    :param dictionary:
    :param property_name:
    :param target_name:
    :param band:
    :param psf_type:
    :param background_estimate_method:
    :param sigma_image_type:
    :param i:
    :return:
    """

    f = open(os.path.join(root, "{}/regions/pkl_files/{}_dict_{}_{}_{}_{}_{}_reg{}.pkl"
                          .format(target_name, property_name, target_name, band, psf_type, background_estimate_method,
                                  sigma_image_type, i)), "wb")
    pickle.dump(dictionary, f)
    f.close()


def save_fullimage_dict_properties(root, dictionary, property_name, target_name, band, psf_type,
                                   background_estimate_method, sigma_image_type):
    """

    :param root:
    :param dictionary:
    :param property_name:
    :param target_name:
    :param band:
    :param psf_type:
    :param background_estimate_method:
    :param sigma_image_type:
    :return:
    """

    f = open(os.path.join(root, "{}/full_image/pkl_files/{}_dict_{}_{}_{}_{}_{}.pkl"
                          .format(target_name, property_name, target_name, band, psf_type, background_estimate_method,
                                  sigma_image_type)), "wb")
    pickle.dump(dictionary, f)
    f.close()


def merge_dictionaries(pickle_file_paths):
    """

    :param pickle_file_paths:
    :return:
    """

    super_dict = {}
    for path in pickle_file_paths:
        file = open(path, 'rb')
        dictionary = pickle.load(file)
        for k, v in dictionary.items():
            super_dict.setdefault(k, []).append(v)
        file.close()

    return super_dict


def get_sky_background_region(background_estimate_method, param_table, waveband):
    """

    :param background_estimate_method:
    :param param_table:
    :param waveband:
    :return:
    """

    w = np.where(param_table['wavebands'] == '{}'.format(waveband))
    bkg_amp = param_table['bkg_amps'][w][0]
    # source_catalogue = Table.read(source_catalogue_filename, format='fits')

    back_sky, sky_x_grad, sky_y_grad, sky_subtract = 0, 0, 0, 0

    if background_estimate_method == 'sky_free_fit':
        back_sky = np.array([bkg_amp, 1])  # back_sky = np.array([np.nanmedian(source_catalogue['sky_value']), 1])
        sky_x_grad = np.array([0, 1])  # sky_x_grad = np.array([np.nanmedian(source_catalogue['sky_x_grad']), 1])
        sky_y_grad = np.array([0, 1])  # sky_y_grad = np.array([np.nanmedian(source_catalogue['sky_y_grad']), 1])
        sky_subtract = 0
    elif background_estimate_method == 'sky_fixed_value':
        back_sky = np.array([bkg_amp, 0])
        sky_x_grad = np.array([0, 0])
        sky_y_grad = np.array([0, 0])
        sky_subtract = 0
    else:
        logger.info('not implemented')

    return back_sky, sky_x_grad, sky_y_grad, sky_subtract


def single_ra_dec_2_xy(ra, dec, image_path):
    """

    :param ra:
    :param dec:
    :param image_path:
    :return:
    """

    hdulist = fits.open(image_path)
    w = wcs.WCS(hdulist[0].header)
    world = [[ra, dec]]
    pixcrd2 = w.wcs_world2pix(world, 1)
    x = pixcrd2[0, 0]
    y = pixcrd2[0, 1]

    return x, y


def compress_list_of_files(archive_path, archive_filename, files_path, list_of_files_to_compress):
    """

    :param archive_path:
    :param archive_filename:
    :param files_path:
    :param list_of_files_to_compress:
    :return:
    """

    subprocess.run(['tar', '-cvf', os.path.join(archive_path, archive_filename), '-C',
                    files_path] + list_of_files_to_compress)


def add_to_compressed_list_of_files(archive_path, archive_filename, files_path, file_to_add):
    """

    :param archive_path:
    :param archive_filename:
    :param files_path:
    :param file_to_add:
    :return:
    """

    subprocess.run(['tar', '-C', files_path, '--append',
                    '--file={}'.format(os.path.join(archive_path,
                                                    archive_filename)),
                    file_to_add])


def uncompress_files(destination_folder, archive_path, archive_filename):
    """

    :param destination_folder:
    :param archive_path:
    :param archive_filename:
    :return:
    """

    subprocess.run(['tar', '-C', destination_folder, '-xvf', os.path.join(archive_path, archive_filename)])


def compress_and_copy_files_for_galfit_stamps(index, root_target_field, temp_dir, files_archive_prefix,
                                              sci_image_filename, rms_image_filename, exp_image_filename,
                                              seg_image_filename, target_galaxies_catalogue_filename,
                                              source_galaxies_catalogue_filename,
                                              psf_image_filename, constraints_file_filename):
    """

    :param index:
    :param root_target_field:
    :param temp_dir:
    :param files_archive_prefix:
    :param sci_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param seg_image_filename:
    :param target_galaxies_catalogue_filename:
    :param source_galaxies_catalogue_filename:
    :param psf_image_filename:
    :param constraints_file_filename:
    :return:
    """

    files_to_compress = [os.path.basename(sci_image_filename), os.path.basename(seg_image_filename),
                         os.path.basename(target_galaxies_catalogue_filename),
                         os.path.basename(source_galaxies_catalogue_filename)]

    subprocess.run(['tar', '-cvf', os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                .format(files_archive_prefix, index)), '-C',
                    root_target_field] + files_to_compress)

    subprocess.run(['tar', '-C', os.path.join(root_target_field, 'stars'), '--append',
                    '--file={}'.format(os.path.join(root_target_field,
                                                    '{}_index{:06d}.tar'
                                                    .format(files_archive_prefix, index))),
                    os.path.basename(psf_image_filename)])

    if os.path.basename(rms_image_filename) == 'None':
        rms_image_filename = None
    else:
        subprocess.run(['tar', '-C', root_target_field, '--append',
                        '--file={}'.format(os.path.join(root_target_field,
                                                        '{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(rms_image_filename)])

    if os.path.basename(exp_image_filename) == 'None':
        exp_image_filename = None
    else:
        subprocess.run(['tar', '-C', root_target_field, '--append',
                        '--file={}'.format(os.path.join(root_target_field,
                                                        '{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(exp_image_filename)])

    if os.path.basename(constraints_file_filename) == 'None':
        constraints_file_filename = 'None'
    else:
        subprocess.run(['tar', '-C', root_target_field, '--append',
                        '--file={}'.format(os.path.join(root_target_field,
                                                        '{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(constraints_file_filename)])

    subprocess.run(['cp', os.path.join(root_target_field, '{}_index{:06d}.tar'.format(files_archive_prefix,
                                                                                      index)), temp_dir])

    subprocess.run(['tar', '-C', temp_dir, '-xvf', os.path.join(temp_dir, '{}_index{:06d}.tar'
                                                                .format(files_archive_prefix, index))])

    subprocess.run(['rm', '-rf', os.path.join(root_target_field, '{}_index{:06d}.tar'.format(files_archive_prefix,
                                                                                             index))])

    return rms_image_filename, exp_image_filename, constraints_file_filename


def compress_and_copy_files_for_galfit_regions(index, root_target_field, temp_dir, files_archive_prefix,
                                               sci_image_filename, rms_image_filename, exp_image_filename,
                                               seg_image_filename, sources_catalogue_filename,
                                               psf_image_filename, constraints_file_filename):
    """

    :param index:
    :param root_target_field:
    :param temp_dir:
    :param files_archive_prefix:
    :param sci_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param seg_image_filename:
    :param sources_catalogue_filename:
    :param psf_image_filename:
    :param constraints_file_filename:
    :return:
    """

    files_to_compress = [os.path.basename(sci_image_filename), os.path.basename(seg_image_filename)]

    subprocess.run(['tar', '-cvf', os.path.join(root_target_field, 'regions/{}_index{:06d}.tar'
                                                .format(files_archive_prefix, index)), '-C',
                    os.path.join(root_target_field, 'regions')] + files_to_compress)

    subprocess.run(['tar', '-C', root_target_field, '--append',
                    '--file={}'.format(os.path.join(root_target_field,
                                                    'regions/{}_index{:06d}.tar'
                                                    .format(files_archive_prefix, index))),
                    os.path.basename(sources_catalogue_filename)])

    subprocess.run(['tar', '-C', os.path.join(root_target_field, 'stars'), '--append',
                    '--file={}'.format(os.path.join(root_target_field,
                                                    'regions/{}_index{:06d}.tar'
                                                    .format(files_archive_prefix, index))),
                    os.path.basename(psf_image_filename)])

    if os.path.basename(rms_image_filename) == 'None':
        rms_image_filename = None
    else:
        subprocess.run(['tar', '-C', os.path.join(root_target_field, 'regions'), '--append',
                        '--file={}'.format(os.path.join(root_target_field,
                                                        'regions/{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(rms_image_filename)])

    if os.path.basename(exp_image_filename) == 'None':
        exp_image_filename = None
    else:
        subprocess.run(['tar', '-C', os.path.join(root_target_field, 'regions'), '--append',
                        '--file={}'.format(os.path.join(root_target_field,
                                                        'regions/{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(exp_image_filename)])

    if os.path.basename(constraints_file_filename) == 'None':
        constraints_file_filename = 'None'
    else:
        subprocess.run(['tar', '-C', os.path.join(root_target_field, 'regions'), '--append',
                        '--file={}'.format(os.path.join(root_target_field,
                                                        'regions/{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(constraints_file_filename)])

    subprocess.run(['cp', os.path.join(root_target_field, 'regions/{}_index{:06d}.tar'.format(files_archive_prefix,
                                                                                              index)), temp_dir])

    subprocess.run(['tar', '-C', temp_dir, '-xvf', os.path.join(temp_dir, '{}_index{:06d}.tar'
                                                                .format(files_archive_prefix, index))])

    subprocess.run(['rm', '-rf', os.path.join(root_target_field, 'regions/{}_index{:06d}.tar'
                                              .format(files_archive_prefix, index))])

    return rms_image_filename, exp_image_filename, constraints_file_filename


def compress_and_copy_files_for_galfit(index, root_target_field, temp_dir, files_archive_prefix,
                                       sci_image_filename, rms_image_filename, exp_image_filename,
                                       seg_image_filename, sources_catalogue_filename,
                                       psf_image_filename, constraints_file_filename):
    """

    :param index:
    :param root_target_field:
    :param temp_dir:
    :param files_archive_prefix:
    :param sci_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param seg_image_filename:
    :param sources_catalogue_filename:
    :param psf_image_filename:
    :param constraints_file_filename:
    :return:
    """

    files_to_compress = [os.path.basename(sci_image_filename), os.path.basename(seg_image_filename)]

    subprocess.run(['tar', '-cvf', os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                .format(files_archive_prefix, index)), '-C',
                    root_target_field] + files_to_compress)

    subprocess.run(['tar', '-C', root_target_field, '--append',
                    '--file={}'.format(os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                    .format(files_archive_prefix, index))),
                    os.path.basename(sources_catalogue_filename)])

    subprocess.run(['tar', '-C', os.path.join(root_target_field, 'stars'), '--append',
                    '--file={}'.format(os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                    .format(files_archive_prefix, index))),
                    os.path.basename(psf_image_filename)])

    if os.path.basename(rms_image_filename) == 'None':
        rms_image_filename = None
    else:
        subprocess.run(['tar', '-C', root_target_field, '--append',
                        '--file={}'.format(os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(rms_image_filename)])

    if os.path.basename(exp_image_filename) == 'None':
        exp_image_filename = None
    else:
        subprocess.run(['tar', '-C', root_target_field, '--append',
                        '--file={}'.format(os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(exp_image_filename)])

    if os.path.basename(constraints_file_filename) == 'None':
        constraints_file_filename = 'None'
    else:
        subprocess.run(['tar', '-C', root_target_field, '--append',
                        '--file={}'.format(os.path.join(root_target_field, '{}_index{:06d}.tar'
                                                        .format(files_archive_prefix, index))),
                        os.path.basename(constraints_file_filename)])

    subprocess.run(['cp', os.path.join(root_target_field, '{}_index{:06d}.tar'.format(files_archive_prefix, index)),
                    temp_dir])

    subprocess.run(['tar', '-C', temp_dir, '-xvf', os.path.join(temp_dir, '{}_index{:06d}.tar'
                                                                .format(files_archive_prefix, index))])

    subprocess.run(['rm', '-rf', os.path.join(root_target_field, '{}_index{:06d}.tar'.format(files_archive_prefix,
                                                                                             index))])

    return rms_image_filename, exp_image_filename, constraints_file_filename


def save_sextractor_output_files(temp_dir, output_directory, parameters_table_filename,
                                 single_band_catalogues, forced_single_band_catalogues,
                                 output_catalogue, detection_image_suffix, multiband_catalogue_suffix,
                                 sextractor_checkimages_endings):
    """

    :param temp_dir:
    :param output_directory:
    :param parameters_table_filename:
    :param single_band_catalogues:
    :param forced_single_band_catalogues:
    :param output_catalogue:
    :param detection_image_suffix:
    :param multiband_catalogue_suffix:
    :param sextractor_checkimages_endings:
    :return:
    """

    for checkimages_ending in sextractor_checkimages_endings:
        check_images = glob.glob(os.path.join(temp_dir, '*{}*'.format(checkimages_ending)))
        subprocess.run(['cp'] + check_images + [output_directory])

    subprocess.run(['cp'] + single_band_catalogues + [output_directory])
    subprocess.run(['cp'] + forced_single_band_catalogues + [output_directory])

    detection_files = glob.glob(os.path.join(temp_dir, '*{}*'.format(detection_image_suffix)))
    subprocess.run(['cp'] + detection_files + [output_directory])

    subprocess.run(['cp', parameters_table_filename, output_directory])

    subprocess.run(['cp', output_catalogue, output_directory])

    subprocess.run(['cp', output_catalogue.split(multiband_catalogue_suffix)[0]
                    + 'nostars.{}'.format(multiband_catalogue_suffix), output_directory])

    subprocess.run(['cp', output_catalogue.split(multiband_catalogue_suffix)[0]
                    + 'final.{}'.format(multiband_catalogue_suffix), output_directory])


def save_psf_output_files(psf_file_path, output_directory, wavebands, target_field_name, moffat_psf_flag=True,
                          observed_psf_flag=True, pca_psf_flag=True, effective_psf_flag=True):
    """

    :param psf_file_path:
    :param output_directory:
    :param wavebands:
    :param target_field_name:
    :param moffat_psf_flag:
    :param observed_psf_flag:
    :param pca_psf_flag:
    :param effective_psf_flag:
    :return:
    """

    if moffat_psf_flag:
        for waveband in wavebands:
            subprocess.run(['cp', os.path.join(psf_file_path, 'moffat_psf_{}_{}.fits'
                                               .format(target_field_name, waveband)), output_directory])

    if observed_psf_flag:
        for waveband in wavebands:
            subprocess.run(['cp', os.path.join(psf_file_path, 'observed_psf_{}_{}.fits'
                                               .format(target_field_name, waveband)), output_directory])

    if pca_psf_flag:
        for waveband in wavebands:
            subprocess.run(['cp', os.path.join(psf_file_path, 'pca_psf_{}_{}.fits'
                                               .format(target_field_name, waveband)), output_directory])

    if effective_psf_flag:
        for waveband in wavebands:
            subprocess.run(['cp', os.path.join(psf_file_path, 'effective_psf_{}_{}.fits'
                                               .format(target_field_name, waveband)), output_directory])


def save_galfit_stamps_output_files(output_directory, sci_image_stamp_filename, input_galfit_filename,
                                    output_model_image_filename, sigma_image_filename, bad_pixel_mask_filename,
                                    neighbouring_sources_catalogue, best_fit_properties_table_filename):
    """

    :param output_directory:
    :param sci_image_stamp_filename:
    :param input_galfit_filename:
    :param output_model_image_filename:
    :param sigma_image_filename:
    :param bad_pixel_mask_filename:
    :param neighbouring_sources_catalogue:
    :param best_fit_properties_table_filename:
    :return:
    """

    neighbouring_sources_catalogue.write(os.path.join(output_directory, 'neighbouring_source_galaxies_catalogue.fits'),
                                         format='fits', overwrite=True)
    files_to_copy = [sci_image_stamp_filename, input_galfit_filename, output_model_image_filename,
                     sigma_image_filename, bad_pixel_mask_filename, best_fit_properties_table_filename]
    subprocess.run(['cp'] + files_to_copy + [output_directory])


def save_galfit_output_files(output_directory, sci_image_filename, input_galfit_filename,
                             output_model_image_filename, sigma_image_filename, bad_pixel_mask_filename,
                             best_fit_properties_table_filename):
    """

    :param output_directory:
    :param sci_image_filename:
    :param input_galfit_filename:
    :param output_model_image_filename:
    :param sigma_image_filename:
    :param bad_pixel_mask_filename:
    :param best_fit_properties_table_filename:
    :return:
    """

    files_to_copy = [sci_image_filename, input_galfit_filename, output_model_image_filename,
                     sigma_image_filename, bad_pixel_mask_filename, best_fit_properties_table_filename]
    subprocess.run(['cp'] + files_to_copy + [output_directory])


def get_logger(file):
    """

    :param file:
    :return:
    """

    log = logging.getLogger(os.path.basename(file)[:10])

    if len(log.handlers) == 0:
        log_formatter = logging.Formatter("%(asctime)s %(name)0.10s %(levelname)0.3s   %(message)s ",
                                          "%y-%m-%d %H:%M:%S")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        log.addHandler(stream_handler)
        log.propagate = False
        log.setLevel(logging.INFO)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=H5pyDeprecationWarning)
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", category=TableReplaceWarning)

    return log


logger = get_logger(__file__)
