#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import os
from astropy.io import fits
from astropy.table import Table
import numpy as np
from astropy.nddata import Cutout2D
from scipy.optimize import curve_fit
# from scipy import linalg
from astropy.stats import sigma_clipped_stats
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from astropy.nddata import NDData
from photutils.psf import extract_stars
from photutils import EPSFBuilder
# from sklearn import preprocessing
from sklearn.decomposition import PCA

# morphofit imports
from morphofit.utils import match_sources_with_star_catalogue, ra_dec_to_pixels
from morphofit.utils import get_logger

logger = get_logger(__file__)


def two_dim_moffat_profile(x_y, sky, amplt, x0, y0, alpha, beta):
    """
    Two dimensional circular Moffat profile function.
    https://www.aspylib.com/doc/aspylib_fitting.html

    :param x_y: position coordinates.
    :param sky: sky background value.
    :param amplt: amplitude of the profile.
    :param x0: x central coordinate.
    :param y0: y central coordinate.
    :param alpha: alpha parameter related to FWHM.
    :param beta: beta parameter of the Moffat profile.
    :return moffat.ravel(): contiguous flattened Moffat profile array.
    """

    x, y = x_y
    moffat = sky + (amplt / ((1 + ((x - x0) ** 2 + (y - y0) ** 2) / (alpha ** 2)) ** beta))
    return moffat.ravel()


def create_cutout_image(x_star, y_star, size, data):
    """

    :param x_star:
    :param y_star:
    :param size:
    :param data:
    :return:
    """

    position = (x_star - 1, y_star)
    sizeimg = (size, size)  # pixels

    cutout = Cutout2D(data, position, sizeimg)

    return cutout


def subtract_background_from_image(data, sigma):
    """

    :param data:
    :param sigma:
    :return:
    """

    mean_back, median_back, std_back = sigma_clipped_stats(data, sigma=sigma)
    background_subtracted_data = data - median_back

    return background_subtracted_data


def subtract_2d_background_from_image(data, size=(30, 30), sigma=2, filter_size=(5, 5), iters=10):
    """

    :param data:
    :param size:
    :param sigma:
    :param filter_size:
    :param iters:
    :return:
    """

    sigma_clip = SigmaClip(sigma=sigma, iters=iters)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, size, filter_size=filter_size,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    background_subtracted_data = data - bkg.background

    return background_subtracted_data


def fit_2d_moffat_profile(data, starting_point, size):
    """

    :param data:
    :param starting_point:
    :param size:
    :return:
    """

    x_img = np.arange(0, size)
    y_img = np.arange(0, size)
    x_img, y_img = np.meshgrid(x_img, y_img)
    datatofit = data.ravel()
    popt, pcov = curve_fit(two_dim_moffat_profile, (x_img, y_img), datatofit, starting_point)
    fwhm = 2 * popt[4] * np.sqrt(2 ** (1 / popt[5]) - 1)
    beta = popt[5]

    return popt, fwhm, beta


def beta_seeing_evaluation(sci_image_filename, catalogue, ext_star_catalogue, pixscale,
                           background_noise_amp, seeing_initial_guess, catalogue_ra_keyword, catalogue_dec_keyword,
                           ext_stars_ra_keyword, ext_stars_dec_keyword):
    max_dist_arcsec = 1.0
    data = fits.getdata(sci_image_filename)
    select_in_star_cat, not_select_in_star_cat = match_sources_with_star_catalogue(catalogue, ext_star_catalogue,
                                                                                   catalogue_ra_keyword,
                                                                                   catalogue_dec_keyword,
                                                                                   ext_stars_ra_keyword,
                                                                                   ext_stars_dec_keyword,
                                                                                   max_dist_arcsec)

    with fits.open(ext_star_catalogue) as f:
        starcat = f[1].data
    ra = np.array(starcat[ext_stars_ra_keyword], dtype=float)[select_in_star_cat]
    dec = np.array(starcat[ext_stars_dec_keyword], dtype=float)[select_in_star_cat]
    x_stars, y_stars = ra_dec_to_pixels(ra, dec, sci_image_filename)

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
        logger.info('No stars found, raise error...')
        raise ValueError

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


def get_seeings(telescope_name, sci_images, wavebands, catalogues, ext_star_cat, pixel_scale,
                background_noise_amps, seeing_initial_guesses, catalogue_ra_keyword, catalogue_dec_keyword,
                ext_stars_ra_keyword, ext_stars_dec_keyword):
    """
    This function computes the fwhm and beta of stars by fitting a 2D circular Moffat profile.
    Stars are find by matching with an external star catalogue.

    :param telescope_name:
    :param sci_images:
    :param wavebands:
    :param catalogues:
    :param ext_star_cat:
    :param pixel_scale:
    :param background_noise_amps:
    :param seeing_initial_guesses:
    :param catalogue_ra_keyword:
    :param catalogue_dec_keyword:
    :param ext_stars_ra_keyword:
    :param ext_stars_dec_keyword:
    :return fwhms, betas: dict, dictionaries of fwhms and betas of the 2D Moffat profiles.
    """

    fwhms = {}
    betas = {}
    for name in sci_images:
        idx_name = sci_images.index(name)
        if telescope_name == 'HST':
            background_noise_amp = background_noise_amps[wavebands[idx_name]]
            seeing_initial_guess = seeing_initial_guesses[idx_name]
            fwhms[wavebands[idx_name]], betas[wavebands[idx_name]] = beta_seeing_evaluation(name,
                                                                                            catalogues[idx_name],
                                                                                            ext_star_cat,
                                                                                            pixel_scale,
                                                                                            background_noise_amp,
                                                                                            seeing_initial_guess,
                                                                                            catalogue_ra_keyword,
                                                                                            catalogue_dec_keyword,
                                                                                            ext_stars_ra_keyword,
                                                                                            ext_stars_dec_keyword)
        else:
            logger.info('To be implemented...')

    return fwhms, betas


def estimate_cutout_background(cutout, seg_cutout, sigma):
    mask = (seg_cutout == 0) & (cutout != 0)
    mean_back, median_back, std_back = sigma_clipped_stats(cutout[mask], sigma=sigma)

    return mean_back, median_back, std_back


def substitute_sources_with_background(cutout, seg_cutout, star_number):
    seg_cutout_copy = seg_cutout.copy()
    cutout_copy = cutout.copy()
    star_pixels = np.where(seg_cutout_copy == star_number)
    seg_cutout_copy[star_pixels] = 0
    mask = np.where(seg_cutout_copy != 0)

    sigma_clip = SigmaClip(sigma=4, maxiters=1)
    bkg_estimator = MedianBackground()
    bkg = Background2D(cutout_copy, (10, 10), filter_size=(2, 2),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    cutout_copy[mask] = bkg.background[mask]

    return cutout_copy


def create_moffat_psf_image_per_target(root_path, target_name, target_star_positions, target_stars_x_keyword,
                                       target_stars_y_keyword, sci_images, psf_image_size, wavebands,
                                       pixel_scale, target_param_table):
    """

    :param root_path:
    :param target_name:
    :param target_star_positions:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param psf_image_size:
    :param wavebands:
    :param pixel_scale:
    :param target_param_table:
    :return:
    """

    size = psf_image_size

    for name in sci_images:
        idx_name = sci_images.index(name)
        target_stars_table = Table.read(target_star_positions[idx_name], format='fits')
        x_stars = target_stars_table[target_stars_x_keyword]
        y_stars = target_stars_table[target_stars_y_keyword]
        idx_target_param_table = np.where(target_param_table['wavebands'] == wavebands[idx_name])
        background_noise_amp_initial_guess = target_param_table['bkg_amps'][idx_target_param_table][0]
        seeing_pixel_initial_guess = target_param_table['fwhms'][idx_target_param_table][0] / pixel_scale

        data, head = fits.getdata(name, header=True)
        popts = np.empty((len(x_stars), 6))
        for i in range(len(x_stars)):
            cutout = create_cutout_image(x_stars[i], y_stars[i], size, data)
            back_sub_cutout = subtract_background_from_image(cutout.data, sigma=2)
            starting_point = [background_noise_amp_initial_guess, 1., int(size / 2), int(size / 2),
                              seeing_pixel_initial_guess, 3.5]
            if back_sub_cutout.shape == (size, size):
                try:
                    popt, fwhm, beta = fit_2d_moffat_profile(back_sub_cutout, starting_point, size)
                    popts[i, :] = popt
                except RuntimeError:
                    popts[i, :] = np.full(len(starting_point), np.nan)
            else:
                continue
        x_img = np.arange(0, size)
        y_img = np.arange(0, size)
        x_y = np.meshgrid(x_img, y_img)
        sky = np.nanmedian(popts[:, 0])
        amplt = np.nanmedian(popts[:, 1])
        x0 = size / 2
        y0 = size / 2
        alpha = np.nanmedian(popts[:, 4])
        beta = np.nanmedian(popts[:, 5])
        moffat_psf = two_dim_moffat_profile(x_y, sky, amplt, x0, y0, alpha, beta)
        moffat_psf = moffat_psf / np.nanmax(moffat_psf)
        moffat_psf = moffat_psf.reshape((size, size))
        fits.writeto(os.path.join(root_path, 'moffat_psf_{}_{}.fits'.format(target_name, wavebands[idx_name])),
                     moffat_psf, head, overwrite=True)


def create_moffat_psf_image(root_path, target_star_positions, target_stars_id_keyword, target_stars_x_keyword,
                            target_stars_y_keyword, sci_images, seg_images, psf_image_size, wavebands,
                            pixel_scale, target_param_tables):
    """

    :param root_path:
    :param target_star_positions:
    :param target_stars_id_keyword:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param seg_images:
    :param psf_image_size:
    :param wavebands:
    :param pixel_scale:
    :param target_param_tables:
    :return:
    """

    size = psf_image_size

    for waveband in wavebands:
        idx_waveband = wavebands.index(waveband)
        psf_cutouts = []

        for name in sci_images[idx_waveband]:
            idx_name = sci_images[idx_waveband].index(name)
            target_stars_table = Table.read(target_star_positions[idx_waveband][idx_name], format='fits')
            x_stars = target_stars_table[target_stars_x_keyword]
            y_stars = target_stars_table[target_stars_y_keyword]
            stars_number = target_stars_table[target_stars_id_keyword]
            target_param_table = Table.read(target_param_tables[idx_waveband][idx_name], format='fits')
            idx_target_param_table = np.where(target_param_table['wavebands'] == waveband)
            background_noise_amp_initial_guess = target_param_table['bkg_amps'][idx_target_param_table][0]
            seeing_pixel_initial_guess = target_param_table['fwhms'][idx_target_param_table][0] / pixel_scale

            data, head = fits.getdata(name, header=True)
            seg_data, seg_head = fits.getdata(seg_images[idx_waveband][idx_name], header=True)
            popts = np.empty((len(x_stars), 6))
            for i in range(len(x_stars)):
                cutout = create_cutout_image(x_stars[i], y_stars[i], size, data)
                seg_cutout = create_cutout_image(x_stars[i] - 0.5, y_stars[i] - 1.5, size, seg_data)
                back_sub_cutout = substitute_sources_with_background(cutout.data, seg_cutout.data, stars_number[i])
                mean_back, median_back, std_back = estimate_cutout_background(back_sub_cutout, seg_cutout.data,
                                                                              sigma=2)
                back_sub_cutout = back_sub_cutout - mean_back
                starting_point = [background_noise_amp_initial_guess, 1., int(size / 2), int(size / 2),
                                  seeing_pixel_initial_guess, 3.5]
                if back_sub_cutout.shape == (size, size):
                    try:
                        popt, fwhm, beta = fit_2d_moffat_profile(back_sub_cutout, starting_point, size)
                        popts[i, :] = popt
                    except RuntimeError:
                        popts[i, :] = np.full(len(starting_point), np.nan)
                else:
                    continue
            x_img = np.arange(0, size)
            y_img = np.arange(0, size)
            x_y = np.meshgrid(x_img, y_img)
            sky = np.nanmedian(popts[:, 0])
            amplt = np.nanmedian(popts[:, 1])
            x0 = size / 2
            y0 = size / 2
            alpha = np.nanmedian(popts[:, 4])
            beta = np.nanmedian(popts[:, 5])
            moffat_psf = two_dim_moffat_profile(x_y, sky, amplt, x0, y0, alpha, beta)
            moffat_psf = moffat_psf / np.nanmax(moffat_psf)
            moffat_psf = moffat_psf.reshape((size, size))
            psf_cutouts.append(moffat_psf)

        fits.writeto(os.path.join(root_path, 'moffat_psf_{}.fits'.format(waveband)),
                     np.nanmedian(psf_cutouts, axis=0), overwrite=True)


def create_observed_psf_image_per_target(root_path, target_name, target_star_positions, target_stars_x_keyword,
                                         target_stars_y_keyword, sci_images, psf_image_size, wavebands):
    """
    Possible improvements: different kind of normalization, median mean or sum? align them by the brightest pixel

    :param root_path:
    :param target_name:
    :param target_star_positions:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param psf_image_size:
    :param wavebands:
    :return:
    """

    size = psf_image_size

    for name in sci_images:
        idx_name = sci_images.index(name)
        target_stars_table = Table.read(target_star_positions[idx_name], format='fits')
        x_stars = target_stars_table[target_stars_x_keyword]
        y_stars = target_stars_table[target_stars_y_keyword]
        psf_cutout = np.empty((len(x_stars), size, size))
        data, head = fits.getdata(name, header=True)
        for i in range(len(x_stars)):
            cutout = create_cutout_image(x_stars[i]-0.5, y_stars[i]-1.5, size, data)
            if (len(cutout.data) == size) & (len(cutout.data[0]) == size):
                psf_cutout[i] = subtract_background_from_image(cutout.data, sigma=2)
                psf_cutout[i] = psf_cutout[i] / np.nanmax(psf_cutout[i])
            else:
                continue

        fits.writeto(os.path.join(root_path, 'observed_psf_{}_{}.fits'.format(target_name, wavebands[idx_name])),
                     np.nanmedian(psf_cutout, axis=0),
                     head, overwrite=True)


def create_observed_psf_image(root_path, target_star_positions, target_stars_id_keyword, target_stars_x_keyword,
                              target_stars_y_keyword, sci_images, seg_images,
                              psf_image_size, wavebands):
    """

    :param root_path:
    :param target_star_positions:
    :param target_stars_id_keyword:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param seg_images:
    :param psf_image_size:
    :param wavebands:
    :return:
    """

    size = psf_image_size

    for waveband in wavebands:
        idx_waveband = wavebands.index(waveband)
        psf_cutouts = []

        for name in sci_images[idx_waveband]:
            idx_name = sci_images[idx_waveband].index(name)
            target_stars_table = Table.read(target_star_positions[idx_waveband][idx_name], format='fits')
            x_stars = target_stars_table[target_stars_x_keyword]
            y_stars = target_stars_table[target_stars_y_keyword]
            stars_number = target_stars_table[target_stars_id_keyword]
            psf_cutout = np.empty((len(x_stars), size, size))
            data, head = fits.getdata(name, header=True)
            seg_data, seg_head = fits.getdata(seg_images[idx_waveband][idx_name], header=True)

            for i in range(len(x_stars)):
                cutout = create_cutout_image(x_stars[i] - 0.5, y_stars[i] - 1.5, size, data)
                seg_cutout = create_cutout_image(x_stars[i] - 0.5, y_stars[i] - 1.5, size, seg_data)
                if (len(cutout.data) == size) & (len(cutout.data[0]) == size):
                    psf_cutout[i] = substitute_sources_with_background(cutout.data, seg_cutout.data, stars_number[i])
                    mean_back, median_back, std_back = estimate_cutout_background(psf_cutout[i], seg_cutout.data,
                                                                                  sigma=2)
                    psf_cutout[i] = psf_cutout[i] - mean_back
                    psf_cutout[i] = psf_cutout[i] / np.nanmax(psf_cutout[i])
                else:
                    continue

            psf_cutouts.append(psf_cutout)

        psf_cutouts = np.concatenate(psf_cutouts)

        fits.writeto(os.path.join(root_path, 'observed_psf_{}.fits'.format(waveband)),
                     np.nanmedian(psf_cutouts, axis=0), overwrite=True)


def create_pca_psf_image_per_target(root_path, target_name, target_star_positions, target_stars_x_keyword,
                                    target_stars_y_keyword, sci_images, psf_image_size, wavebands):
    """

    :param root_path:
    :param target_name:
    :param target_star_positions:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param psf_image_size:
    :param wavebands:
    :return:
    """

    # per fare la pca le immagini vanno centrate normalizzate e poi mean subtracted
    size = psf_image_size
    # principal_components = np.empty((len(wavebands), size ** 2, size, size))

    for name in sci_images:
        idx_name = sci_images.index(name)
        target_stars_table = Table.read(target_star_positions[idx_name], format='fits')
        x_stars = target_stars_table[target_stars_x_keyword]
        y_stars = target_stars_table[target_stars_y_keyword]
        psf_cutout = np.empty((len(x_stars), size, size))
        data, head = fits.getdata(name, header=True)
        S = np.empty((len(x_stars), size ** 2))
        for i in range(len(x_stars)):
            cutout = create_cutout_image(x_stars[i]-0.5, y_stars[i]-1.5, size, data)
            if (len(cutout.data) == size) & (len(cutout.data[0]) == size):
                psf_cutout[i] = subtract_background_from_image(cutout.data, sigma=2)
                # psf_cutout[i] = psf_cutout[i] / np.nanmax(psf_cutout[i])
                # psf_cutout_raveled = np.ravel(psf_cutout[i].transpose())
                # psf_cutout_norm = preprocessing.normalize(np.reshape(psf_cutout_raveled, (1,
                # len(psf_cutout_raveled))))
                S[i, :] = np.ravel(psf_cutout[i].transpose())
                # S[i, :] = psf_cutout_norm
            else:
                continue
        goodindices = []
        for k in range(len(S)):
            if np.isnan(S[k, :]).any():
                continue
            else:
                goodindices.append(k)
        S_clean = S[goodindices]
        S_clean_mean_sub = S_clean - np.mean(S_clean, axis=0)
        # U, s, VT = linalg.svd(S_clean_mean_sub, full_matrices=True)
        # principal_components[idx_name, :, :, :] = VT.reshape((size ** 2, size, size))
        # pca_star = principal_components[idx_name, 0, :, :]  # + principal_components[idx_band,1,:,:] + ...
        # pca_star = pca_star / np.nanmin(pca_star)
        n_components = min(len(x_stars), 5)
        pca_model = PCA(n_components=n_components)
        pca_model.fit(S_clean_mean_sub)
        pca_pc = pca_model.components_
        pca_star = np.reshape(pca_pc[0, :], (size, size))  # + np.reshape(pca_pc[1,:],(size, size)) + \
        #            np.reshape(pca_pc[2,:],(size, size)) + np.reshape(pca_pc[3,:],(size, size)) + \
        #            np.reshape(pca_pc[4,:],(size, size))
        head['NAXIS1'] = size
        head['NAXIS2'] = size

        fits.writeto(os.path.join(root_path, 'pca_psf_{}_{}.fits'.format(target_name, wavebands[idx_name])),
                     pca_star, head, overwrite=True)


def create_pca_psf_image(root_path, target_star_positions, target_stars_id_keyword, target_stars_x_keyword,
                         target_stars_y_keyword, sci_images, seg_images,
                         psf_image_size, wavebands):
    """

    :param root_path:
    :param target_star_positions:
    :param target_stars_id_keyword:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param seg_images:
    :param psf_image_size:
    :param wavebands:
    :return:
    """

    size = psf_image_size

    for waveband in wavebands:
        idx_waveband = wavebands.index(waveband)
        psf_cutouts = []

        for name in sci_images[idx_waveband]:
            idx_name = sci_images[idx_waveband].index(name)
            target_stars_table = Table.read(target_star_positions[idx_waveband][idx_name], format='fits')
            x_stars = target_stars_table[target_stars_x_keyword]
            y_stars = target_stars_table[target_stars_y_keyword]
            stars_number = target_stars_table[target_stars_id_keyword]
            psf_cutout = np.empty((len(x_stars), size, size))
            data, head = fits.getdata(name, header=True)
            seg_data, seg_head = fits.getdata(seg_images[idx_waveband][idx_name], header=True)
            S = np.empty((len(x_stars), size ** 2))
            for i in range(len(x_stars)):
                cutout = create_cutout_image(x_stars[i], y_stars[i], size, data)
                seg_cutout = create_cutout_image(x_stars[i], y_stars[i], size, seg_data)
                if (len(cutout.data) == size) & (len(cutout.data[0]) == size):
                    psf_cutout[i] = substitute_sources_with_background(cutout.data, seg_cutout.data, stars_number[i])
                    mean_back, median_back, std_back = estimate_cutout_background(psf_cutout[i], seg_cutout.data,
                                                                                  sigma=2)
                    psf_cutout[i] = psf_cutout[i] - mean_back
                    S[i, :] = np.ravel(psf_cutout[i].transpose())
                else:
                    continue
            psf_cutouts.append(S)

        psf_cutouts = np.concatenate(psf_cutouts)
        goodindices = []
        for k in range(len(S)):
            if np.isnan(S[k, :]).any():
                continue
            else:
                goodindices.append(k)
        S_clean = psf_cutouts[goodindices]
        S_clean_mean_sub = S_clean - np.mean(S_clean, axis=0)
        # n_components = min(len(x_stars), 5)
        pca_model = PCA(n_components=1)
        pca_model.fit(S_clean_mean_sub)
        pca_pc = pca_model.components_
        pca_star = np.reshape(pca_pc[0, :], (size, size))

        fits.writeto(os.path.join(root_path, 'pca_psf_{}.fits'.format(waveband)),
                     pca_star, overwrite=True)


def create_effective_psf_image_per_target(root_path, target_name, target_star_positions, target_stars_x_keyword,
                                          target_stars_y_keyword, sci_images, psf_image_size, wavebands):
    """

    :param root_path:
    :param target_name:
    :param target_star_positions:
    :param target_stars_x_keyword:
    :param target_stars_y_keyword:
    :param sci_images:
    :param psf_image_size:
    :param wavebands:
    :return:
    """

    size = psf_image_size

    for name in sci_images:
        idx_name = sci_images.index(name)
        data, head = fits.getdata(name, header=True)
        back_sub_data = subtract_background_from_image(data, sigma=2)
        nddata = NDData(data=back_sub_data)
        target_stars_table = Table.read(target_star_positions[idx_name], format='fits')
        stars_tbl = Table()
        stars_tbl['x'] = target_stars_table[target_stars_x_keyword]
        stars_tbl['y'] = target_stars_table[target_stars_y_keyword]
        stars = extract_stars(nddata, stars_tbl, size=size)

        epsf_builder = EPSFBuilder(oversampling=2, progress_bar=False,
                                   smoothing_kernel='quadratic',
                                   recentering_maxiters=20, maxiters=10,
                                   norm_radius=5.5, shift_val=0.5,
                                   recentering_boxsize=(5, 5), center_accuracy=1.0e-3)

        epsf, fitted_stars = epsf_builder(stars)

        fits.writeto(os.path.join(root_path, 'effective_psf_{}_{}.fits'.format(target_name, wavebands[idx_name])),
                     epsf.data, head, overwrite=True)
