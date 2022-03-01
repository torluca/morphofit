#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
from astropy.table import Table
import numpy as np
from astropy.io import fits
import scipy.stats
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground

# morphofit imports
from morphofit.run_sextractor import get_sextractor_cmd, run_sex_single_mode
from morphofit.utils import get_logger

logger = get_logger(__file__)


def oned_background_estimate(img_name, sextractor_catalog_name, seeing, saturation, zeropoint, gain, pixelscale,
                             photo_cmd, sextractor_binary, sextractor_config, sextractor_params, sextractor_filter,
                             sextractor_nnw, sextractor_checkimages, sextractor_checkimages_endings, rms_image=''):
    """
    This function computes the background amplitude and rms of images. First we run SE to obtain a segmentation map,
    then we select those pixels which do not belong to sources and on these we perform an iterative sigma clipping.

    :param img_name:
    :param sextractor_catalog_name:
    :param seeing:
    :param saturation:
    :param zeropoint:
    :param gain:
    :param pixelscale:
    :param photo_cmd:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :param rms_image:
    :return back_median, back_std, back_se_median, back_se_std: background amplitudes and rms estimated from the sigma
    clipping and from SE `BACKGROUND' parameter.
    """

    cmd = get_sextractor_cmd(img_name, sextractor_catalog_name, seeing, saturation,
                             zeropoint, gain, pixelscale, photo_cmd,
                             sextractor_binary, sextractor_config,
                             sextractor_params, sextractor_filter,
                             sextractor_nnw, sextractor_checkimages,
                             sextractor_checkimages_endings, rms_image=rms_image)
    run_sex_single_mode(cmd)
    table = Table.read(sextractor_catalog_name, format='fits')
    flags = table['FLAGS']
    class_star = table['CLASS_STAR']
    w = np.where((flags == 0) & (class_star < 0.95))
    back_se_median = np.nanmedian(table[w]['BACKGROUND'])
    back_se_std = np.std(table[w]['BACKGROUND'])
    image = fits.getdata(img_name, ext=0)
    segmap = fits.getdata(img_name.split('.fits')[0] + '_{}.fits'.format(sextractor_checkimages_endings[0]), ext=0)
    mask = (segmap == 0) & (image != 0)
    image_clipped = scipy.stats.sigmaclip(image[mask], low=4.0, high=4.0)
    back_median = np.nanmedian(image_clipped[0])
    back_std = np.nanstd(image_clipped[0])
    logger.info('background median value: {}, background std value: {}'.format(back_median, back_std))

    return back_median, back_std, back_se_median, back_se_std


def twod_background_estimate(img_name, sigma, iters, box_size, filter_size):
    """
    This function estimates the background amplitude and rms by creating a 2D sigma clipped background map.
    Not used for the moment.

    :param img_name:
    :param sigma:
    :param iters:
    :param box_size:
    :param filter_size:
    :return:
    """
    image, h = fits.getdata(img_name, header=True)
    sigma_clip = SigmaClip(sigma=sigma, iters=iters)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (box_size, box_size), filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    fits.writeto(img_name.split('.fits')[0] + '_bkg.fits', bkg.background, h, overwrite=True)

    return bkg.background_median, bkg.background_rms_median


def get_hst_background_parameters(img_names, wavebands, se_cats, saturations, zeropoints, gains, pixel_scale,
                                  photo_cmd, sextractor_binary, sextractor_config, sextractor_params,
                                  sextractor_filter, sextractor_nnw, sextractor_checkimages,
                                  sextractor_checkimages_endings, rms_images=None):
    """
    This function computes the background noise parameters.

    :param img_names:
    :param wavebands:
    :param se_cats:
    :param saturations:
    :param zeropoints:
    :param gains:
    :param pixel_scale:
    :param photo_cmd:
    :param rms_images:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :return:
    """

    if rms_images is None:
        rms_images = []

    bkg_amps = {}
    bkg_sigmas = {}
    if not rms_images:
        rms_images = ['']
    for name in img_names:
        idx_name = img_names.index(name)
        back_median, back_std, back_se_median, back_se_std = oned_background_estimate(name, se_cats[idx_name],
                                                                                      0.1,
                                                                                      saturations[wavebands[idx_name]],
                                                                                      zeropoints[wavebands[idx_name]],
                                                                                      gains[wavebands[idx_name]],
                                                                                      pixel_scale,
                                                                                      photo_cmd,
                                                                                      sextractor_binary,
                                                                                      sextractor_config,
                                                                                      sextractor_params,
                                                                                      sextractor_filter,
                                                                                      sextractor_nnw,
                                                                                      sextractor_checkimages,
                                                                                      sextractor_checkimages_endings,
                                                                                      rms_image=rms_images[idx_name])
        bkg_amps[wavebands[idx_name]] = back_median
        bkg_sigmas[wavebands[idx_name]] = back_std

    return bkg_amps, bkg_sigmas


def local_background_estimate(sci_image_filename, seg_image_filename, background_value,
                              local_estimate=True):
    """

    :param sci_image_filename:
    :param seg_image_filename:
    :param background_value:
    :param local_estimate:
    :return:
    """

    if local_estimate:
        image = fits.getdata(sci_image_filename, ext=0)
        segmap = fits.getdata(seg_image_filename, ext=0)
        mask = (segmap == 0) & (image != 0)
        image_clipped = scipy.stats.sigmaclip(image[mask], low=2.0, high=2.0)
        background_value = np.nanmedian(image_clipped[0])

    return background_value
