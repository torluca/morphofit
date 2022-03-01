#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import os
from pkg_resources import resource_filename
from astropy.io import fits
import subprocess

# morphofit imports
import morphofit
from morphofit.utils import get_logger

logger = get_logger(__file__)


def get_sextractor_cmd(sci_image, sextractor_catalogue_name, psf_fwhm, saturation_level, magzero, gain, pixscale,
                       photo_cmd, sextractor_binary='/usr/local/bin/sex',
                       sextractor_config='default.sex', sextractor_params='default.param',
                       sextractor_filter='gauss_3.0_5x5.conv', sextractor_nnw='default.nnw',
                       sextractor_checkimages=None,
                       sextractor_checkimages_endings=None, rms_image=None):
    """
    This function creates the command file for running Source Extractor.

    :param sci_image:
    :param sextractor_catalogue_name:
    :param psf_fwhm:
    :param saturation_level:
    :param magzero:
    :param gain:
    :param pixscale:
    :param photo_cmd:
    :param rms_image:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :return:
    """

    if sextractor_checkimages_endings is None:
        sextractor_checkimages_endings = ['seg', 'ap', 'bkg', 'bkgrms', 'minibkg', 'minibkgrms',
                                          'minbkg', 'filt', 'obj', 'minobj']
    if sextractor_checkimages is None:
        sextractor_checkimages = ['SEGMENTATION', 'APERTURES', 'BACKGROUND', 'BACKGROUND_RMS', 'MINIBACKGROUND',
                                  'MINIBACK_RMS', '-BACKGROUND', 'FILTERED', 'OBJECTS', '-OBJECTS']

    logger.info('Creating SExtractor commands...')

    if not os.path.isabs(sextractor_config):
        sextractor_config = resource_filename(morphofit.__name__, "res/sextractor/default.sex")
    if not os.path.isabs(sextractor_params):
        sextractor_params = resource_filename(morphofit.__name__, "res/sextractor/default.param")
    if not os.path.isabs(sextractor_filter):
        sextractor_filter = resource_filename(morphofit.__name__, "res/sextractor/gauss_3.0_5x5.conv")
    if not os.path.isabs(sextractor_nnw):
        sextractor_nnw = resource_filename(morphofit.__name__, "res/sextractor/default.nnw")

    catalogue_name_short = sextractor_catalogue_name.rsplit(".", 1)[0]
    checkimages_names = []
    checkimages = sextractor_checkimages
    for suffix in sextractor_checkimages_endings:
        checkimages_names += [catalogue_name_short + '_{}.fits'.format(suffix)]
    if len(checkimages_names) == 0:
        checkimages_names = ['NONE']
        checkimages = ['NONE']

    cmd = [sextractor_binary,
           sci_image,
           "-c", sextractor_config,
           "-SEEING_FWHM", str(psf_fwhm),
           "-SATUR_KEY", "SATURATE",
           "-SATUR_LEVEL", str(saturation_level),
           "-MAG_ZEROPOINT", str(magzero),
           "-GAIN_KEY", "GAIN",
           "-GAIN", str(gain),
           "-PIXEL_SCALE", str(pixscale),
           "-STARNNW_NAME", sextractor_nnw,
           "-FILTER_NAME", sextractor_filter,
           "-PARAMETERS_NAME", sextractor_params,
           "-CATALOG_NAME", sextractor_catalogue_name,
           "-CHECKIMAGE_TYPE", ",".join(checkimages),
           "-CHECKIMAGE_NAME", ",".join(checkimages_names),
           "-CATALOG_TYPE", "FITS_1.0",
           "-VERBOSE_TYPE", "QUIET"]

    if rms_image is not None:
        cmd.append("-WEIGHT_TYPE")
        cmd.append("MAP_RMS")
        cmd.append("-WEIGHT_IMAGE")
        cmd.append(rms_image)

    cmd = cmd + photo_cmd

    return cmd


def get_sextractor_forced_cmd(sci_images, detection_image, detection_image_catalogue_name,
                              sextractor_catalogue_names, wavebands,
                              psf_fwhm_dict, saturation_level_dict, magzero_dict, gain_dict, pixscale,
                              photo_cmd, sextractor_binary='/usr/bin/sex',
                              sextractor_config='default.sex', sextractor_params='default.param',
                              sextractor_filter='gauss_3.0_5x5.conv', sextractor_nnw='default.nnw',
                              sextractor_checkimages=None,
                              sextractor_checkimages_endings=None, rms_images=None,
                              detection_rms_image=None):
    """
    This function creates a list of commands that are fed to subprocess to run SExtractor. Seeing value should be in
    arcseconds.

    :param sci_images:
    :param detection_image:
    :param detection_image_catalogue_name:
    :param sextractor_catalogue_names:
    :param wavebands:
    :param psf_fwhm_dict:
    :param saturation_level_dict:
    :param magzero_dict:
    :param gain_dict:
    :param pixscale:
    :param photo_cmd:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :param rms_images:
    :param detection_rms_image:
    :return cmd: list of commands for SExtractor binary.
    :return checkimages_names: names of SExtractor checkimages to generate.
    """

    if sextractor_checkimages_endings is None:
        sextractor_checkimages_endings = ['seg', 'ap', 'bkg', 'bkgrms', 'minibkg', 'minibkgrms',
                                          'minbkg', 'filt', 'obj', 'minobj']
    if sextractor_checkimages is None:
        sextractor_checkimages = ['SEGMENTATION', 'APERTURES', 'BACKGROUND', 'BACKGROUND_RMS', 'MINIBACKGROUND',
                                  'MINIBACK_RMS', '-BACKGROUND', 'FILTERED', 'OBJECTS', '-OBJECTS']

    logger.info('Creating SExtractor commands...')

    cmd = []

    if not os.path.isabs(sextractor_config):
        sextractor_config = resource_filename(morphofit.__name__, "res/sextractor/default.sex")
    if not os.path.isabs(sextractor_params):
        sextractor_params = resource_filename(morphofit.__name__, "res/sextractor/default.param")
    if not os.path.isabs(sextractor_filter):
        sextractor_filter = resource_filename(morphofit.__name__, "res/sextractor/gauss_3.0_5x5.conv")
    if not os.path.isabs(sextractor_nnw):
        sextractor_nnw = resource_filename(morphofit.__name__, "res/sextractor/default.nnw")

    h = fits.getheader(detection_image)
    checkimages_names = []
    for suffix in sextractor_checkimages_endings:
        checkimages_names += [detection_image[:-5] + '_{}.fits'.format(suffix)]

    # SExtractor commands for the detection image
    single_cmd = [sextractor_binary,
                  detection_image + ',' + detection_image,
                  "-c", sextractor_config,
                  "-SEEING_FWHM", str(h['SEEING']),
                  "-SATUR_KEY", "SATURATION",
                  "-SATUR_LEVEL", str(h['SATURATION']),
                  "-MAG_ZEROPOINT", str(h['MAGZPT']),
                  "-GAIN_KEY", "GAIN",
                  "-GAIN", str(h['GAIN']),
                  "-PIXEL_SCALE", str(h['PIXSCALE']),
                  "-STARNNW_NAME", sextractor_nnw,
                  "-FILTER_NAME", sextractor_filter,
                  "-PARAMETERS_NAME", sextractor_params,
                  "-CATALOG_NAME", detection_image_catalogue_name,  # detection_image[:-4] + 'forced.sexcat',
                  "-CHECKIMAGE_TYPE", ",".join(sextractor_checkimages),
                  "-CHECKIMAGE_NAME", ",".join(checkimages_names),
                  "-CATALOG_TYPE", "FITS_1.0",
                  "-VERBOSE_TYPE", "QUIET"]
    if detection_rms_image is not None:
        single_cmd.append("-WEIGHT_TYPE")
        single_cmd.append("MAP_RMS, MAP_RMS")
        single_cmd.append("-WEIGHT_IMAGE")
        single_cmd.append("{},{}".format(detection_rms_image, detection_rms_image))
    single_cmd = single_cmd + photo_cmd
    cmd.append(single_cmd)

    for i in range(len(wavebands)):
        label = wavebands[i]

        catalogue_name_short = sextractor_catalogue_names[i].rsplit(".", 1)[0]
        checkimages_names = []
        checkimages = sextractor_checkimages
        for suffix in sextractor_checkimages_endings:
            checkimages_names += [catalogue_name_short + '_{}.fits'.format(suffix)]
        if len(checkimages_names) == 0:
            checkimages_names = ['NONE']
            checkimages = ['NONE']
        single_cmd = [sextractor_binary,
                      detection_image + ',' + sci_images[i],
                      "-c", sextractor_config,
                      "-SEEING_FWHM", str(psf_fwhm_dict[label]),
                      "-SATUR_KEY", "SATURATE",
                      "-SATUR_LEVEL", str(saturation_level_dict[label]),
                      "-MAG_ZEROPOINT", str(magzero_dict[label]),
                      "-GAIN_KEY", "GAIN",
                      "-GAIN", str(gain_dict[label]),
                      "-PIXEL_SCALE", str(pixscale),
                      "-STARNNW_NAME", sextractor_nnw,
                      "-FILTER_NAME", sextractor_filter,
                      "-PARAMETERS_NAME", sextractor_params,
                      "-CATALOG_NAME", sextractor_catalogue_names[i],
                      "-CHECKIMAGE_TYPE", ",".join(checkimages),
                      "-CHECKIMAGE_NAME", ",".join(checkimages_names),
                      "-CATALOG_TYPE", "FITS_1.0",
                      "-VERBOSE_TYPE", "QUIET"]
        if rms_images is not None:
            single_cmd.append("-WEIGHT_TYPE")
            single_cmd.append("MAP_RMS, MAP_RMS")
            single_cmd.append("-WEIGHT_IMAGE")
            single_cmd.append("{},{}".format(detection_rms_image, rms_images[i]))
        single_cmd = single_cmd + photo_cmd
        cmd.append(single_cmd)

    return cmd


def run_sex_dual_mode(cmd):
    """
    This function runs SExtractor in the dual image mode, using as detection images the ones generated with Ufig.

    :param cmd: List of commands (as lists).
    :return: cat_name: output SExtractor catalogue name.
    """

    logger.info('Running SExtractor in double image mode on:')

    for i in range(len(cmd)):
        logger.info(cmd[i])
        subprocess.check_call(cmd[i], stderr=subprocess.STDOUT)


def run_sex_single_mode(cmd):
    """
    This function runs SExtractor in the single image mode.

    :param cmd:
    :return:
    """

    logger.info('Running SExtractor in single image mode on: \n %s' % os.path.basename(cmd[1]))
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)
