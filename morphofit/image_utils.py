#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.ndimage import gaussian_filter
import os
from astropy import wcs

# morphofit imports
from morphofit.utils import get_logger

logger = get_logger(__file__)


def create_detection_image(img_names, detect_name, wavebands, psf_fwhm, magnitude_zeropoints,
                           saturation_parameters, gains_parameters, pixelscale_parameters,
                           background_sigma_parameters):
    """
    This function creates the detection image used by Source Extractor to perform forced photometry.
    The different images are divided by their rms background and then summed following the prescription
    in Coe et al 2006.

    :param img_names:
    :param detect_name:
    :param wavebands:
    :param psf_fwhm:
    :param magnitude_zeropoints:
    :param saturation_parameters:
    :param gains_parameters:
    :param pixelscale_parameters:
    :param background_sigma_parameters:
    :return:
    """

    detection_img = np.zeros(np.shape(fits.getdata(img_names[0])))
    detection_img_header = fits.getheader(img_names[0])

    detection_img_header['SEEING'] = max([value for value in psf_fwhm.items()])[1]
    detection_img_header['MAGZPT'] = max([value for value in magnitude_zeropoints.items()])[1]
    detection_img_header['SATURATION'] = max([value for value in saturation_parameters.items()])[1]
    detection_img_header['GAIN'] = max([value for value in gains_parameters.items()])[1]
    detection_img_header['PIXSCALE'] = pixelscale_parameters
    # max([value for value in pixelscale_parameters.items()])[1]

    for name in img_names:
        idx_name = img_names.index(name)
        data = fits.getdata(name) / background_sigma_parameters[wavebands[idx_name]]
        detection_img = detection_img + data
    fits.writeto(detect_name, detection_img, detection_img_header,
                 overwrite=True)


def create_rms_detection_image(rms_images, rms_detect_name, wavebands, background_sigma_parameters):
    """
    This function creates the detection rms image used by Source Extractor to perform forced photometry.
    The different images are divided by their rms background and then summed following the prescription
    in Coe et al 2006.

    :param rms_images:
    :param rms_detect_name:
    :param wavebands:
    :param background_sigma_parameters:
    :return:
    """

    detection_rms_img = np.zeros(np.shape(fits.getdata(rms_images[0])))
    detection_rms_img_header = fits.getheader(rms_images[0])

    for name in rms_images:
        idx_name = rms_images.index(name)
        data = fits.getdata(name)
        mask = (data > 10 ** 9)
        data[mask] = np.nanmedian(data[~mask])
        data = data ** 2 / background_sigma_parameters[wavebands[idx_name]] ** 2
        # data = fits.getdata(name)**2 / background_sigma_parameters[band]**2
        detection_rms_img = detection_rms_img + data
    detection_rms_img = np.sqrt(detection_rms_img)
    mask = (detection_rms_img == 0)
    # detection_rms_img[mask] = 1 * 10 ** 10
    detection_rms_img[mask] = np.nanmedian(detection_rms_img[~mask])
    fits.writeto(rms_detect_name, detection_rms_img, detection_rms_img_header, overwrite=True)


def create_rms_detection_image_clash(wht_images, rms_detect_name, background_sigma_parameters):
    """
    This function creates the detection rms image for CLASH images used by Source Extractor to perform
    forced photometry.
    The different images are divided by their rms background and then summed following the prescription
    in Coe et al 2006.

    :param wht_images:
    :param rms_detect_name:
    :param background_sigma_parameters:
    :return:
    """

    detection_rms_img = np.zeros(np.shape(fits.getdata(wht_images[0])))
    detection_rms_img_header = fits.getheader(wht_images[0])

    for name in wht_images:
        # idx_name = wht_images.index(name)
        band = name.split('_')[-3]
        data = fits.getdata(name)
        mask = (data > 0)
        data[mask] = np.inf
        data = (1 / data) * (1 / background_sigma_parameters[band] ** 2)
        # data = fits.getdata(name)**2 / background_sigma_parameters[band]**2
        detection_rms_img = detection_rms_img + data
    detection_rms_img = np.sqrt(detection_rms_img)
    mask = (detection_rms_img == 0)
    detection_rms_img[mask] = 1 * 10 ** 10
    fits.writeto(rms_detect_name, detection_rms_img, detection_rms_img_header, overwrite=True)


def create_cutout(image, header, x_source, y_source, effective_radius_source, enlarging_factor, axis_ratio, angle):
    """
    The cutout size is based on the enlarged elliptical size of the galaxy.

    :param image:
    :param x_source:
    :param header:
    :param y_source:
    :param effective_radius_source:
    :param enlarging_factor:
    :param axis_ratio:
    :param angle:
    :return
    """

    source_pixel_position = (x_source, y_source)
    size_cutout_x = effective_radius_source * enlarging_factor * (abs(np.cos(angle)) + axis_ratio * abs(np.sin(angle)))
    size_cutout_y = effective_radius_source * enlarging_factor * (abs(np.sin(angle)) + axis_ratio * abs(np.cos(angle)))
    cutout_image_size = (size_cutout_y, size_cutout_x)
    w = wcs.WCS(header)
    cutout_image = Cutout2D(image, source_pixel_position, cutout_image_size, wcs=w, copy=True)
    cutout_header = header.copy()
    cutout_image_wcs = cutout_image.wcs
    updated_keywords = cutout_image_wcs.to_header()
    for key in updated_keywords:
        cutout_header[key] = updated_keywords[key]

    return cutout_image, cutout_header


def save_stamp(index_stamp, image_filename, cutout_image, cutout_header):
    """

    :param index_stamp:
    :param image_filename:
    :param cutout_image:
    :param cutout_header:
    :return
    """

    image_stamp_filename = os.path.splitext(image_filename)[0] + '_stamp{}.fits'.format(index_stamp)
    fits.writeto(image_stamp_filename, cutout_image, cutout_header, overwrite=True)

    return image_stamp_filename


def cut_stamp(index_stamp, sci_image_filename, rms_image_filename, seg_image_filename, exp_image_filename,
              x_source, y_source, effective_radius_source, enlarging_factor, minor_axis_source,
              major_axis_source, position_angle_source):
    """
    This function cuts image stamps. The center corresponds to the spectroscopic confirmed source and the
    size of the image is based on the enlarged elliptical size of the object.

    :param index_stamp:
    :param sci_image_filename:
    :param rms_image_filename:
    :param seg_image_filename:
    :param exp_image_filename:
    :param x_source:
    :param y_source:
    :param effective_radius_source:
    :param enlarging_factor:
    :param minor_axis_source:
    :param major_axis_source:
    :param position_angle_source:
    :return
    """

    if minor_axis_source == major_axis_source:
        axis_ratio = minor_axis_source
    else:
        axis_ratio = minor_axis_source / major_axis_source
    angle = position_angle_source * (2*np.pi) / 360

    sci_image, sci_header = fits.getdata(sci_image_filename, header=True)
    seg_image, seg_header = fits.getdata(seg_image_filename, header=True)

    sci_stamp_cutout, sci_stamp_cutout_header = create_cutout(sci_image, sci_header, x_source, y_source,
                                                              effective_radius_source, enlarging_factor,
                                                              axis_ratio, angle)
    seg_stamp_cutout, seg_stamp_cutout_header = create_cutout(seg_image, seg_header, x_source, y_source,
                                                              effective_radius_source, enlarging_factor,
                                                              axis_ratio, angle)

    sci_image_stamp_filename = save_stamp(index_stamp, sci_image_filename, sci_stamp_cutout.data,
                                          sci_stamp_cutout_header)
    seg_image_stamp_filename = save_stamp(index_stamp, seg_image_filename, seg_stamp_cutout.data,
                                          seg_stamp_cutout_header)

    if rms_image_filename is not None:
        rms_image, rms_header = fits.getdata(rms_image_filename, header=True)
        rms_stamp_cutout, rms_stamp_cutout_header = create_cutout(rms_image, rms_header, x_source, y_source,
                                                                  effective_radius_source, enlarging_factor,
                                                                  axis_ratio, angle)
        rms_image_stamp_filename = save_stamp(index_stamp, rms_image_filename, rms_stamp_cutout.data,
                                              rms_stamp_cutout_header)
    else:
        rms_image_stamp_filename = None
    if exp_image_filename is not None:
        exp_image, exp_header = fits.getdata(exp_image_filename, header=True)
        exp_stamp_cutout, exp_stamp_cutout_header = create_cutout(exp_image, exp_header, x_source, y_source,
                                                                  effective_radius_source, enlarging_factor,
                                                                  axis_ratio, angle)
        exp_image_stamp_filename = save_stamp(index_stamp, exp_image_filename, exp_stamp_cutout.data,
                                              exp_stamp_cutout_header)
    else:
        exp_image_stamp_filename = None

    return sci_image_stamp_filename, rms_image_stamp_filename, seg_image_stamp_filename, exp_image_stamp_filename


def create_bad_pixel_mask_for_stamps(sextractor_id_source, neighbouring_sources_catalogue,
                                     id_key_neighbouring_sources_catalogue, seg_image_filename):
    """
    This function creates the bad pixel mask for GALFIT. The segmentation image has size scaled by the
    enlarging_image_factor, while the neighbouring sources are those within a range scaled by the
    enlarging_separation_factor.

    :param sextractor_id_source:
    :param neighbouring_sources_catalogue
    :param id_key_neighbouring_sources_catalogue:
    :param seg_image_filename:
    :return bad_pixel_mask_name: bad pixel mask path.
    """

    seg_image, seg_head = fits.getdata(seg_image_filename, header=True)
    mask_source = np.where(seg_image == sextractor_id_source)
    seg_image[mask_source] = 0

    for i in range(len(neighbouring_sources_catalogue)):
        mask_neighbouring_galaxies = np.where(seg_image ==
                                              neighbouring_sources_catalogue[id_key_neighbouring_sources_catalogue][i])
        seg_image[mask_neighbouring_galaxies] = 0

    bad_pixel_mask_filename = seg_image_filename[:-5] + '_badpixel.fits'
    fits.writeto(bad_pixel_mask_filename, seg_image, seg_head, overwrite=True)

    return bad_pixel_mask_filename


def create_custom_sigma_image(telescope_name, sigma_image_filename, sci_image_filename, rms_image_filename,
                              exp_image_filename, background_value, exposure_time):
    """

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
    mask_low_values_rms_image = np.where(rms_image < 10**5)
    mask_high_values_rms_image = np.where(rms_image > 10 ** 5)
    rms_image[mask_high_values_rms_image] = np.median(rms_image[mask_low_values_rms_image])
    if exp_image_filename:
        exp_image = fits.getdata(exp_image_filename)
        exp_image_zeros = np.where(exp_image == 0)
        exp_image_notzeros = np.where(exp_image != 0)
        exp_image[exp_image_zeros] = np.median(exp_image[exp_image_notzeros])
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
    # mask_high_values_rms_image = np.where(rms_image > 10 ** 5)
    # smoothed_sigma_image[mask_high_values_rms_image] = np.nan
    # smoothed_sigma_image[mask_high_values_rms_image] = np.nanmedian(smoothed_sigma_image)
    fits.writeto(sigma_image_filename, smoothed_sigma_image, sci_header, overwrite=True)
    fits.writeto(sci_image_filename, sci_image_original, sci_header, overwrite=True)


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
    if telescope_name == 'HST':
        sci_image_adu = (sci_image / instrumental_gain) * exposure_time
        sci_header['BUNIT'] = 'ADU'
        sci_header['EXPTIME'] = exposure_time
        sci_header['GAIN'] = np.mean([sci_header['ATODGNA'], sci_header['ATODGNB'], sci_header['ATODGNC'],
                                      sci_header['ATODGND']])
        # sci_header['NCOMBINE'] = 1 # set to 1 only if NCOMBINE is already factorized in GAIN
    else:
        if exp_image_filename is not None:
            exposure_time_image = fits.getdata(exp_image_filename)
            sci_image_adu = (sci_image / instrumental_gain) * exposure_time_image
        else:
            sci_image_adu = (sci_image / instrumental_gain) * exposure_time

    fits.writeto(sci_image_filename, sci_image_adu, sci_header, overwrite=True)
    magnitude_zeropoint = magnitude_zeropoint - 2.5 * np.log10(instrumental_gain)
    # magnitude_zeropoint = magnitude_zeropoint  + 2.5 * np.log10(instrumental_gain) - 2.5 * np.log10(exposure_time)
    background_value = (background_value / instrumental_gain) * exposure_time

    return magnitude_zeropoint, background_value


def create_sigma_image_for_galfit(telescope_name, sigma_image_filename, sci_image_filename, rms_image_filename,
                                  exp_image_filename, sigma_image_type, background_value, exposure_time,
                                  magnitude_zeropoint, instrumental_gain):
    """
    This function creates the sigma image. 'sigma_custom' refers to the sigma image created by us, 'sigma_int_gen'
    refers to the one internally generated by GALFIT. In the latter case we change the units of the image to ADU.

    :param telescope_name:
    :param sigma_image_filename:
    :param sci_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param sigma_image_type:
    :param background_value:
    :param exposure_time:
    :param magnitude_zeropoint:
    :param instrumental_gain:
    :return:
    """

    if sigma_image_type == 'custom_sigma_image':
        create_custom_sigma_image(telescope_name, sigma_image_filename, sci_image_filename, rms_image_filename,
                                  exp_image_filename, background_value, exposure_time)
    elif sigma_image_type == 'internal_generated_sigma_image':
        magnitude_zeropoint, background_value = create_internal_generated_sigma_image(telescope_name,
                                                                                      sci_image_filename,
                                                                                      exp_image_filename,
                                                                                      exposure_time, instrumental_gain,
                                                                                      magnitude_zeropoint,
                                                                                      background_value)
    else:
        logger.info('not implemented')
        raise ValueError

    return magnitude_zeropoint, background_value


def size_based_crop(image_filename, size_range_x, size_range_y, crop_suffix, output_directory):
    """

    :param image_filename:
    :param size_range_x:
    :param size_range_y:
    :param crop_suffix:
    :param output_directory:
    :return:
    """

    image, image_header = fits.getdata(image_filename, header=True)
    if (image_header['NAXIS1'] == int(size_range_x[1] - size_range_x[0])) & (image_header['NAXIS2'] ==
                                                                             int(size_range_y[1] - size_range_y[0])):
        pass
    else:
        central_pixel_position = ((size_range_x[0] + size_range_x[1]) / 2, (size_range_y[0] + size_range_y[1]) / 2)
        size_crop_x = size_range_x[1] - size_range_x[0]
        size_crop_y = size_range_y[1] - size_range_y[0]
        crop_image_size = (size_crop_y, size_crop_x)
        w = wcs.WCS(image_header)
        cropped_image = Cutout2D(image, central_pixel_position, crop_image_size, wcs=w, copy=True)
        cropped_image_header = image_header.copy()
        cropped_image_wcs = cropped_image.wcs
        updated_keywords = cropped_image_wcs.to_header()
        for key in updated_keywords:
            cropped_image_header[key] = updated_keywords[key]
        image_filename = os.path.splitext(os.path.basename(image_filename))[0] + '_{}.fits'.format(crop_suffix)
        fits.writeto(os.path.join(output_directory, image_filename), cropped_image.data, cropped_image_header,
                     overwrite=True)

    return output_directory + image_filename


def catalogue_based_crop(image_filename, external_catalogue, crop_suffix, x_keyword, y_keyword, output_directory):
    """

    :param image_filename:
    :param external_catalogue:
    :param crop_suffix:
    :param x_keyword:
    :param y_keyword:
    :param output_directory:
    :return:
    """

    image, image_header = fits.getdata(image_filename, header=True)
    x = external_catalogue[x_keyword]
    y = external_catalogue[y_keyword]
    enlarge = 100  # pixels
    x_max = max(x) + enlarge
    x_min = min(x) - enlarge
    y_max = max(y) + enlarge
    y_min = min(y) - enlarge

    if x_max > image_header['NAXIS1']:
        x_max = image_header['NAXIS1']
    if y_max > image_header['NAXIS2']:
        y_max = image_header['NAXIS2']
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0

    central_pixel_position = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    size_crop_x = x_max - x_min
    size_crop_y = y_max - y_min
    crop_image_size = (size_crop_y, size_crop_x)
    w = wcs.WCS(image_header)
    cropped_image = Cutout2D(image, central_pixel_position, crop_image_size, wcs=w, copy=True)
    cropped_image_header = image_header.copy()
    cropped_image_wcs = cropped_image.wcs
    updated_keywords = cropped_image_wcs.to_header()
    for key in updated_keywords:
        cropped_image_header[key] = updated_keywords[key]

    image_filename = os.path.splitext(os.path.basename(image_filename))[0] + '_{}'.format(crop_suffix)

    fits.writeto(os.path.join(output_directory, image_filename), cropped_image.data, cropped_image_header,
                 overwrite=True)

    return os.path.join(output_directory, image_filename)


def crop_routine_size_based(sci_image_filename, seg_image_filename, rms_image_filename, exp_image_filename,
                            size_range_x, size_range_y, crop_suffix, output_directory,
                            cropped_sci_image_filenames, cropped_seg_image_filenames, cropped_rms_image_filenames,
                            cropped_exp_image_filenames):
    """

    :param sci_image_filename:
    :param seg_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param size_range_x:
    :param size_range_y:
    :param crop_suffix:
    :param output_directory:
    :param cropped_sci_image_filenames:
    :param cropped_seg_image_filenames:
    :param cropped_rms_image_filenames:
    :param cropped_exp_image_filenames:
    :return:
    """

    cropped_sci_image_filename = size_based_crop(sci_image_filename, size_range_x, size_range_y,
                                                 crop_suffix, output_directory)
    cropped_sci_image_filenames.append(cropped_sci_image_filename)
    cropped_seg_image_filename = size_based_crop(seg_image_filename, size_range_x, size_range_y,
                                                 crop_suffix, output_directory)
    cropped_seg_image_filenames.append(cropped_seg_image_filename)
    try:
        cropped_rms_image_filename = size_based_crop(rms_image_filename, size_range_x, size_range_y,
                                                     crop_suffix, output_directory)
        cropped_rms_image_filenames.append(cropped_rms_image_filename)
    except Exception as e:
        logger.info(e)
        logger.info('Missing rms image')
    try:
        cropped_exp_image_filename = size_based_crop(exp_image_filename, size_range_x, size_range_y,
                                                     crop_suffix, output_directory)
        cropped_exp_image_filenames.append(cropped_exp_image_filename)
    except Exception as e:
        logger.info(e)
        logger.info('Missing exp image')


def crop_routine_catalogue_based(sci_image_filename, seg_image_filename, rms_image_filename, exp_image_filename,
                                 external_catalogue, x_keyword, y_keyword, crop_suffix, output_directory,
                                 cropped_sci_image_filenames, cropped_seg_image_filenames, cropped_rms_image_filenames,
                                 cropped_exp_image_filenames):
    """

    :param sci_image_filename:
    :param seg_image_filename:
    :param rms_image_filename:
    :param exp_image_filename:
    :param external_catalogue:
    :param x_keyword:
    :param y_keyword:
    :param crop_suffix:
    :param output_directory:
    :param cropped_sci_image_filenames:
    :param cropped_seg_image_filenames:
    :param cropped_rms_image_filenames:
    :param cropped_exp_image_filenames:
    :return:
    """

    cropped_sci_image_filename = catalogue_based_crop(sci_image_filename, external_catalogue,
                                                      crop_suffix, x_keyword, y_keyword,
                                                      output_directory)
    cropped_sci_image_filenames.append(cropped_sci_image_filename)
    cropped_seg_image_filename = catalogue_based_crop(seg_image_filename, external_catalogue,
                                                      crop_suffix, x_keyword, y_keyword,
                                                      output_directory)
    cropped_seg_image_filenames.append(cropped_seg_image_filename)
    try:
        cropped_rms_image_filename = catalogue_based_crop(rms_image_filename, external_catalogue,
                                                          crop_suffix, x_keyword, y_keyword,
                                                          output_directory)
        cropped_rms_image_filenames.append(cropped_rms_image_filename)
    except Exception as e:
        logger.info(e)
        logger.info('Missing rms image')
    try:
        cropped_exp_image_filename = catalogue_based_crop(exp_image_filename, external_catalogue,
                                                          crop_suffix, x_keyword, y_keyword,
                                                          output_directory)
        cropped_exp_image_filenames.append(cropped_exp_image_filename)
    except Exception as e:
        logger.info(e)
        logger.info('Missing rms image')


def crop_images_deprecated(sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames,
                           wavebands, output_directory, crop_routine='size_based', external_catalogue=None,
                           size_range_x=None, size_range_y=None, crop_suffix='cropped',
                           x_keyword='XWIN_IMAGE_f814w', y_keyword='YWIN_IMAGE_f814w'):
    """

    :param sci_image_filenames:
    :param rms_image_filenames:
    :param seg_image_filenames:
    :param exp_image_filenames:
    :param wavebands:
    :param output_directory:
    :param crop_routine:
    :param external_catalogue:
    :param size_range_x:
    :param size_range_y:
    :param crop_suffix:
    :param x_keyword:
    :param y_keyword:
    :return:
    """

    if external_catalogue is None:
        external_catalogue = ''
    if (size_range_x is None) | (size_range_y is None):
        size_range_x = [0, 1000]
        size_range_y = [0, 1000]

    cropped_sci_image_filenames, cropped_rms_image_filenames = [], []
    cropped_seg_image_filenames, cropped_exp_image_filenames = [], []

    for i in range(len(wavebands)):
        if crop_routine == 'size_based':
            crop_routine_size_based(sci_image_filenames[i], seg_image_filenames[i],
                                    rms_image_filenames[i], exp_image_filenames[i],
                                    size_range_x, size_range_y,
                                    crop_suffix, output_directory,
                                    cropped_sci_image_filenames,
                                    cropped_seg_image_filenames,
                                    cropped_rms_image_filenames,
                                    cropped_exp_image_filenames)
        elif crop_routine == 'catalogue_based':
            crop_routine_catalogue_based(sci_image_filenames[i],
                                         seg_image_filenames[i],
                                         rms_image_filenames[i],
                                         exp_image_filenames[i],
                                         external_catalogue, x_keyword,
                                         y_keyword, crop_suffix,
                                         output_directory,
                                         cropped_sci_image_filenames,
                                         cropped_seg_image_filenames,
                                         cropped_rms_image_filenames,
                                         cropped_exp_image_filenames)
        else:
            logger.info('Not implemented')
            raise ValueError

    return cropped_sci_image_filenames, cropped_rms_image_filenames, cropped_seg_image_filenames, \
        cropped_exp_image_filenames


def crop_images(sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames,
                wavebands, output_directory, external_catalogue=None, crop_suffix='cropped',
                x_keyword='XWIN_IMAGE_f814w', y_keyword='YWIN_IMAGE_f814w'):
    """

    :param sci_image_filenames:
    :param rms_image_filenames:
    :param seg_image_filenames:
    :param exp_image_filenames:
    :param wavebands:
    :param output_directory:
    :param external_catalogue:
    :param crop_suffix:
    :param x_keyword:
    :param y_keyword:
    :return:
    """

    cropped_sci_image_filenames, cropped_rms_image_filenames = [], []
    cropped_seg_image_filenames, cropped_exp_image_filenames = [], []

    for i in range(len(wavebands)):
        crop_routine_catalogue_based(sci_image_filenames[i],
                                     seg_image_filenames[i],
                                     rms_image_filenames[i],
                                     exp_image_filenames[i],
                                     external_catalogue, x_keyword,
                                     y_keyword, crop_suffix,
                                     output_directory,
                                     cropped_sci_image_filenames,
                                     cropped_seg_image_filenames,
                                     cropped_rms_image_filenames,
                                     cropped_exp_image_filenames)

    return cropped_sci_image_filenames, cropped_rms_image_filenames, cropped_seg_image_filenames, \
        cropped_exp_image_filenames


def cut_regions(image_filename, number_of_regions_perside, region_image_suffix, output_directory):
    """

    :param image_filename:
    :param number_of_regions_perside:
    :param region_image_suffix:
    :param output_directory:
    :return:
    """

    region_filenames = []

    image, image_header = fits.getdata(image_filename, header=True)
    size_region = image_header['NAXIS1'] / number_of_regions_perside
    w = wcs.WCS(image_header)
    for i in range(0, number_of_regions_perside):
        for j in range(0, number_of_regions_perside):
            central_x_coordinate = (2 * j * size_region + size_region) / 2
            central_y_coordinate = (2 * i * size_region + size_region) / 2
            central_pixel_position = (central_x_coordinate, central_y_coordinate)
            region_image_size = (size_region, size_region)
            region_image = Cutout2D(image, central_pixel_position, region_image_size, wcs=w, copy=True)
            region_image_header = image_header.copy()
            region_image_wcs = region_image.wcs
            updated_keywords = region_image_wcs.to_header()
            for key in updated_keywords:
                region_image_header[key] = updated_keywords[key]
            region_filename = os.path.splitext(os.path.basename(image_filename))[0] + \
                '_{}{}{}.fits'.format(region_image_suffix, i, j)
            region_filenames.append(os.path.join(output_directory, region_filename))
            fits.writeto(os.path.join(output_directory, region_filename), region_image.data,
                         region_image_header, overwrite=True)

    return region_filenames


def create_regions(sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames,
                   wavebands, region_image_suffix, number_of_regions_perside, output_directory):
    """

    :param sci_image_filenames:
    :param rms_image_filenames:
    :param seg_image_filenames:
    :param exp_image_filenames:
    :param wavebands:
    :param region_image_suffix:
    :param number_of_regions_perside:
    :param output_directory:
    :return:
    """

    sci_image_region_filenames = []
    seg_image_region_filenames = []
    rms_image_region_filenames = []
    exp_image_region_filenames = []

    for i in range(len(wavebands)):
        sci_region_filenames = cut_regions(sci_image_filenames[i], number_of_regions_perside, region_image_suffix,
                                           output_directory)
        seg_region_filenames = cut_regions(seg_image_filenames[i], number_of_regions_perside, region_image_suffix,
                                           output_directory)
        sci_image_region_filenames.extend(sci_region_filenames)
        seg_image_region_filenames.extend(seg_region_filenames)
        try:
            rms_region_filenames = cut_regions(rms_image_filenames[i], number_of_regions_perside, region_image_suffix,
                                               output_directory)
            rms_image_region_filenames.extend(rms_region_filenames)
        except Exception as e:
            logger.info(e)
            logger.info('Missing rms image')
        try:
            exp_region_filenames = cut_regions(exp_image_filenames[i], number_of_regions_perside, region_image_suffix,
                                               output_directory)
            exp_image_region_filenames.extend(exp_region_filenames)
        except Exception as e:
            logger.info(e)
            logger.info('Missing exp image')

    return sci_image_region_filenames, seg_image_region_filenames, rms_image_region_filenames, \
        exp_image_region_filenames


def create_bad_pixel_mask_for_galfit(source_catalogue, seg_image_filename, id_key_sources_catalogue='NUMBER'):
    """

    :param source_catalogue:
    :param seg_image_filename:
    :param id_key_sources_catalogue:
    :return:
    """

    seg_image, seg_header = fits.getdata(seg_image_filename, header=True)
    for i in range(len(source_catalogue)):
        w = np.where(seg_image == source_catalogue[id_key_sources_catalogue][i])
        seg_image[w] = 0
    bad_pixel_mask_filename = seg_image_filename[:-5] + '_badpixel.fits'
    fits.writeto(bad_pixel_mask_filename, seg_image, seg_header, overwrite=True)

    return bad_pixel_mask_filename
