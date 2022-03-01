#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
from astropy.io import fits
import os
import numpy as np
from scipy.optimize import curve_fit
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.special import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# morphofit imports
from morphofit.utils import get_logger

logger = get_logger(__file__)


def plot_diagnostic_image(output_folder, image_filename, image, normalisation, color_map, plot_title):
    """

    :param output_folder:
    :param image_filename:
    :param image:
    :param normalisation:
    :param color_map:
    :param plot_title:
    :return:
    """

    plt.clf()
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

    plt.imshow(image, cmap=color_map, norm=normalisation, origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Pixel value', fontsize=10)
    plt.title(plot_title, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, image_filename))


def gaussian_function(x, mu, intensity, sigma):
    """

    :param x:
    :param mu:
    :param intensity:
    :param sigma:
    :return:
    """

    line = intensity * np.exp(-(x - mu)**2 / (2 * sigma**2))

    return line


def create_diagnostic_images(output_model_image_path, output_folder, color_map='jet'):
    """

    :param output_model_image_path:
    :param output_folder:
    :param color_map:
    :return:
    """

    source_image = fits.getdata(output_model_image_path, ext=1)
    model_image = fits.getdata(output_model_image_path, ext=2)
    residual_image = fits.getdata(output_model_image_path, ext=3)

    normalisation = matplotlib.colors.Normalize(vmin=min(source_image.ravel()), vmax=max(source_image.ravel()))

    plot_diagnostic_image(output_folder, 'target_galaxy_image.pdf',
                          source_image, normalisation, color_map, 'Original Image')
    plot_diagnostic_image(output_folder, 'target_galaxy_model.pdf',
                          model_image, normalisation, color_map, 'Model Image')
    plot_diagnostic_image(output_folder, 'target_galaxy_residual.pdf',
                          residual_image, normalisation, color_map, 'Residual Image')


def create_diagnostic_pixel_counts_histogram(output_model_image_path, output_folder):
    """

    :param output_model_image_path:
    :param output_folder:
    :return:
    """

    source_image = fits.getdata(output_model_image_path, ext=1)
    model_image = fits.getdata(output_model_image_path, ext=2)
    residual_image = fits.getdata(output_model_image_path, ext=3)

    plt.clf()
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.hist(source_image.ravel(), bins='auto', color='blue', density=True, label='Original Image', histtype='step')
    plt.hist(model_image.ravel(), bins='auto', color='green', density=True, alpha=0.5, label='Model Image',
             histtype='step')
    plt.hist(residual_image.ravel(), bins='auto', color='red', density=True, alpha=0.5, label='Residual Image',
             histtype='step')
    plt.xlabel('Pixel Values', fontsize=10)
    plt.ylabel('Number of Pixels', fontsize=10)
    combined = np.concatenate((source_image.ravel(), model_image.ravel(), residual_image.ravel()))
    plt.xlim(np.percentile(combined, 1), np.percentile(combined, 99))
    plt.yscale('log')
    plt.legend(loc='best', fontsize=10)
    plt.title('Pixel Count Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Pixel_count_histogram.pdf'))


def create_gaussian_fit_residual_image_counts(output_model_image_path, output_folder):
    """

    :param output_model_image_path:
    :param output_folder:
    :return:
    """

    residual_image = fits.getdata(output_model_image_path, ext=3)
    residual_hist = np.histogram(residual_image.ravel(), bins='auto', density=True)

    p0 = [0, 0, 0.1]
    popt, pcov = curve_fit(gaussian_function, residual_hist[1][:-1], residual_hist[0], p0=p0)

    plt.clf()
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

    plt.plot(residual_hist[1][:-1], residual_hist[0], color='blue', lw=2, label='Pixel counts')
    plt.plot(residual_hist[1][:-1], gaussian_function(residual_hist[1][:-1], popt[0], popt[1], popt[2]),
             color='red', lw=2, label='Gaussian fit')

    plt.vlines(popt[0], ymin=0, ymax=max(gaussian_function(residual_hist[1][:-1], popt[0], popt[1], popt[2])),
               color='red', lw=2, ls='dashed', label='Mean: {:1f}'.format(popt[0]))

    plt.xlabel('Pixel Values', fontsize=10)
    plt.ylabel('Number of Pixels', fontsize=10)
    plt.xlim(popt[0] - 10 * popt[2], popt[0] + 10 * popt[2])
    plt.legend(loc='upper left', fontsize=10)
    plt.title('Gaussian fit to Residual Image pixel counts', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Residual_image_pixel_count_histogram.pdf'))


def sersic_profile_mag_units(mu_e, kappa, r_arcsec, Re_arcsec, n):
    """

    :param mu_e:
    :param kappa:
    :param r_arcsec:
    :param Re_arcsec:
    :param n:
    :return:
    """

    return mu_e + 2.5 * (kappa / np.log(10)) * ((r_arcsec / Re_arcsec)**(1/n) - 1)


def sb_profile_from_sextractor_mag_aper(mag_aper1, mag_aper2, r1_arcsec, r2_arcsec):
    """

    :param mag_aper1:
    :param mag_aper2:
    :param r1_arcsec:
    :param r2_arcsec:
    :return:
    """

    flux_mag1 = 10 ** (mag_aper1 / (-2.5))
    flux_mag2 = 10 ** (mag_aper2 / (-2.5))

    return -2.5 * np.log10(flux_mag2 - flux_mag1) + 2.5 * np.log10(np.pi) + \
        2.5 * np.log10(r2_arcsec**2 - r1_arcsec**2)


def sb_profile_errors_from_sextractor_mag_aper(mag_aper1, mag_aper2, mag_aper1_err, mag_aper2_err):
    """

    :param mag_aper1:
    :param mag_aper2:
    :param mag_aper1_err:
    :param mag_aper2_err:
    :return:
    """

    flux_mag1 = 10 ** (mag_aper1 / (-2.5))
    flux_mag2 = 10 ** (mag_aper2 / (-2.5))

    return abs((1 / (flux_mag2 - flux_mag1)) * np.sqrt((flux_mag1 * mag_aper1_err) ** 2.0 +
                                                       (flux_mag2 * mag_aper2_err) ** 2.0))


def create_best_fitting_photometry_comparison(best_fitting_galaxy_catalogue_filename, source_galaxies_catalogue,
                                              waveband, pixel_scale, ra_key, dec_key, galaxy_id_key,
                                              component_number_key, light_profile_key, phot_apertures, output_folder):
    """

    :param best_fitting_galaxy_catalogue_filename:
    :param source_galaxies_catalogue:
    :param waveband:
    :param pixel_scale:
    :param ra_key:
    :param dec_key:
    :param galaxy_id_key:
    :param component_number_key:
    :param light_profile_key:
    :param phot_apertures
    :param output_folder:
    :return:
    """

    best_fitting_galaxy_catalogue = Table.read(best_fitting_galaxy_catalogue_filename, format='fits')
    galaxy_id = best_fitting_galaxy_catalogue[galaxy_id_key]
    ra = best_fitting_galaxy_catalogue['RA_GALFIT']
    dec = best_fitting_galaxy_catalogue['DEC_GALFIT']
    light_profile = best_fitting_galaxy_catalogue[light_profile_key]
    mag = best_fitting_galaxy_catalogue['MAG_GALFIT']
    re = best_fitting_galaxy_catalogue['RE_GALFIT']
    n = best_fitting_galaxy_catalogue['N_GALFIT']
    component_number = best_fitting_galaxy_catalogue[component_number_key]

    sextractor_source_properties = []
    for i in range(len(ra)):
        catalog = SkyCoord(ra=np.array([ra[i]]) * u.degree,
                           dec=np.array([dec[i]]) * u.degree)
        c = SkyCoord(ra=source_galaxies_catalogue[ra_key],
                     dec=source_galaxies_catalogue[dec_key])
        idx, d2d, d3d = catalog.match_to_catalog_sky(c)
        sextractor_source_properties.append(source_galaxies_catalogue[idx])
    sextractor_source_properties = vstack(sextractor_source_properties)

    kappa = 1.9992 * n - 0.3271

    mu_e = mag + 5 * np.log10(re * pixel_scale) + 2.5 * \
        np.log10(2 * np.pi * n * (np.exp(kappa) / kappa ** (2 * n)) * gamma(2 * n))

    r_aper = np.array([float(name) for name in phot_apertures.split(',')]) / 2
    mag_aper = sextractor_source_properties['MAG_APER_{}'.format(waveband)]
    mag_aper_err = sextractor_source_properties['MAGERR_APER_{}'.format(waveband)]

    for i in range(len(mu_e)):

        mu_from_mag_aper = np.empty_like(r_aper)
        mu_err_from_mag_aper = np.empty_like(r_aper)
        mu_from_mag_aper[0] = mag_aper[i, 0] + 2.5 * np.log10(2 * np.pi) + 5 * np.log10(r_aper[0] * pixel_scale)
        mu_err_from_mag_aper[0] = mag_aper_err[i, 0]
        for j in range(1, len(r_aper)):
            mu_from_mag_aper[j] = sb_profile_from_sextractor_mag_aper(mag_aper[i, j - 1],
                                                                      mag_aper[i, j],
                                                                      r_aper[j - 1] * pixel_scale,
                                                                      r_aper[j] * pixel_scale)
            mu_err_from_mag_aper[j] = sb_profile_errors_from_sextractor_mag_aper(mag_aper[i, j],
                                                                                 mag_aper[i, j - 1],
                                                                                 mag_aper_err[i, j],
                                                                                 mag_aper_err[i, j - 1])

        plt.clf()
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        r_arcsec = np.arange(r_aper[0] * pixel_scale, r_aper[-1] * pixel_scale, 0.01)
        if light_profile[i] == 'sersic':
            plt.plot(r_arcsec, sersic_profile_mag_units(mu_e[i], kappa[i], r_arcsec, re[i] * pixel_scale, n[i]),
                     lw=2,
                     color='black', label='Best-fitting Sersic profile')
        elif light_profile[i] == 'devauc':
            plt.plot(r_arcsec, sersic_profile_mag_units(mu_e[i], 7.669, r_arcsec, re[i] * pixel_scale, 4),
                     lw=2, color='black', label='Best-fitting de Vaucouleurs profile')
        elif light_profile[i] == 'expdisk':
            plt.plot(r_arcsec, sersic_profile_mag_units(mu_e[i], 1.678, r_arcsec, re[i] * pixel_scale, 1),
                     lw=2, color='black', label='Best-fitting Exponential disk profile')
        else:
            raise ValueError
        plt.errorbar(r_aper * pixel_scale, mu_from_mag_aper, yerr=mu_err_from_mag_aper,
                     fmt='o', capthick=2, elinewidth=2, mec='black', color='red',
                     label='SExtractor aperture photometry')
        plt.xlabel('r [arcsec]', fontsize=10)
        plt.ylabel(r'SB [mag arcsec$^{-2}$]', fontsize=10)
        plt.gca().invert_yaxis()
        plt.legend(loc='best', fontsize=10)
        plt.title('Comparison light profile ID:{} with SExtractor photometry'.format(galaxy_id[i]),
                  fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,
                                 'ID{}_component{}_{}_best_fit_model_vs_sextractor_photometry.pdf'
                                 .format(galaxy_id[i], component_number[i], waveband)))
