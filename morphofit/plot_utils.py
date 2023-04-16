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


def plot_diagnostic_image(output_folder, image_filename, image, normalisation, color_map, plot_title,
                          figure_size=(10, 8), xtick_labelsize=15, ytick_labelsize=15,
                          xlabel_fontsize=20, ylabel_fontsize=20, cbar_fontsize=20, title_fontsize=20):
    """

    :param output_folder:
    :param image_filename:
    :param image:
    :param normalisation:
    :param color_map:
    :param plot_title:
    :param figure_size:
    :param xtick_labelsize:
    :param ytick_labelsize:
    :param xlabel_fontsize:
    :param ylabel_fontsize:
    :param cbar_fontsize:
    :param title_fontsize:
    :return:
    """

    plt.clf()
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['xtick.labelsize'] = xtick_labelsize
    plt.rcParams['ytick.labelsize'] = ytick_labelsize

    plt.imshow(image, cmap=color_map, norm=normalisation, origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Pixel value', fontsize=cbar_fontsize)
    plt.title(plot_title, fontsize=title_fontsize)
    plt.xlabel('x [pixel]', fontsize=xlabel_fontsize)
    plt.ylabel('y [pixel]', fontsize=ylabel_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, image_filename))
    plt.close()


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


def create_diagnostic_images(output_model_image_path, output_folder, waveband, color_map='jet',
                             figure_size=(10, 8), xtick_labelsize=15, ytick_labelsize=15,
                             xlabel_fontsize=20, ylabel_fontsize=20, cbar_fontsize=20, title_fontsize=20):
    """

    :param output_model_image_path:
    :param output_folder:
    :param waveband:
    :param color_map:
    :param figure_size:
    :param xtick_labelsize:
    :param ytick_labelsize:
    :param xlabel_fontsize:
    :param ylabel_fontsize:
    :param cbar_fontsize:
    :param title_fontsize:
    :return:
    """

    source_image = fits.getdata(output_model_image_path, ext=1)
    model_image = fits.getdata(output_model_image_path, ext=2)
    residual_image = fits.getdata(output_model_image_path, ext=3)

    normalisation = matplotlib.colors.Normalize(vmin=(np.mean(source_image.ravel()) - np.std(source_image.ravel())),
                                                vmax=(np.mean(source_image.ravel()) + np.std(source_image.ravel())))

    plot_diagnostic_image(output_folder, 'target_galaxy_image_{}.pdf'.format(waveband),
                          source_image, normalisation, color_map, 'Original Image, {}'.format(waveband),
                          figure_size=figure_size, xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                          xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize, cbar_fontsize=cbar_fontsize,
                          title_fontsize=title_fontsize)
    plot_diagnostic_image(output_folder, 'target_galaxy_model_{}.pdf'.format(waveband),
                          model_image, normalisation, color_map, 'Model Image, {}'.format(waveband),
                          figure_size=figure_size, xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                          xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize, cbar_fontsize=cbar_fontsize,
                          title_fontsize=title_fontsize)
    plot_diagnostic_image(output_folder, 'target_galaxy_residual_{}.pdf'.format(waveband),
                          residual_image, normalisation, color_map, 'Residual Image, {}'.format(waveband),
                          figure_size=figure_size, xtick_labelsize=xtick_labelsize, ytick_labelsize=ytick_labelsize,
                          xlabel_fontsize=xlabel_fontsize, ylabel_fontsize=ylabel_fontsize, cbar_fontsize=cbar_fontsize,
                          title_fontsize=title_fontsize)


def create_diagnostic_pixel_counts_histogram(output_model_image_path, output_folder, waveband,
                                             figure_size=(10, 8), xtick_labelsize=15, ytick_labelsize=15,
                                             xlabel_fontsize=20, ylabel_fontsize=20, legend_fontsize=20,
                                             title_fontsize=20):
    """

    :param output_model_image_path:
    :param output_folder:
    :param waveband:
    :param figure_size:
    :param xtick_labelsize:
    :param ytick_labelsize:
    :param xlabel_fontsize:
    :param ylabel_fontsize:
    :param legend_fontsize:
    :param title_fontsize:
    :return:
    """

    source_image = fits.getdata(output_model_image_path, ext=1)
    model_image = fits.getdata(output_model_image_path, ext=2)
    residual_image = fits.getdata(output_model_image_path, ext=3)

    plt.clf()
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['xtick.labelsize'] = xtick_labelsize
    plt.rcParams['ytick.labelsize'] = ytick_labelsize
    plt.hist(source_image.ravel(), bins='auto', color='blue', density=True, label='Original Image', histtype='step')
    plt.hist(model_image.ravel(), bins='auto', color='green', density=True, alpha=0.5, label='Model Image',
             histtype='step')
    plt.hist(residual_image.ravel(), bins='auto', color='red', density=True, alpha=0.5, label='Residual Image',
             histtype='step')
    plt.xlabel('Pixel Values', fontsize=xlabel_fontsize)
    plt.ylabel('Number of Pixels', fontsize=ylabel_fontsize)
    combined = np.concatenate((source_image.ravel(), model_image.ravel(), residual_image.ravel()))
    plt.xlim(np.percentile(combined, 1), np.percentile(combined, 99))
    plt.yscale('log')
    plt.legend(loc='best', fontsize=legend_fontsize)
    plt.title('Pixel Count Histogram, {}'.format(waveband), fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Pixel_count_histogram_{}.pdf'.format(waveband)))
    plt.close()


def create_gaussian_fit_residual_image_counts(output_model_image_path, output_folder, waveband,
                                              figure_size=(10, 8), xtick_labelsize=15, ytick_labelsize=15,
                                              xlabel_fontsize=20, ylabel_fontsize=20, legend_fontsize=20,
                                              title_fontsize=20):
    """

    :param output_model_image_path:
    :param output_folder:
    :param waveband:
    :param figure_size:
    :param xtick_labelsize:
    :param ytick_labelsize:
    :param xlabel_fontsize:
    :param ylabel_fontsize:
    :param legend_fontsize:
    :param title_fontsize:
    :return:
    """

    residual_image = fits.getdata(output_model_image_path, ext=3)
    residual_hist = np.histogram(residual_image.ravel(), bins='auto', density=True)

    p0 = [0, 1, 0.01]
    popt, pcov = curve_fit(gaussian_function, residual_hist[1][:-1], residual_hist[0], p0=p0)

    plt.clf()
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['xtick.labelsize'] = xtick_labelsize
    plt.rcParams['ytick.labelsize'] = ytick_labelsize

    plt.plot(residual_hist[1][:-1], residual_hist[0], color='blue', lw=2, label='Pixel counts')
    plt.plot(residual_hist[1][:-1], gaussian_function(residual_hist[1][:-1], popt[0], popt[1], popt[2]),
             color='red', lw=2, label='Gaussian fit')

    plt.vlines(popt[0], ymin=0, ymax=max(gaussian_function(residual_hist[1][:-1], popt[0], popt[1], popt[2])),
               color='red', lw=2, ls='dashed', label='Mean: {:.2E}'.format(popt[0]))

    plt.xlabel('Pixel Values', fontsize=xlabel_fontsize)
    plt.ylabel('Number of Pixels', fontsize=ylabel_fontsize)
    plt.xlim(popt[0] - 3 * popt[-1], popt[0] + 3 * popt[-1])
    plt.legend(loc='upper left', fontsize=legend_fontsize)
    plt.title('Gaussian fit to Residual Image pixel counts, {}'.format(waveband), fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'Residual_image_pixel_count_histogram_{}.pdf'.format(waveband)))
    plt.close()


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
                                              light_profile_key, phot_apertures, output_folder,
                                              figure_size=(10, 8), xtick_labelsize=15, ytick_labelsize=15,
                                              xlabel_fontsize=20, ylabel_fontsize=20, legend_fontsize=20,
                                              title_fontsize=20):
    """

    :param best_fitting_galaxy_catalogue_filename:
    :param source_galaxies_catalogue:
    :param waveband:
    :param pixel_scale:
    :param ra_key:
    :param dec_key:
    :param galaxy_id_key:
    :param light_profile_key:
    :param phot_apertures
    :param output_folder:
    :param figure_size:
    :param xtick_labelsize:
    :param ytick_labelsize:
    :param xlabel_fontsize:
    :param ylabel_fontsize:
    :param legend_fontsize:
    :param title_fontsize:
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

    unique_galaxy_ids = list(set(galaxy_id))

    for i in range(len(unique_galaxy_ids)):
        obj_idxs = np.where(galaxy_id == unique_galaxy_ids[i])

        mu_from_mag_aper = np.empty_like(r_aper)
        mu_err_from_mag_aper = np.empty_like(r_aper)
        mu_from_mag_aper[0] = mag_aper[obj_idxs][0][0] + 2.5 * np.log10(2 * np.pi) + 5 * np.log10(r_aper[0] *
                                                                                                  pixel_scale)
        mu_err_from_mag_aper[0] = mag_aper_err[obj_idxs][0][0]
        for j in range(1, len(r_aper)):
            mu_from_mag_aper[j] = sb_profile_from_sextractor_mag_aper(mag_aper[obj_idxs][0][j - 1],
                                                                      mag_aper[obj_idxs][0][j],
                                                                      r_aper[j - 1] * pixel_scale,
                                                                      r_aper[j] * pixel_scale)
            mu_err_from_mag_aper[j] = sb_profile_errors_from_sextractor_mag_aper(mag_aper[obj_idxs][0][j - 1],
                                                                                 mag_aper[obj_idxs][0][j],
                                                                                 mag_aper_err[obj_idxs][0][j],
                                                                                 mag_aper_err[obj_idxs][0][j - 1])

        plt.clf()
        plt.rcParams['figure.figsize'] = figure_size
        plt.rcParams['xtick.labelsize'] = xtick_labelsize
        plt.rcParams['ytick.labelsize'] = ytick_labelsize
        r_arcsec = np.arange(r_aper[0] * pixel_scale, r_aper[-1] * pixel_scale, 0.01)

        final_profile = []
        for k in range(len(galaxy_id[obj_idxs])):
            if light_profile[obj_idxs][k] == 'sersic':
                plt.plot(r_arcsec, sersic_profile_mag_units(mu_e[obj_idxs][k], kappa[obj_idxs][k], r_arcsec,
                                                            re[obj_idxs][k] * pixel_scale, n[obj_idxs][k]),
                         lw=2, ls='dashed', color='black', label='Best-fitting Sersic profile, component {}'.format(k))
                final_profile.append(sersic_profile_mag_units(mu_e[obj_idxs][k], kappa[obj_idxs][k], r_arcsec,
                                                              re[obj_idxs][k] * pixel_scale, n[obj_idxs][k]))
            elif light_profile[obj_idxs][k] == 'devauc':
                plt.plot(r_arcsec, sersic_profile_mag_units(mu_e[obj_idxs][k], 7.669, r_arcsec,
                                                            re[obj_idxs][k] * pixel_scale, 4),
                         lw=2, ls='dashed', color='black',
                         label='Best-fitting de Vaucouleurs profile, component {}'.format(k))
                final_profile.append(sersic_profile_mag_units(mu_e[obj_idxs][k], 7.669, r_arcsec,
                                                              re[obj_idxs][k] * pixel_scale, 4))
            elif light_profile[obj_idxs][k] == 'expdisk':
                plt.plot(r_arcsec, sersic_profile_mag_units(mu_e[obj_idxs][k], 1.678, r_arcsec,
                                                            re[obj_idxs][k] * pixel_scale, 1),
                         lw=2, ls='dashed', color='black',
                         label='Best-fitting Exponential disk profile, component {}'.format(k))
                final_profile.append(sersic_profile_mag_units(mu_e[obj_idxs][k], 1.678, r_arcsec,
                                                              re[obj_idxs][k] * pixel_scale, 1))
            else:
                raise ValueError

        if len(galaxy_id[obj_idxs]) > 1:
            final_profile_fluxes = 10**(np.array(final_profile) / (-2.5))
            plt.plot(r_arcsec, -2.5 * np.log10(sum(final_profile_fluxes)), lw=2, ls='solid', color='black',
                     label='Total light profile')

        plt.errorbar(r_aper * pixel_scale, mu_from_mag_aper, yerr=mu_err_from_mag_aper,
                     fmt='o', ms=10, capsize=5, capthick=2, elinewidth=5, mec='black', color='red',
                     label='SExtractor aperture photometry')
        plt.xlabel('r [arcsec]', fontsize=xlabel_fontsize)
        plt.ylabel(r'SB [mag arcsec$^{-2}$]', fontsize=ylabel_fontsize)
        plt.ylim(np.nanmin(mu_from_mag_aper) - 5, np.nanmax(mu_from_mag_aper) + 5)
        plt.gca().invert_yaxis()
        plt.legend(loc='best', fontsize=legend_fontsize)
        plt.title('Comparison ID:{} {} with aperture photometry'.format(galaxy_id[i], waveband),
                  fontsize=title_fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,
                                 'ID{}_{}_best_fit_model_vs_sextractor_photometry.pdf'
                                 .format(galaxy_id[i], waveband)))
        plt.close()
