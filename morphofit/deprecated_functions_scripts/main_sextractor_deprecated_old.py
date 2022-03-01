#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import os
import glob
from astropy.table import Table
from pkg_resources import resource_filename

# morphofit imports
import morphofit
from morphofit.utils import get_saturations, get_gains, get_exposure_times, get_zeropoints, create_image_params_table
from morphofit.background_estimation import get_background_parameters
from morphofit.psf_estimation import get_seeings
from morphofit.image_utils import create_detection_image, create_rms_image, create_rms_image_clash
from morphofit.run_sextractor import get_sextractor_forced_cmd, run_sex_dual_mode
from morphofit.catalogue_managing import get_multiband_catalogue, match_with_zcat


def main_sextractor(cluster_name, root_folder, ext_star_cat, redshift_catalogues, muse_redshift_catalogues,
                    sextractor_binary, sextractor_config, sextractor_params, sextractor_filter, sextractor_nnw,
                    sextractor_checkimages, sextractor_checkimages_endings, photo_cmd, epochs_acs, epochs_wfc3):
    """

    :param cluster_name:
    :param root_folder:
    :param ext_star_cat:
    :param redshift_catalogues:
    :param muse_redshift_catalogues:
    :param sextractor_binary:
    :param sextractor_config:
    :param sextractor_params:
    :param sextractor_filter:
    :param sextractor_nnw:
    :param sextractor_checkimages:
    :param sextractor_checkimages_endings:
    :param photo_cmd:
    :param epochs_acs:
    :param epochs_wfc3:
    :return:
    """

    root_cluster_folder = root_folder + '{}/'.format(cluster_name)
    drz_images = glob.glob(root_cluster_folder + '*drz.fits')
    drz_images.sort()
    exp_images = glob.glob(root_cluster_folder + '*exp.fits')
    exp_images.sort()
    rms_images = glob.glob(root_cluster_folder + '*rms.fits')
    rms_images.sort()
    wht_images = glob.glob(root_cluster_folder + '*wht.fits')
    wht_images.sort()
    se_cats = [root_cluster_folder + os.path.basename(img).split('.fits')[0] + '.sexcat' for img in drz_images]
    se_forced_cats = [root_cluster_folder + os.path.basename(img).split('.fits')[0] + '_forced.sexcat' for img in drz_images]

    if cluster_name == 'macs1206':
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
        acs_waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']
        wfc3_waveband_list = ['f105w']
        pixel_scale = {'f435w': 0.030, 'f475w': 0.030, 'f606w': 0.030, 'f625w': 0.030, 'f775w': 0.030,
                       'f814w': 0.030, 'f850lp': 0.030, 'f105w': 0.030}
        acs_detect_name = root_cluster_folder + 'hlsp_clash_hst_acs-30mas_{}_{}_drz_detection.fits'.format(cluster_name,
                                                                                                           epochs_acs)
        wfc3_detect_name = root_cluster_folder + 'hlsp_clash_hst_wfc3ir-30mas_{}_{}_drz_detection.fits'.format(cluster_name,
                                                                                                               epochs_wfc3)
        detect_name = root_cluster_folder + 'hlsp_clash_hst_30mas_{}_{}_drz.fits'.format(cluster_name, epochs_acs)
        forced_detect_catalogue = detect_name.split('.fits')[0] + '.forced.sexcat'
        rms_detect_name = root_cluster_folder + 'hlsp_clash_hst_30mas_{}_{}_rms.fits'.format(cluster_name, epochs_acs)
        multiband_cat_name = root_cluster_folder + 'hlsp_clash_hst_30mas_{}_multiband.forced.sexcat'.format(cluster_name)
        prefix = 'hlsp_clash_hst_30mas'
    else:
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
        acs_waveband_list = ['f435w', 'f606w', 'f814w']
        wfc3_waveband_list = ['f105w', 'f125w', 'f140w', 'f160w']
        pixel_scale = {'f435w': 0.030, 'f606w': 0.030, 'f814w': 0.030, 'f105w': 0.030, 'f125w': 0.030,
                       'f140w': 0.030, 'f160w': 0.030}
        acs_detect_name = root_cluster_folder + 'hlsp_frontier_hst_acs-30mas-selfcal_{}_{}_drz_detection.fits'.format(
            cluster_name, epochs_acs)
        wfc3_detect_name = root_cluster_folder + 'hlsp_frontier_hst_wfc3-30mas-bkgdcor_{}_{}_drz_detection.fits'.format(
            cluster_name, epochs_wfc3)
        detect_name = root_cluster_folder + 'hlsp_frontier_hst_30mas_{}_{}_drz_detection.fits'.format(cluster_name,
                                                                                                      epochs_acs)
        forced_detect_catalogue = detect_name.split('.fits')[0] + '_forced.sexcat'
        rms_detect_name = root_cluster_folder + 'hlsp_frontier_hst_30mas_{}_{}_rms_detection.fits'.format(cluster_name,
                                                                                                          epochs_acs)
        multiband_cat_name = root_cluster_folder + 'hlsp_frontier_hst_30mas_{}_multiband.forced.sexcat'.format(cluster_name)
        prefix = 'hlsp_frontier_hst_30mas'

    saturations = get_saturations(drz_images, waveband_list)
    gains = get_gains(drz_images, waveband_list)
    exptimes = get_exposure_times(drz_images, waveband_list)
    zeropoints = get_zeropoints(cluster_name, drz_images, waveband_list)
    bkg_amps, bkg_sigmas = get_background_parameters(drz_images, waveband_list, se_cats,
                                                     saturations, zeropoints, gains, pixel_scale,
                                                     photo_cmd, rms_images,
                                                     sextractor_binary, sextractor_config, sextractor_params,
                                                     sextractor_filter, sextractor_nnw, sextractor_checkimages,
                                                     sextractor_checkimages_endings)
    fwhms, betas = get_seeings(drz_images, waveband_list, se_cats, ext_star_cat[cluster_name], pixel_scale, '0')
    param_table = create_image_params_table(waveband_list, saturations, gains, exptimes,
                                            zeropoints, bkg_amps, bkg_sigmas, fwhms, betas)
    param_table.write(root_cluster_folder + '{}_param_table.fits'.format(cluster_name), format='fits', overwrite=True)

    create_detection_image(drz_images, detect_name, fwhms, zeropoints,
                           saturations, gains, pixel_scale,
                           bkg_sigmas)
    if cluster_name == 'macs1206':
        create_rms_image_clash(wht_images, rms_detect_name, bkg_sigmas)
    else:
        create_rms_image(rms_images, rms_detect_name, bkg_sigmas)

    forced_cmd = get_sextractor_forced_cmd(drz_images, detect_name, se_forced_cats,
                                           waveband_list, fwhms, saturations,
                                           zeropoints, gains, pixel_scale,
                                           photo_cmd, rms_detect_name, sextractor_binary,
                                           sextractor_config, sextractor_params,
                                           sextractor_filter, sextractor_nnw,
                                           sextractor_checkimages,
                                           sextractor_checkimages_endings)
    run_sex_dual_mode(forced_cmd)

    multiband_cat = get_multiband_catalogue(se_forced_cats, forced_detect_catalogue, waveband_list)
    multiband_cat.write(multiband_cat_name, format='fits', overwrite=True)

    zcat = Table.read(redshift_catalogues[cluster_name], format='fits')
    muse_zcat = Table.read(muse_redshift_catalogues[cluster_name], format='fits')

    multiband_cat_zmatched = match_with_zcat(multiband_cat, zcat)
    multiband_cat_musematched = match_with_zcat(multiband_cat, muse_zcat)

    multiband_cat_zmatched.write(root_cluster_folder +
                                 '{}_{}_multiband_zcatmatched.forced.sexcat'.format(prefix, cluster_name),
                                 format='fits', overwrite=True)
    multiband_cat_musematched.write(root_cluster_folder +
                                    '{}_{}_multiband_musecatmatched.forced.sexcat'.format(prefix, cluster_name),
                                    format='fits', overwrite=True)


# cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
root_folder = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/'
root_folder_redshift_catalogues = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/res/catalogues/'
ext_star_cat = {'abell370': resource_filename(morphofit.__name__, "res/gaia_catalogues/abell370_gaiacat.fits"),
                'abell2744': resource_filename(morphofit.__name__, "res/gaia_catalogues/abell2744_gaiacat.fits"),
                'abells1063': resource_filename(morphofit.__name__, "res/gaia_catalogues/abells1063_gaiacat.fits"),
                'macs0416': resource_filename(morphofit.__name__, "res/gaia_catalogues/macs0416_gaiacat.fits"),
                'macs0717': resource_filename(morphofit.__name__, "res/gaia_catalogues/macs0717_gaiacat.fits"),
                'macs1149': resource_filename(morphofit.__name__, "res/gaia_catalogues/macs1149_gaiacat.fits"),
                'macs1206': resource_filename(morphofit.__name__, "res/gaia_catalogues/macs1206_gaiacat.fits")}
epochs_acs = {'abell370':'v1.0-epoch1', 'abell2744':'v1.0-epoch2', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0',
              'macs0717':'v1.0-epoch1', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
epochs_wfc3 = {'abell370':'v1.0-epoch2', 'abell2744':'v1.0', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0-epoch2',
               'macs0717':'v1.0-epoch2', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
redshift_catalogues = {'abell370': root_folder_redshift_catalogues,
                       'abell2744': root_folder_redshift_catalogues,
                       'abells1063': root_folder_redshift_catalogues + 'abells1063_v3.5_zcat.fits',
                       'macs0416': root_folder_redshift_catalogues,
                       'macs0717': root_folder_redshift_catalogues,
                       'macs1149': root_folder_redshift_catalogues,
                       'macs1206': root_folder_redshift_catalogues}
muse_redshift_catalogues = {'abell370': root_folder_redshift_catalogues,
                       'abell2744': root_folder_redshift_catalogues,
                       'abells1063': root_folder_redshift_catalogues + 'abells1063_v3.5_zcat_MUSE.fits',
                       'macs0416': root_folder_redshift_catalogues,
                       'macs0717': root_folder_redshift_catalogues,
                       'macs1149': root_folder_redshift_catalogues,
                       'macs1206': root_folder_redshift_catalogues}
sextractor_binary = '/usr/local/bin/sex'
sextractor_config = resource_filename(morphofit.__name__, "res/sextractor/default.sex")
sextractor_params = resource_filename(morphofit.__name__, "res/sextractor/default.param")
sextractor_filter = resource_filename(morphofit.__name__, "res/sextractor/gauss_3.0_5x5.conv")
sextractor_nnw = resource_filename(morphofit.__name__, "res/sextractor/default.nnw")
# sextractor_checkimages = ['SEGMENTATION', 'APERTURES']
# sextractor_checkimages_endings = ["_seg.fits", "_ap.fits"]
sextractor_checkimages = ['SEGMENTATION']
sextractor_checkimages_endings = ["_seg.fits"]
photo_cmd = ["-DETECT_MINAREA", str(10), "-DETECT_THRESH", str(1.0), "-ANALYSIS_THRESH", str(1.5),
             "-DEBLEND_NTHRESH", str(64), "-DEBLEND_MINCONT", str(0.001), "-BACK_SIZE", str(64),
             "-BACK_FILTERSIZE", str(3)]

for cluster_name in cluster_names:
    main_sextractor(cluster_name, root_folder, ext_star_cat, redshift_catalogues, muse_redshift_catalogues,
                    sextractor_binary, sextractor_config, sextractor_params, sextractor_filter, sextractor_nnw,
                    sextractor_checkimages, sextractor_checkimages_endings, photo_cmd, epochs_acs[cluster_name],
                    epochs_wfc3[cluster_name])

# Folder structure: e.g., /Users/../analysis/abells1063/
