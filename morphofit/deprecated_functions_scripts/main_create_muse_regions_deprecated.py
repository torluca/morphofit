#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import os
from astropy.table import Table

# morphofit imports
from morphofit.image_utils import cut_muse_fov, cut_regions
from morphofit.catalogue_managing import check_parameters_for_next_fitting, assign_sources_to_regions

cluster_names = ['abells1063']
epochs_acs = {'abell370': 'v1.0-epoch1', 'abell2744': 'v1.0-epoch2', 'abells1063': 'v1.0-epoch1',
              'macs0416': 'v1.0',
              'macs0717': 'v1.0-epoch1', 'macs1149': 'v1.0-epoch2', 'macs1206': 'v1'}
epochs_wfc3 = {'abell370': 'v1.0-epoch2', 'abell2744': 'v1.0', 'abells1063': 'v1.0-epoch1',
               'macs0416': 'v1.0-epoch2',
               'macs0717': 'v1.0-epoch2', 'macs1149': 'v1.0-epoch2', 'macs1206': 'v1'}
root = '/cluster/scratch/torluca/gal_evo/'
for name in cluster_names:
    if name == 'macs1206':
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
        acs_waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']
        prefix = 'hlsp_clash_hst_30mas'
    else:
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
        acs_waveband_list = ['f435w', 'f606w', 'f814w']
        prefix = 'hlsp_frontier_hst_30mas'
    stamps_mastercat = Table.read(
        root + '{}/stamps/cats/{}_{}_stamps_mediangalfit_multiband.forced.sexcat'.format(name, prefix, name),
        format='fits')
    root_input = root + '{}/'.format(name)
    for band in waveband_list:
        if band in acs_waveband_list:
            if name == 'macs1206':
                additional_text = 'hlsp_clash_hst_acs-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_acs-30mas-selfcal'

            image_name = '{}_{}_{}_{}_drz.fits'.format(additional_text,
                                                       name, band,
                                                       epochs_acs[name])
            seg_image_name = '{}_{}_{}_{}_drz_forced_seg.fits'.format(additional_text,
                                                                      name, band,
                                                                      epochs_acs[name])
            rms_image_name = '{}_{}_{}_{}_rms.fits'.format(additional_text,
                                                           name, band,
                                                           epochs_acs[name])
        else:
            if name == 'macs1206':
                additional_text = 'hlsp_clash_hst_wfc3ir-30mas'
            else:
                additional_text = 'hlsp_frontier_hst_wfc3-30mas-bkgdcor'
            image_name = '{}_{}_{}_{}_drz.fits'.format(additional_text,
                                                       name, band,
                                                       epochs_wfc3[name])
            seg_image_name = '{}_{}_{}_{}_drz_forced_seg.fits'.format(additional_text,
                                                                      name, band,
                                                                      epochs_wfc3[name])
            rms_image_name = '{}_{}_{}_{}_rms.fits'.format(additional_text,
                                                           name, band,
                                                           epochs_wfc3[name])

        muse_fov_image, muse_fov_seg_image, muse_fov_rms_image = cut_muse_fov(root_input, image_name,
                                                                              seg_image_name, rms_image_name,
                                                                              root_input,
                                                                              stamps_mastercat)
        root_output = root_input + 'regions/'
        N = 4
        reg_image_filenames, reg_seg_image_filenames, reg_rms_image_filenames = cut_regions(root_input,
                                                                                            os.path.basename(
                                                                                                muse_fov_image),
                                                                                            os.path.basename(
                                                                                                muse_fov_seg_image),
                                                                                            os.path.basename(
                                                                                                muse_fov_rms_image),
                                                                                            root_output, N)
        stamps_mod_mastercat = check_parameters_for_next_fitting(stamps_mastercat, waveband_list)
        region_cats = assign_sources_to_regions(reg_image_filenames, stamps_mod_mastercat)
