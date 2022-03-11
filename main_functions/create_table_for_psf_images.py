#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import h5py
import numpy as np

root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'
target_names = ['abells1063', 'macs0416', 'macs1149']  # 'abell370', 'abell2744', 'macs0717', 'macs1206']

telescope_name = 'HST'
root_output = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/test_psfs/'
wavebands = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']

target_star_positions = [[root.encode('utf8') +
                          '{}/{}_star_positions_isol_{}.fits'.format(target, target, waveband).encode('utf8')
                          for target in target_names] for waveband in wavebands]
sci_images = [[root.encode('utf8') +
               '{}/{}_60mas_{}_{}_drz.fits'.format(target, telescope_name, target, waveband).encode('utf8')
               for target in target_names] for waveband in wavebands]
seg_images = [[root.encode('utf8') +
               '{}/{}_60mas_{}_{}_drz_seg.fits'.format(target, telescope_name, target, waveband).encode('utf8')
               for target in target_names] for waveband in wavebands]
target_param_tables = [[root.encode('utf8') +
                        '{}/{}_param_table.fits'.format(target, target).encode('utf8') for target in target_names]
                       for waveband in wavebands]

wavebands = [name.encode('utf8') for name in wavebands]
wavebands = np.full((len(target_names), 7), wavebands)

root_targets = ['{}{}/'.format(root, name).encode('utf8') for name in target_names]

psf_image_size = 100

pixel_scale = 0.060

with h5py.File(root + 'h5tables/table_psf_creation_run.h5', mode='w') as h5table:
    h5table.create_dataset(name='root_targets', data=root_targets)
    h5table.create_dataset(name='sci_images', data=sci_images)
    h5table.create_dataset(name='seg_images', data=seg_images)
    h5table.create_dataset(name='wavebands', data=wavebands)
    h5table.create_dataset(name='target_star_positions', data=target_star_positions)
    h5table.create_dataset(name='psf_image_size', data=psf_image_size)
    h5table.create_dataset(name='target_param_tables', data=target_param_tables)
    h5table.create_dataset(name='pixel_scale', data=pixel_scale)
