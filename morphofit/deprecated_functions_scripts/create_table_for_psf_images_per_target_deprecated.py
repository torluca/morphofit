#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import h5py
import glob
import numpy as np

root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'
target_names = ['abells1063', 'macs0416', 'macs1149']  # 'abell370', 'abell2744', 'macs0717', 'macs1206']
target_param_tables = [root + '{}/{}_param_table.fits'.format(target, target) for target in target_names]
target_param_tables = [target.encode('utf8') for target in target_param_tables]

target_star_positions = []
wavebands = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
for name in target_names:
    target_star_positions.append(['{}{}/{}_star_positions_isol_{}.fits'
                                 .format(root, name, name, waveband).encode('utf8') for waveband in wavebands])

wavebands = [name.encode('utf8') for name in wavebands]
wavebands = np.full((len(target_names), 7), wavebands)

root_targets = ['{}{}/'.format(root, name).encode('utf8') for name in target_names]

sci_images = []
for name in target_names:
    images = glob.glob(root + name + '/*_drz.fits')
    wave_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
    waves = {b: i for i, b in enumerate(wave_list)}
    images = sorted(images, key=lambda x: waves[x.split('_')[-2]])
    # images.sort()
    sci = [name.encode('utf8') for name in images]
    sci_images.append(sci)

target_names = [name.encode('utf8') for name in target_names]

psf_image_size = 100

pixel_scale = 0.060

with h5py.File(root + 'h5tables/table_psf_creation_run.h5', mode='w') as h5table:
    h5table.create_dataset(name='target_names', data=target_names)
    h5table.create_dataset(name='root_targets', data=root_targets)
    h5table.create_dataset(name='sci_images', data=sci_images)
    h5table.create_dataset(name='wavebands', data=wavebands)
    h5table.create_dataset(name='target_star_positions', data=target_star_positions)
    h5table.create_dataset(name='psf_image_size', data=psf_image_size)
    h5table.create_dataset(name='target_param_tables', data=target_param_tables)
    h5table.create_dataset(name='pixel_scale', data=pixel_scale)
