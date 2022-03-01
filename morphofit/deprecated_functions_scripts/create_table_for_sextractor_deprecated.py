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
import morphofit
from pkg_resources import resource_filename


root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'
target_names = ['abells1063', 'macs0416', 'macs1149']  # 'abell370', 'abell2744', 'macs0717', 'macs1206']
root_targets = ['{}{}/'.format(root, name).encode('utf8') for name in target_names]

sci_images, rms_images, exp_images = [], [], []

for name in target_names:
    images = glob.glob(root + name + '/*_drz.fits')
    wave_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
    waves = {b: i for i, b in enumerate(wave_list)}
    images = sorted(images, key=lambda x: waves[x.split('_')[-3]])
    sci = [name.encode('utf8') for name in images]
    sci_images.append(sci)
    images = glob.glob(root + name + '/*_rms.fits')
    images = sorted(images, key=lambda x: waves[x.split('_')[-3]])
    rms = [name.encode('utf8') for name in images]
    if not rms:
        rms = list(np.full(len(sci), 'None'.encode('utf8')))
    rms_images.append(rms)
    images = glob.glob(root + name + '/*_exp.fits')
    images = sorted(images, key=lambda x: waves[x.split('_')[-3]])
    exp = [name.encode('utf8') for name in images]
    if not exp:
        exp = list(np.full(len(sci), 'None'.encode('utf8')))
    exp_images.append(exp)

target_names = [name.encode('utf8') for name in target_names]

# subprocess.run(['tar', '-cvf', '{}.tar'.format(field)] + lista)

filters = ['f435w'.encode('utf8'), 'f606w'.encode('utf8'), 'f814w'.encode('utf8'),
           'f105w'.encode('utf8'), 'f125w'.encode('utf8'), 'f140w'.encode('utf8'),
           'f160w'.encode('utf8')]
wavebands = np.full((len(target_names), len(filters)), filters)
pixel_scales = np.full(len(target_names), 0.060)
telescope_names = np.full(len(target_names), 'HST'.encode('utf8'))

photo_cmd = ['-DETECT_MINAREA'.encode('utf8'), str(10).encode('utf8'),
             '-DETECT_THRESH'.encode('utf8'), str(1.0).encode('utf8'),
             '-ANALYSIS_THRESH'.encode('utf8'), str(1.5).encode('utf8'),
             '-DEBLEND_NTHRESH'.encode('utf8'), str(64).encode('utf8'),
             '-DEBLEND_MINCONT'.encode('utf8'), str(0.0001).encode('utf8'),
             '-BACK_SIZE'.encode('utf8'), str(64).encode('utf8'),
             '-BACK_FILTERSIZE'.encode('utf8'), str(3).encode('utf8')]

seeing_initial_guesses = np.full((len(target_names), len(filters)), [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
sextractor_binary = '/usr/local/bin/sex'.encode('utf8')
sextractor_config = resource_filename(morphofit.__name__, 'res/sextractor/default.sex').encode('utf8')
sextractor_params = resource_filename(morphofit.__name__, 'res/sextractor/default.param').encode('utf8')
sextractor_filter = resource_filename(morphofit.__name__, 'res/sextractor/gauss_3.0_5x5.conv').encode('utf8')
sextractor_nnw = resource_filename(morphofit.__name__, 'res/sextractor/default.nnw').encode('utf8')
sextractor_checkimages = ['SEGMENTATION'.encode('utf8')]
sextractor_checkimages_endings = ['_seg.fits'.encode('utf8')]

ext_star_cat = [resource_filename(morphofit.__name__,
                                  'res/star_catalogues/abells1063_star_positions_isol_f160w.fits').encode('utf8'),
                resource_filename(morphofit.__name__,
                                  'res/star_catalogues/macs0416_star_positions_isol_f160w.fits').encode('utf8'),
                resource_filename(morphofit.__name__,
                                  'res/star_catalogues/macs1149_star_positions_isol_f160w.fits').encode('utf8')]

with h5py.File(root + '/h5tables/table_sextractor_run.h5', mode='w') as h5table:
    h5table.create_dataset(name='telescope_names', data=telescope_names)
    h5table.create_dataset(name='target_names', data=target_names)
    h5table.create_dataset(name='root_targets', data=root_targets)
    h5table.create_dataset(name='sci_images', data=sci_images)
    h5table.create_dataset(name='rms_images', data=rms_images)
    h5table.create_dataset(name='exp_images', data=exp_images)
    h5table.create_dataset(name='wavebands', data=wavebands)
    h5table.create_dataset(name='pixel_scales', data=pixel_scales)
    h5table.create_dataset(name='photo_cmd', data=photo_cmd)
    h5table.create_dataset(name='seeing_initial_guesses', data=seeing_initial_guesses)
    h5table.create_dataset(name='sextractor_binary', data=sextractor_binary)
    h5table.create_dataset(name='sextractor_config', data=sextractor_config)
    h5table.create_dataset(name='sextractor_params', data=sextractor_params)
    h5table.create_dataset(name='sextractor_filter', data=sextractor_filter)
    h5table.create_dataset(name='sextractor_nnw', data=sextractor_nnw)
    h5table.create_dataset(name='sextractor_checkimages', data=sextractor_checkimages)
    h5table.create_dataset(name='sextractor_checkimages_endings', data=sextractor_checkimages_endings)
    h5table.create_dataset(name='ext_star_cat', data=ext_star_cat)
