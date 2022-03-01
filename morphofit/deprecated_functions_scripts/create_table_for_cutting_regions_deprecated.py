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
target_names = ['abells1063', 'macs0416', 'macs1149']  # , 'abell370', 'abell2744', 'macs0717', 'macs1206']
root_targets = ['{}{}/'.format(root, name).encode('utf8') for name in target_names]
telescope_name = 'HST'

sci_image_filenames, rms_image_filenames, seg_image_filenames, exp_image_filenames = [], [], [], []
external_catalogue_filenames = []

for name in target_names:

    images = glob.glob(root + name + '/*_drz.fits')
    wave_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
    waves = {b: i for i, b in enumerate(wave_list)}
    images = sorted(images, key=lambda x: waves[x.split('_')[-2]])
    # images.sort()
    sci = [name.encode('utf8') for name in images]
    sci_image_filenames.append(sci)

    images = glob.glob(root + name + '/*_rms.fits')
    images = sorted(images, key=lambda x: waves[x.split('_')[-2]])
    # images.sort()
    rms = [name.encode('utf8') for name in images]
    if not rms:
        rms = list(np.full(len(sci), 'None'.encode('utf8')))
    rms_image_filenames.append(rms)

    images = glob.glob(root + name + '/*drz_forced_seg.fits')
    images = sorted(images, key=lambda x: waves[x.split('_')[-4]])
    # images.sort()
    seg = [name.encode('utf8') for name in images]
    seg_image_filenames.append(seg)

    images = glob.glob(root + name + '/*_exp.fits')
    images = sorted(images, key=lambda x: waves[x.split('_')[-2]])
    # images.sort()
    exp = [name.encode('utf8') for name in images]
    if not exp:
        exp = list(np.full(len(sci), 'None'.encode('utf8')))
    exp_image_filenames.append(exp)

    external_catalogue_filename = root + name + '/stamps/cats/{}_{}_stamps_multiband.forced.sexcat'\
        .format(telescope_name, name)
    external_catalogue_filenames.append(external_catalogue_filename.encode('utf8'))

filters = ['f435w'.encode('utf8'), 'f606w'.encode('utf8'), 'f814w'.encode('utf8'),
           'f105w'.encode('utf8'), 'f125w'.encode('utf8'), 'f140w'.encode('utf8'),
           'f160w'.encode('utf8')]
wavebands = np.full((len(target_names), 7), filters)
crop_routine = 'catalogue_based'.encode('utf8')
crop_suffix = 'muse'.encode('utf8')
x_keyword = 'XWIN_IMAGE_f814w'.encode('utf8')
y_keyword = 'YWIN_IMAGE_f814w'.encode('utf8')
number_of_regions_perside = 3

with h5py.File(root + 'h5tables/table_cutting_regions_run.h5', mode='w') as h5table:
    h5table.create_dataset(name='root_targets', data=root_targets)
    h5table.create_dataset(name='sci_image_filenames', data=sci_image_filenames)
    h5table.create_dataset(name='rms_image_filenames', data=rms_image_filenames)
    h5table.create_dataset(name='seg_image_filenames', data=seg_image_filenames)
    h5table.create_dataset(name='exp_image_filenames', data=exp_image_filenames)
    h5table.create_dataset(name='wavebands', data=wavebands)
    h5table.create_dataset(name='crop_routine', data=crop_routine)
    h5table.create_dataset(name='external_catalogue_filenames', data=external_catalogue_filenames)
    h5table.create_dataset(name='crop_suffix', data=crop_suffix)
    h5table.create_dataset(name='x_keyword', data=x_keyword)
    h5table.create_dataset(name='y_keyword', data=y_keyword)
    h5table.create_dataset(name='number_of_regions_perside', data=number_of_regions_perside)
