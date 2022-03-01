#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import h5py
import numpy as np
import itertools

# morphofit imports

# cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
psf_types = ['psf_pca', 'direct']
background_estimate_methods = ['sky_free_fit', 'sky_fixed_value']
sigma_image_types = ['sigma_custom', 'sigma_int_gen']
waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
indices = np.arange(0, 95, 1, dtype=int)
length = len(cluster_names) * len(psf_types) * len(background_estimate_methods) * len(sigma_image_types) * \
         len(waveband_list) * len(indices)
print(length)
root = np.full(length, '/cluster/scratch/torluca/gal_evo/'.encode('utf8'))

combinations = [['{}'.format(x), '{}'.format(y), '{}'.format(z), '{}'.format(m), '{}'.format(n), '{}'.format(l)]
                for x, y, z, m, n, l in itertools.product(cluster_names, psf_types, background_estimate_methods,
                                                          sigma_image_types, waveband_list, indices)]

cluster_names_list = []
psf_types_list = []
background_estimate_methods_list = []
sigma_image_types_list = []
waveband_list_list = []
indices_list = []

for i in range(length):
    cluster_names_list.append(combinations[i][0].encode('utf8'))
    psf_types_list.append(combinations[i][1].encode('utf8'))
    background_estimate_methods_list.append(combinations[i][2].encode('utf8'))
    sigma_image_types_list.append(combinations[i][3].encode('utf8'))
    waveband_list_list.append(combinations[i][4].encode('utf8'))
    indices_list.append(combinations[i][5].encode('utf8'))

with h5py.File('galfit_stamps_table.h5', 'w') as f:
    f.create_dataset("root", data=root)
    f.create_dataset("idxs", data=indices_list)
    f.create_dataset("cluster_name", data=cluster_names_list)
    f.create_dataset("psf_types", data=psf_types_list)
    f.create_dataset("background_estimate_methods", data=background_estimate_methods_list)
    f.create_dataset("sigma_image_types", data=sigma_image_types_list)
    f.create_dataset("waveband", data=waveband_list_list)
