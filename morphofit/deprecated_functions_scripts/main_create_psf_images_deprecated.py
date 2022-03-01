#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules

# morphofit imports
from morphofit.psf_estimation import create_psf_image


root = '/cluster/scratch/torluca/gal_evo/'
#cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
psf_image_size = 30
epochs_acs = {'abell370':'v1.0-epoch1', 'abell2744':'v1.0-epoch2', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0',
              'macs0717':'v1.0-epoch1', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}
epochs_wfc3 = {'abell370':'v1.0-epoch2', 'abell2744':'v1.0', 'abells1063':'v1.0-epoch1', 'macs0416':'v1.0-epoch2',
               'macs0717':'v1.0-epoch2', 'macs1149':'v1.0-epoch2', 'macs1206':'v1'}

for cluster_name in cluster_names:
    cluster_param_table = root + '{}/{}_param_table.fits'.format(cluster_name, cluster_name)
    star_positions = root + '{}/stars/star_positions_isolated.txt'.format(cluster_name)
    if cluster_name == 'macs1206':
        waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp', 'f105w']
        acs_waveband_list = ['f435w', 'f475w', 'f606w', 'f625w', 'f775w', 'f814w', 'f850lp']
        wfc3_waveband_list = ['f105w']
    else:
        waveband_list = ['f435w', 'f606w', 'f814w', 'f105w', 'f125w', 'f140w', 'f160w']
        acs_waveband_list = ['f435w', 'f606w', 'f814w']
        wfc3_waveband_list = ['f105w', 'f125w', 'f140w', 'f160w']
    create_psf_image(root, cluster_name, star_positions, psf_image_size, cluster_param_table, waveband_list,
                     acs_waveband_list, wfc3_waveband_list, epochs_acs[cluster_name], epochs_wfc3[cluster_name])
