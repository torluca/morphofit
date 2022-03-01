#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
from astropy.table import Table
import numpy as np

# morphofit imports
from morphofit.catalogue_managing import match_galfit_table_with_dacunha, match_galfit_table_with_gobat, match_with_zcat, match_galfit_table_with_published_params


root = '/cluster/scratch/torluca/gal_evo/'
root_folder_redshift_catalogues = '/cluster/scratch/torluca/gal_evo/res/'
cluster_name = 'macs1149'
redshift_catalogues = {'abell370': root_folder_redshift_catalogues,
                           'abell2744': root_folder_redshift_catalogues,
                           'abells1063': root_folder_redshift_catalogues + 'abells1063_v3.5_zcat.fits',
                           'macs0416': root_folder_redshift_catalogues,
                           'macs0717': root_folder_redshift_catalogues,
                           'macs1149': root_folder_redshift_catalogues + 'macs1149_v1.0_zcat.fits',
                           'macs1206': root_folder_redshift_catalogues}

table = Table.read(root + '{}/stamps/cats/hlsp_frontier_hst_30mas_{}_stamps_mediangalfit_multiband.forced.sexcat'.format(cluster_name,cluster_name),format='fits')
#muse_zcat = Table.read(redshift_catalogues[cluster_name], format='fits')
#table = match_with_zcat(table,muse_zcat)
w = np.where((table['ID'] != 99) & (table['z_ref'] != 2))
dacunha = Table.read(root + '{}/{}_dacunha_sedfit.fits'.format(cluster_name,cluster_name),format='fits')

newtable = match_galfit_table_with_dacunha(table[w], dacunha)

gobat = Table.read(root + '{}/{}_gobat_sedfit.fits'.format(cluster_name,cluster_name),format='fits')

finaltable = match_galfit_table_with_gobat(newtable, gobat)

published_cat = Table.read(root + '{}/{}_muse_published_param.fits'.format(cluster_name,cluster_name),format='fits')

finaltable = match_galfit_table_with_published_params(finaltable, published_cat)

finaltable.write(root+'{}/hlsp_frontier_hst_30mas_{}_stamps_mediangalfit_multiband_allparams_muse.forced.sexcat'.format(cluster_name,cluster_name),format='fits',overwrite=True)
