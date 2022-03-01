#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)


# External modules
import pickle
import glob
import os

# morphofit imports
from morphofit.utils import merge_dictionaries

root = '/Users/torluca/Documents/PHD/gal_evo_paper/stellar_pop_paper/'
# cluster_names = ['abell370','abell2744','abells1063','macs0416','macs0717','macs1149','macs1206']
cluster_names = ['abells1063']
# substitute regions with stamps for galfit region fitting

for cluster_name in cluster_names:
    pickle_files = glob.glob(root + '{}/stamps/pkl_files/*.pkl'.format(cluster_name))
    x_dict = merge_dictionaries([name for name in pickle_files if 'x_dict' in name])
    y_dict = merge_dictionaries([name for name in pickle_files if 'y_dict' in name])
    ra_dict = merge_dictionaries([name for name in pickle_files if 'ra_dict' in name])
    dec_dict = merge_dictionaries([name for name in pickle_files if 'dec_dict' in name])
    mag_dict = merge_dictionaries([name for name in pickle_files if 'mag_dict' in name])
    re_dict = merge_dictionaries([name for name in pickle_files if 're_dict' in os.path.basename(name)[0:7]])
    n_dict = merge_dictionaries([name for name in pickle_files if 'n_dict' in name])
    ar_dict = merge_dictionaries([name for name in pickle_files if 'ar_dict' in name])
    pa_dict = merge_dictionaries([name for name in pickle_files if 'pa_dict' in name])
    sky_value_dict = merge_dictionaries([name for name in pickle_files if 'sky_value_dict' in name])
    sky_x_grad_dict = merge_dictionaries([name for name in pickle_files if 'sky_x_grad_dict' in name])
    sky_y_grad_dict = merge_dictionaries([name for name in pickle_files if 'sky_y_grad_dict' in name])
    red_chisquare_dict = merge_dictionaries([name for name in pickle_files if 'red_chisquare_dict' in name])

    f = open(root + "{}/stamps/x_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(x_dict, f)
    f.close()
    f = open(root + "{}/stamps/y_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(y_dict, f)
    f.close()
    f = open(root + "{}/stamps/ra_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(ra_dict, f)
    f.close()
    f = open(root + "{}/stamps/dec_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(dec_dict, f)
    f.close()
    f = open(root + "{}/stamps/mag_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(mag_dict, f)
    f.close()
    f = open(root + "{}/stamps/re_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(re_dict, f)
    f.close()
    f = open(root + "{}/stamps/n_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(n_dict, f)
    f.close()
    f = open(root + "{}/stamps/ar_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(ar_dict, f)
    f.close()
    f = open(root + "{}/stamps/pa_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(pa_dict, f)
    f.close()
    f = open(root + "{}/stamps/sky_value_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(sky_value_dict, f)
    f.close()
    f = open(root + "{}/stamps/sky_x_grad_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(sky_x_grad_dict, f)
    f.close()
    f = open(root + "{}/stamps/sky_y_grad_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(sky_y_grad_dict, f)
    f.close()
    f = open(root + "{}/stamps/red_chisquare_dict_{}.pkl".format(cluster_name, cluster_name), "wb")
    pickle.dump(red_chisquare_dict, f)
    f.close()
