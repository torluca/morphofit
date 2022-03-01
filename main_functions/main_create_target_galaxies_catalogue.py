#! /usr/bin/env python

# Copyright (C) 2019 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import argparse
from astropy.table import Table


# morphofit imports
from morphofit.utils import get_logger
from morphofit.catalogue_managing import match_catalogues

logger = get_logger(__file__)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    root = '/Users/torluca/Documents/PHD/gal_evo_paper/kormendy_relation_wavelength_paper/'
    targets = ['abells1063', 'macs0416', 'macs1149']
    input_catalogues = [root + '{}/HST_{}_multiband.forced.sexcat'.format(name, name) for name in targets]
    targets_catalogues = [root + 'res/catalogues/abells1063_v3.5_zcat_MUSE.fits',
                          root + 'res/catalogues/macs0416_v4.2_zcat_MUSE.fits',
                          root + 'res/catalogues/macs1149_v1.0_zcat_MUSE.fits']

    for index in indices:

        logger.info('=============================== running on index={}'.format(index))

        input_catalogue = Table.read(input_catalogues[index], format='fits')
        targets_catalogue = Table.read(targets_catalogues[index], format='fits')
        matched_target_galaxies_catalogue = match_catalogues(input_catalogue, targets_catalogue,
                                                             cat1_ra_key='ALPHAWIN_J2000_f814w',
                                                             cat1_dec_key='DELTAWIN_J2000_f814w',
                                                             cat2_ra_key='RA', cat2_dec_key='DEC')
        matched_target_galaxies_catalogue.write(root + '{}/HST_{}_target_multiband.forced.sexcat'
                                                .format(targets[index], targets[index]), format='fits', overwrite=True)

        yield index


def setup(args):
    """

    :param args:
    :return:
    """

    description = "Create target galaxies catalogue"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    args = parser.parse_args(args)
