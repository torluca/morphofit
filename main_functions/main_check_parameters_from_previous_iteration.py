#! /usr/bin/env python

# Copyright (C) 2019,2020 ETH Zurich, Institute for Particle Physics and Astrophysics
# Copyright (C) 2021 University Observatory, Ludwig-Maximilians-Universitaet Muenchen
# Author: Luca Tortorelli

# System imports
from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# External modules
import argparse
from astropy.table import Table

# morphofit imports
from morphofit.utils import get_logger
from morphofit.catalogue_managing import check_parameters_for_next_iteration


logger = get_logger(__file__)


def main(indices, args):
    """

    :param indices:
    :param args:
    :return:
    """

    args = setup(args)

    for index in indices:

        logger.info('=============================== running on index={}'.format(index))
        logger.info('=============================== check galaxy parameters')

        wavebands = args.wavebands.split(',')
        source_galaxies_catalogue_filenames = args.source_galaxies_catalogue_filenames.split(',')

        source_galaxies_catalogue = Table.read(source_galaxies_catalogue_filenames[index], format='fits', memmap=True)
        source_galaxies_check_parameters_keys = args.source_galaxies_check_parameters_keys.split(',')
        source_galaxies_check_parameters_limits = [int(value)
                                                   for value in args.source_galaxies_check_parameters_limits.aplit(',')]

        for waveband in wavebands:

            stamps_mastercatalogue = \
                check_parameters_for_next_iteration(source_galaxies_catalogue, waveband,
                                                    magnitude_keyword=source_galaxies_check_parameters_keys[0],
                                                    size_keyword=source_galaxies_check_parameters_keys[1],
                                                    minor_axis_keyword=source_galaxies_check_parameters_keys[2],
                                                    major_axis_keyword=source_galaxies_check_parameters_keys[3],
                                                    position_angle_keyword=source_galaxies_check_parameters_keys[4],
                                                    magnitude_error_limit=source_galaxies_check_parameters_limits[0],
                                                    magnitude_upper_limit=source_galaxies_check_parameters_limits[1],
                                                    size_error_limit=source_galaxies_check_parameters_limits[2],
                                                    size_upper_limit=source_galaxies_check_parameters_limits[3],
                                                    sersic_index_error_limit=source_galaxies_check_parameters_limits[4],
                                                    sersic_index_upper_limit=source_galaxies_check_parameters_limits[5],
                                                    sersic_index_lower_limit=source_galaxies_check_parameters_limits[6],
                                                    key=args.error_comparison_key)
            stamps_mastercatalogue.write(source_galaxies_catalogue_filenames[index].split('.fits') + '_corr.fits',
                                         format='fits', overwrite=True)

        yield index


def setup(args):
    """

    :param args:
    :return:
    """

    description = "Run GALFIT on stamps around target galaxies in Target Fields"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument('--source_galaxies_catalogue_filenames', type=str, action='store', default='mastercat.fits',
                        help='Comma-separated list of source galaxies catalogue filenames')
    parser.add_argument('--wavebands', type=str, action='store', default='f814w',
                        help='Comma-separated list of wavebands')
    parser.add_argument('--source_galaxies_check_parameters_keys', type=str, action='store',
                        default='MAG_AUTO,FLUX_RADIUS,BWIN_IMAGE,AWIN_IMAGE,THETAWIN_SKY',
                        help='When --check_parameters=True, it compares the input galaxy parameters with their'
                             'corresponding ones from SExtractor')
    parser.add_argument('--source_galaxies_check_parameters_limits', type=str, action='store',
                        default='0.1,30,1,30,0.1,8,03',
                        help='When --check_parameters=True, it compares the input galaxy parameters with their'
                             'corresponding ones from SExtractor using these limits')
    parser.add_argument('--error_comparison_key', type=str, action='store', default='mean_error',
                        help='Key for error comparison')

    args = parser.parse_args(args)

    return args
