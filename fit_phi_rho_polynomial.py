import argparse
import logging
import ast
import os
import random
import pandas
import ransac.models.polynomial as polynomial_model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
        polarPointsFilepath,
        outputDirectory
        ):
    logging.info("fit_phi_rho_polynomial.py main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Load the polar points
    polar_points_df = pandas.read_csv(polarPointsFilepath)
    xy_tuples = []
    for row in polar_points_df.itertuples():
        xy_tuples.append((row.phi, row.rho))

    # Create the modeller
    polynomial_modeller

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('polarPointsFilepath', help="The filepath to the csv file containing the (phi, rho) points")
    parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
    args = parser.parse_args()

    random.seed(args.randomSeed)
    main(
        args.polarPointsFilepath,
        args.outputDirectory
    )