import argparse
import logging
import ast
import os
import random
import pandas
import wrap_around_polynomial
import math
import ransac.core as ransac
import conic_section

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
        phi = row.phi
        if phi < 0:
            phi += 2 * math.pi
        rho = row.rho

        xy_tuples.append((phi/(2 * math.pi), rho))

    #wrap_poly = wrap_around_polynomial.WrapAroundPolynomial(16)
    #wrap_poly.Create(xy_tuples)

    """with open(os.path.join(outputDirectory, "fitted_alpha_rho.csv"), 'w+') as fit_file:
        fit_file.write("alpha,rho,fitted_rho\n")
        for (alpha, rho) in xy_tuples:
            evaluation = wrap_poly.Evaluate(alpha)
            print ("{}, {}, {}\n".format(alpha, rho, evaluation))
            fit_file.write("{},{},{}\n".format(alpha, rho, evaluation))

    with open(os.path.join(outputDirectory, "poly_0_1.csv"), 'w+') as poly_file:
        poly_file.write("alpha,r\n")
        for alphaNdx in range(101):
            alpha = alphaNdx/100
            r = wrap_poly.Evaluate(alpha)
            poly_file.write("{},{}\n".format(alpha, r))
    """

    """wrap_poly_modeller = ransac.Modeler(model_class=wrap_around_polynomial.WrapAroundPolynomial,
                                        number_of_trials=1000, acceptable_error=5)
    consensus_poly, inliers, outliers = wrap_poly_modeller.ConsensusModel(xy_tuples, degree=16)
    with open(os.path.join(outputDirectory, "ransac_alpha_rho.csv"), 'w+') as fit_file:
        fit_file.write("alpha,rho,fitted_rho\n")
        for (alpha, rho) in xy_tuples:
            evaluation = consensus_poly.Evaluate(alpha)
            print ("{}, {}, {}\n".format(alpha, rho, evaluation))
            fit_file.write("{},{},{}\n".format(alpha, rho, evaluation))
    """



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