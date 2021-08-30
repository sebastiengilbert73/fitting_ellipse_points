import argparse
import logging
import ast
import os
import random
import pandas
import cmath

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
    pointsFilepath,
    outputDirectory,
    polar_center
    ):
    logging.debug("convert_to_polar.py main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    # Read the file
    points_df = pandas.read_csv(pointsFilepath)

    output_filepath = os.path.join(outputDirectory, os.path.basename(pointsFilepath)[:-4] + "_polar_({},{}).csv".format(polar_center[0], polar_center[1]))
    with open(output_filepath, 'w+') as polar_points_file:
        polar_points_file.write("phi,rho\n")
        for row in points_df.itertuples():
            shifted_p = (row.x - polar_center[0], polar_center[1] - row.y)  # Inverse the y axis
            rho, phi = cmath.polar(complex(shifted_p[0], shifted_p[1]))
            polar_points_file.write("{},{}\n".format(phi, rho))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pointsFilepath', help="The filepath to the csv file containing the 2D points")
    parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
    parser.add_argument('--polarCenter', help="The center for the polar coordinates. Default: '(512, 512)'",
                        default='(512, 512)')
    args = parser.parse_args()

    random.seed(args.randomSeed)
    polar_center = ast.literal_eval(args.polarCenter)

    main(args.pointsFilepath,
         args.outputDirectory,
         polar_center)