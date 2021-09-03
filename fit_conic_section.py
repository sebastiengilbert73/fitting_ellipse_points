import argparse
import logging
import ast
import os
import random
import pandas
import math
import ransac.core as ransac
from ransac.models.conic_section import ConicSection
import cv2
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')

def main(
        pointsFilepath,
        outputDirectory,
        imageSizeHW
        ):
    logging.info("fit_conic_section.py: main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    xy_tuples = []
    xy_df = pandas.read_csv(pointsFilepath)
    for row in xy_df.itertuples():
        x = row.x
        y = row.y
        xy_tuples.append( ((x, y), 0) )
    logging.debug("len(xy_tuples) = {}".format(len(xy_tuples)))
    """conic_section = ConicSection()
    conic_section.Create(xy_tuples)
    conic_section_type = conic_section.ConicSectionType()
    logging.info("conic_section_type = {}".format(conic_section_type))

    # Create an annotated image
    image = np.zeros((imageSizeHW[0], imageSizeHW[1], 3), dtype=np.uint8)
    for ((x, y), d) in xy_tuples:
        image[y, x, 0] = 255
    threshold = 0.0001
    for y in range(imageSizeHW[0]):
        for x in range(imageSizeHW[1]):
            if abs(conic_section.Evaluate((x, y))) < threshold:
                image[y, x, 1] = 255

    center = conic_section.Center()
    logging.debug("center = {}".format(center))

    center, a, b, theta = conic_section.EllipseParameters()
    logging.debug("center = {}; a = {}; b = {}; theta = {}".format(center, a, b, theta))
    """

    modeller = ransac.Modeler(ConicSection, number_of_trials=300, acceptable_error=5)
    conic_section, inliers, outliers = modeller.ConsensusModel(xy_tuples)
    logging.debug("len(inliers) = {}; len(outliers) = {}".format(len(inliers), len(outliers)))
    #center, a, b, theta = conic_section.EllipseParameters()
    #logging.debug("center = {}; a = {}; b = {}; theta = {}".format(center, a, b, theta))

    # Create an annotated image
    image = np.zeros((imageSizeHW[0], imageSizeHW[1], 3), dtype=np.uint8)
    for ((x, y), d) in xy_tuples:
        image[y, x, 0] = 255
    threshold = 0.0001
    """for y in range(imageSizeHW[0]):
        for x in range(imageSizeHW[1]):
            if abs(conic_section.Evaluate((x, y))) < threshold:
                image[y, x, 1] = 255
    """
    for outlier in outliers:
        image[outlier[0][1], outlier[0][0], :] = (0, 0, 255)
        #logging.debug("outlier = {}".format(outlier))

    ellipse_points = conic_section.EllipsePoints(rounding=True)
    logging.debug("len(ellipse_points) = {}".format(len(ellipse_points)))
    for ellipse_pt_ndx in range(0, len(ellipse_points) - 1):
        p1 = ellipse_points[ellipse_pt_ndx]
        p2 = ellipse_points[ellipse_pt_ndx + 1]
        cv2.line(image, p1, p2, (0, 255, 255))
    cv2.line(image, ellipse_points[0], ellipse_points[-1], (0, 255, 255))

    image_filepath = os.path.join(outputDirectory, "fitConicSection_main_annotated.png")
    cv2.imwrite(image_filepath, image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pointsFilepath', help="The filepath to the csv file containing the (x, y) points")
    parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
    parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
    parser.add_argument('--imageSizeHW', help="The size (height, width) of the output image. Default: '(1024, 1024)'", default='(1024, 1024)')
    args = parser.parse_args()

    random.seed(args.randomSeed)
    imageSizeHW = ast.literal_eval(args.imageSizeHW)
    main(
        args.pointsFilepath,
        args.outputDirectory,
        imageSizeHW
    )