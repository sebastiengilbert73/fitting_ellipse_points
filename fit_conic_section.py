import argparse
import logging
import ast
import os
import random
import pandas
import math
import ransac.core as ransac
from conic_section import ConicSection
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
    conic_section = ConicSection()
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