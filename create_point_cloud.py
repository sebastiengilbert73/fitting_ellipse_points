import argparse
import logging
import numpy as np
import ast
import os
import cv2
import random
import math

parser = argparse.ArgumentParser()
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
parser.add_argument('--randomSeed', help="The seed for the random module. Default: 0", type=int, default=0)
parser.add_argument('--imageSizeHW', help="The image size (height, width). Default: '(1024, 1024)'", default='(1024, 1024)')
parser.add_argument('--ellipseCenter', help="The ellipse center. Default: '(400, 600)'", default='(400, 600)')
parser.add_argument('--ellipseLongHalfAxis', help="The ellipse long half axis. Default: 260", type=float, default=260)
parser.add_argument('--ellipseShortHalfAxis', help="The ellipse short half axis. Default: 100", type=float, default=100)
parser.add_argument('--ellipseAngle', help="The ellipse angle of the main axis with respect to the horizontal, in radians. Default: 0.8", type=float, default=0.8)
parser.add_argument('--ellipseNoise', help="The 2D noise amplitude to add to the ellipse points. Default: 6.0", type=float, default=6.0)
parser.add_argument('--numberOfPoints', help="The number of points in the cloud. Default: 1000", type=int, default=1000)
parser.add_argument('--outliersRatio', help="The proportion of outliers. Default: 0.5", type=float, default=0.5)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s [%(levelname)s] %(message)s')
image_sizeHW = ast.literal_eval(args.imageSizeHW)
ellipse_center = ast.literal_eval(args.ellipseCenter)
random.seed(args.randomSeed)

def main():
    logging.info("create_point_cloud.py main()")
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    image = np.zeros((image_sizeHW[0], image_sizeHW[1], 3), dtype=np.uint8)
    points_list = []
    # Create points from the ellipse
    while len(points_list) < round( (1 - args.outliersRatio) * args.numberOfPoints):
        # Generate a point on the unit circle
        theta = 2 * math.pi * random.random()
        unit_circle_pt = (math.cos(theta), math.sin(theta))
        # Squash the point according to the half-axes
        squashed_pt = (unit_circle_pt[0] * args.ellipseLongHalfAxis, unit_circle_pt[1] * args.ellipseShortHalfAxis)
        # Rotate the point
        rotated_pt = (squashed_pt[0] * math.cos(args.ellipseAngle) + squashed_pt[1] * math.sin(args.ellipseAngle),
                      -squashed_pt[0] * math.sin(args.ellipseAngle) + squashed_pt[1] * math.cos(args.ellipseAngle))
        # Shift
        shifted_pt = (rotated_pt[0] + ellipse_center[0], rotated_pt[1] + ellipse_center[1])
        # Noise
        noisy_pt = (shifted_pt[0] + args.ellipseNoise * random.random(), shifted_pt[1] + args.ellipseNoise * random.random())
        points_list.append((round(noisy_pt[0]), round(noisy_pt[1])))

    # Add random points
    while len(points_list) < args.numberOfPoints:
        random_pt = (image_sizeHW[1] * random.random(), image_sizeHW[0] * random.random())
        points_list.append((round(random_pt[0]), round(random_pt[1])))

    with open(os.path.join(args.outputDirectory, "createPointCloud_ellipse_{}_{}_{}.csv".format(args.ellipseLongHalfAxis, args.ellipseShortHalfAxis, args.ellipseAngle)), 'w+') as points_file:
        points_file.write("x,y\n")
        for point in points_list:
            if point[0] >= 0 and point[0] < image_sizeHW[1] and point[1] >= 0 and point[1] < image_sizeHW[0]:
                image[point[1], point[0]] = (0, 255, 0)
                points_file.write("{},{}\n".format(point[0], point[1]))

    image_filepath = os.path.join(args.outputDirectory, "createPointCloud_ellipse_{}_{}_{}.png".format(args.ellipseLongHalfAxis, args.ellipseShortHalfAxis, args.ellipseAngle))
    cv2.imwrite(image_filepath, image)


if __name__ == '__main__':
    main()
