import numpy as np
import cv2 as cv
import glob
from AutoCalib import *
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
width = 6
height = 9
size = 0.0215
Calibrator = AutoCalib(width, height, size)
images = glob.glob('./Calibration_Imgs/*.jpg')
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
Calibrator.find_chess_board_corners(images, criteria)
reproject_err =Calibrator.calibrate_camera()
Calibrator.un_distort(images, visualize=True)
print(reproject_err)
