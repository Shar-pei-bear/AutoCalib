import os.path

import cv2
import numpy as np
import cv2 as cv
from scipy.linalg import null_space, svd
from scipy.optimize import least_squares
import glob

class AutoCalib:
    def __init__(self, width, height, size):
        self.width = width
        self.height = height
        self.size = size

        self.cameraMatrix = np.zeros((3, 3))
        self.distCoeffs = np.zeros(2)

        self.fx = 0
        self.fy = 0

        self.cx = 0
        self.cy = 0

        self.k1 = 0
        self.k2 = 0

        self.r_vec = None
        self.t_vec = None

        # Arrays to store object points and image points from all the images.
        self.obj_points = [] # 3d point in real world space
        self.img_points = [] # 2d points in image plane.

        self.image_size = None

    def find_chess_board_corners(self, images, criteria, visualize=False):
        objp = np.zeros((self.width*self.height,3), np.float32)
        objp[:, :2] = np.mgrid[0:self.height, 0:self.width].T.reshape(-1,2)*self.size
        for frame_name in images:
            img = cv.imread(frame_name)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            if self.image_size is None:
                self.image_size = gray.shape[::-1]

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (self.height, self.width), None)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # If found, add object points, image points (after refining them)
            if ret:
                self.obj_points.append(objp)
                self.img_points.append(corners)
                # Draw and display the corners
                if visualize:
                    cv.drawChessboardCorners(img, (self.height, self.width), corners2, ret)
                    image_name = frame_name.split('.jpg')[0]
                    image_name = image_name.split(os.path.sep)[-1]
                    image_name = image_name + '_corners.jpg'
                    image_path = os.path.join('./Results/', image_name )
                    cv.imwrite(image_path, img)

    def calibrate_camera(self):
        n_images = len(self.obj_points)
        assert n_images
        self.r_vec = np.zeros((n_images, 3))
        self.t_vec =  np.zeros((n_images, 3))

        n_points = self.collect_calibration_data()
        reproject_err = self.cv_calibrate_camera2internal(n_points)

        return reproject_err

    def cv_calibrate_camera2internal(self, n_points):
        n_intrinsic = 6
        max_points = 0
        total = 0

        n_images = n_points.shape[0]
        n_params = n_intrinsic + n_images * 6

        for i in range(n_images):
            ni = n_points[i]
            assert ni >= 4
            max_points = max(max_points, ni)
            total+= ni

        allH = self.cv_init_intrinsic_params2d(n_points)
        self.cv_find_extrinsic_camera_params(allH)

        x0 = np.zeros(n_params)
        x0[0:6] = [self.fx, self.fy, self.cx, self.cy, self.k1, self.k2]
        for i in range(n_images):
            x0[i * 6 + 6: i * 6 + 9] = self.r_vec[i]
            x0[i * 6 + 9: i * 6 + 12] = self.t_vec[i]

        res = least_squares(self.reproject_error, x0, method='lm', xtol=np.finfo(dtype=np.float32).eps, max_nfev=30)
        self.fx, self.fy, self.cx, self.cy, self.k1, self.k2 = res.x[0:6]

        for i in range(n_images):
            self.r_vec[i] = res.x[i * 6 + 6: i * 6 + 9]
            self.t_vec[i] = res.x[i * 6 + 9: i * 6 + 12]

        self.cameraMatrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        print(res.x[0:6])
        return res.cost

    def cv_init_intrinsic_params2d(self, n_points):
        n_images = n_points.shape[0]
        V = np.zeros((2*n_images, 5))
        allH = np.zeros((n_images, 9))

        n_images = len(self.obj_points)
        for i in range(n_images):
            object_point = self.obj_points[i]
            image_point = self.img_points[i]

            H, mask = cv2.findHomography(srcPoints=object_point[:, 0:2], dstPoints=image_point)
            # calculate homography with respect to image center
            H[0, :] -= H[2, :]*(self.image_size[0] - 1)*0.5
            H[1, :] -= H[2, :]*(self.image_size[1] - 1)*0.5

            h1 = H[:, 0]
            h2 = H[:, 1]

            v12 = self.compute_v(h1, h2)
            v11 = self.compute_v(h1, h1)
            v22 = self.compute_v(h2, h2)

            V[2*i] = v12
            V[2*i+1] = v11 - v22
            # calculate homography with respect to image origin
            H[0, :] += H[2, :] * (self.image_size[0] - 1) * 0.5
            H[1, :] += H[2, :] * (self.image_size[1] - 1) * 0.5

            allH[i] = H.flatten()

        _, _, Vh = svd(V)
        b = Vh[-1, :]
        b11, b22, b13, b23, b33 = b

        self.cx = -b13/b11 + (self.image_size[0] - 1)*0.5
        self.cy = -b23/b22 + (self.image_size[1] - 1)*0.5

        scale = b33 - b23*b23/b22 + b13*b13/b11

        self.fx = np.sqrt(scale/b11)
        self.fy = np.sqrt(scale/b22)

        self.cameraMatrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        return allH

    def cv_find_extrinsic_camera_params(self, allH):
        for i in range(allH.shape[0]):
            h = allH[i]
            H = h.reshape((3, 3))

            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]

            r1 = np.linalg.inv(self.cameraMatrix)@h1
            r1_norm = np.linalg.norm(r1)
            r1 = r1/r1_norm

            r2 = np.linalg.inv(self.cameraMatrix)@h2
            r2_norm = np.linalg.norm(r2)
            r2 = r2/r2_norm

            t = np.linalg.inv(self.cameraMatrix)@h3
            t = 2*t/(r1_norm + r2_norm)

            r3 = np.cross(r1, r2)
            R = np.stack((r1, r2, r3), axis=1)
            if np.linalg.det(R) < 0:
                R = -R
                t = -t

            U, s, Vh= svd(R)
            R = U@Vh
            rvec, _ = cv.Rodrigues(R)

            self.r_vec[i] = rvec.squeeze()
            self.t_vec[i] = t.squeeze()

    @staticmethod
    def compute_v(hi, hj):
        return np.asarray([hi[0]*hj[0], hi[1] * hj[1], hi[0] * hj[2] + hi[2] * hj[0], hi[1] * hj[2] + hi[2] * hj[1], hi[2] * hj[2]])

    def collect_calibration_data(self):
        n_images = len(self.obj_points)
        assert n_images == len(self.img_points)
        n_points = np.zeros(n_images)
        for i in range(n_images):
            object_point = self.obj_points[i]
            number_of_object_points = object_point.shape[0]
            assert number_of_object_points

            image_point = self.img_points[i]
            number_of_image_points = image_point.shape[0]

            assert number_of_image_points
            assert number_of_image_points == number_of_object_points

            n_points[i] = number_of_object_points

        return n_points

    @staticmethod
    def project_points(object_points, r_vec, t_vec, fx, fy, cx, cy, k1, k2):
        R, _ = cv.Rodrigues(r_vec)
        object_points_i = R@object_points.T + t_vec.reshape((3, 1))

        x = object_points_i[0, :]
        y = object_points_i[1, :]
        z = object_points_i[2, :]

        x = x/z
        y = y/z

        r2 = np.square(x) + np.square(y)
        r4 = np.square(r2)

        cdist = 1 + k1 * r2 + k2 * r4

        xd = x * cdist
        yd = y * cdist

        u = xd*fx + cx
        v = yd*fy + cy

        return u, v

    def reproject_error(self, params):
        fx, fy, cx, cy, k1, k2 = params[0:6]
        n_images = len(self.obj_points)

        error = []
        for i in range(n_images):
            object_points = self.obj_points[i]
            image_points = self.img_points[i]

            r_vec = params[i*6 + 6: i*6+9]
            t_vec = params[i*6 + 9: i*6+12]

            u, v = self.project_points(object_points, r_vec, t_vec, fx, fy, cx, cy, k1, k2)
            error_per_view = np.concatenate((u - image_points[:, 0, 0], v - image_points[:, 0, 1]))
            error = np.concatenate((error, error_per_view))

        return error

    def un_distort(self, images, visualize=False):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.obj_points, self.img_points, self.image_size, None, None)
        print(ret)
        print(mtx)
        print(dist)

        for i, frame_name in enumerate(images):
            img = cv.imread(frame_name)

            v, u = np.mgrid[0:img.shape[0], 0:img.shape[1]]

            u = u.astype(np.float32)
            v = v.astype(np.float32)

            x = (u - self.cx)/self.fx
            y = (v - self.cy)/self.fy

            r2 = np.square(x) + np.square(y)
            r4 = np.square(r2)

            cdist = 1 + self.k1 * r2 + self.k2 * r4

            xd = x * cdist
            yd = y * cdist

            map_x = xd * self.fx + self.cx
            map_y = yd * self.fy + self.cy

            dst = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

            object_points = self.obj_points[i]

            r_vec = self.r_vec[i]
            t_vec = self.t_vec[i]

            uc, vc = self.project_points(object_points, r_vec, t_vec, self.fx, self.fy, self.cx, self.cy, 0, 0)
            key_points = [cv.KeyPoint(uc[i], vc[i], 15) for i in range(uc.shape[0])]
            cv2.drawKeypoints(dst, key_points, dst, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if visualize:
                image_name = frame_name.split('.jpg')[0]
                image_name = image_name.split(os.path.sep)[-1]
                image_name = image_name + '_rectify.jpg'
                image_path = os.path.join('./Results/', image_name)
                cv.imwrite(image_path, dst)

if __name__ == '__main__':
    sigmas = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]