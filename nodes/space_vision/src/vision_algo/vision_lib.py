"""
Module that contains all the functions related to the detection of the target with the camera and computing its range, azimuth, elevation and pose

Author: Gaetan Ramet
License: TBD

"""

import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import mixture


def remove_green(image):
    """Return image with black pixels instead of green. If the image is in grayscale, does not do anything.
    The image is converted to HSV to remove the green component.

        Args:
            image : an RGB image or grayscale image as a numpy array

        Returns:
            image_no_green : an image of the same format as input with black instead of green
    """

    if(len(image.shape) <3):
        print("Warning : received image is grayscale")
        return image.copy()
    else:
        hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        #mask = cv2.inRange(hsv_img,(30,0,0), (90,255,255))
        mask = cv2.inRange(hsv_img,(35,0,0), (90,255,255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
        mask = mask.astype(bool)

        #print(mask)
        image_no_green = image.copy()
        image_no_green[mask] = (0,0,0)

        #cv2.imshow("image_no_green",image_no_green)
        #cv2.waitKey(0)
        return image_no_green


def hist_eq(image):
    """Return image with equalize histogram

        Args:
            image : A grayscale image as a numpy array

        Returns:
            eq_img : an image of the same format as input with equalized histogram"""

    eq_img = cv2.equalizeHist(image)

    return eq_img


def find_main_axes(angles, rhos, mode='fast'):
    """ Use the angles from the Hough lines detection to find two axis of the cube. If mode=='gmm', the angles are fit
    using a 2 parameters GMM. If mode=='fast', the two angles that are the most perpendicular together are kept

        Args:
            angles : The angles returned by cv2.Houghlines ( out = cv2.Houghliines(...), angles = out[:,:,1])
            mode : either 'fast' or 'gmm'

        Returns:
            main_dirs : The angles of the two axis detected"""

    if mode=='gmm':
        best_bic = np.inf
        data = np.concatenate((angles, rhos), axis=1)
        for n_comp in range(2, 7):

            g = mixture.GaussianMixture(n_components=n_comp)

            g.fit(data)

            if g.bic(data) < best_bic:
                best_bic = g.bic(data)

                means = g.means_

        return means

    if mode =='fast':
        angles_diff = np.zeros((len(angles), len(angles)))

        for row, angle in enumerate(angles):
            angles_diff[row] = np.squeeze(np.abs(np.abs(angles - angle) - np.pi/2))

        main_dirs_idx = np.unravel_index(angles_diff.argmin(), angles_diff.shape)
        main_dirs = np.asarray((angles[main_dirs_idx[0]], angles[main_dirs_idx[1]]))

        return main_dirs


def find_vertex(img_thresh, m, h, x_inter, y_inter):
    """Find a vertex in the cube from one vertex and the line parameters (m,h).
    The algorithm will gather the points of the cube that are the closest to the given line, and then find
    the one amongst them that is the furthest from the initial vertex.

        Args:
            img_thresh: A BW image of the cube, preferably after a Canny edge detection
            m: The slope of the line
            h: The y-intersect of the line
            x_inter: The X position of the initial vertex
            y_inter: The Y position of the initial vertex

        Returns:
            cube_px_x: The X position of the new vertex
            cube_px_y: The Y position of the new vertex
    """

    cube_px_y, cube_px_x = np.nonzero(img_thresh)
    #temp = img_thresh
    dist_px = np.sqrt(np.square(cube_px_x - x_inter) + np.square(cube_px_y - y_inter))
    dist_line = np.sqrt((h + m * cube_px_x - cube_px_y) ** 2 / (m ** 2 + 1))

    thresh = 3
    num_close = 0
    while num_close < 50:
        num_close = np.sum(dist_line < thresh)
        thresh += 1

    closest_to_line = np.argsort(dist_line)[:num_close]
    #if mode=='debug':
    #    for pt in closest_to_line:
    #        temp = cv2.circle(temp, (cube_px_x[pt], cube_px_y[pt]), 10, 80)

    max_d = np.max(dist_px[closest_to_line])

    further_from_inter = np.argmin(np.absolute(dist_px - max_d))

    return cube_px_x[further_from_inter], cube_px_y[further_from_inter]


def find_cube_vertices(image, img_thresh, dirs, rhos, mode='debug'):
    """Find 3 vertices of the target based on two lines and a thresholded (preferably Canny edge detection) of an image

        Args:
            image: The image to draw on
            img_thresh: A thresholded or preferably a Canny edge detection of image
            dirs: The two angles returned by find_main_axes
            rhos: The associated rhos from the two angles
            mode: Either 'debug' or 'test'. Regulates the number of figures generated

        Returns:
            v_inter: The vertex at the intersection of the two lines
            v0: The vertex at the other end of line 0
            v1: The vertex at the other end of line 1
            image_vert: The image with the vertices drawn
            """

    if np.sin(dirs[0])==0 or np.sin(dirs[1])==0:
        print("Division by 0 incoming, aborting")

        return 0,0,0,None

    m0 = -np.cos(dirs[0])/np.sin(dirs[0])
    h0 = rhos[0]/np.sin(dirs[0])

    m1 = -np.cos(dirs[1])/np.sin(dirs[1])
    h1 = rhos[1]/np.sin(dirs[1])

    x_inter = int((h1-h0)/(m0-m1))
    y_inter = int(m0*x_inter+h0)

    v0_x, v0_y = find_vertex(img_thresh, m0, h0, x_inter, y_inter)
    v1_x, v1_y = find_vertex(img_thresh, m1, h1, x_inter, y_inter)

    image_vert = image
    if mode=='debug' or mode =='test':
        image_vert = cv2.circle(image_vert, (v1_x, v1_y), 10, (0,255,255))
        image_vert = cv2.circle(image_vert, (v0_x, v0_y), 10, (0,255,255))
        image_vert = cv2.circle(image_vert, (x_inter, y_inter), 10, (0,255,255))
        if mode=='debug':
            cv2.imshow("vertices", image_vert)
    v0 = (v0_x, v0_y)
    v1 = (v1_x, v1_y)
    v_inter = (x_inter, y_inter)

    return v_inter, v0, v1, image_vert


def find_center_of_mass(bw_image):
    """Compute the center of mass of the blob in a BW image

        Args:
            bw_image: a BW image

        Returns:
            y_ : The Y position of the center of mass
            x_ : The X position of the center of mass
    """

    non_zero = np.where(bw_image != 0)

    y_ = int(round(np.mean(non_zero[0]))) # mean value of row
    x_ = int(round(np.mean(non_zero[1]))) # mean value of col

    return y_, x_


def coo_to_azim_elev(x, y, K):
    """Compute the azimuth and elevation angle (in degrees) of a point in image, requires the intrinsic camera matrix K

        Args:
            x: The X position of a pixel in the image
            y: The Y position of a pixel in the image
            K: The intrinsic matrix of the camera

        Returns:
            azim: The azimuth in degree
            elev: The elevation in degree
            """

    P = np.asarray((x, y, 1)).T
    P_ics = np.linalg.inv(K).dot(P)
    azim = P_ics[0]*180/np.pi
    elev = P_ics[1]*180/np.pi

    return azim, elev


def compute_edge_len(rvec, tvec, K, dist):
    """return the approximate length of the edge of the cube in #pixel.
     The function undo the rotation along the z-axis so that the face of the cube would be parallel to the image plane,
     and then measure the length of the edges along the x and y axis.

        Args:
            rvec: The rotation vector returned by cv2.solvePnP
            tvec: The translation vector returned by cv2.solvePnP
            K: The intrinsic camera matrix
            dist: The distortion coefficients of the camera if available. If no distortion is assumed, use []

        Returns:
            dx: The distance estimated using the x coordinates
            dy: The distance estimated using the y coordinate
            """

    undo_z = np.asarray([[0], [0], [rvec[2]]]) - rvec
    R, _ = cv2.Rodrigues(undo_z)

    p1_3d = R.dot(np.asarray([[0.0], [0.0], [0.0]]))
    p2_3d = R.dot(np.asarray([[0.0], [1.0], [0.0]]))
    p3_3d = R.dot(np.asarray([[1.0], [0.0], [0.0]]))

    p_3d = p1_3d
    p_3d = np.reshape(np.append(p_3d,(p2_3d, p3_3d)).astype(float), (3, 3))
    p_2d, _ = cv2.projectPoints(p_3d, rvec, tvec, K , dist)

    dy = np.sqrt((p_2d[0, 0, 0]-p_2d[1, 0, 0])**2 + (p_2d[0, 0, 1]-p_2d[1, 0, 1])**2)
    dx = np.sqrt((p_2d[0, 0, 0]-p_2d[2, 0, 0])**2 + (p_2d[0, 0, 1]-p_2d[2, 0, 1])**2)

    return dx, dy


def compute_range(tvec, rvec, K, dist, method='fast'):
    """Computes the range of the cube vs the camera (in meters).
     'fast' method is based on the translation vector and returns the distance to the center of the cube (default)
     'simtri' method is more computationally expensive and returns the average distance to the detected face

        Args:
            tvec: The translation vector returned by cv2.solvePnP
            rvec: The rotation vector returned by cv2.solvePnP
            K: The intrinsic camera matrix
            dist: The distortion coefficients of the camera if available. If no distortion is assumed, use []
            method: either 'fast' or 'simtri'

        Returns:
            range: The distance from the camera to the target
         """

    if method == 'fast':
        rmat, _ = cv2.Rodrigues(rvec)
        rmat = np.asarray(rmat)

        return 0.1*np.linalg.norm(rmat*np.expand_dims(np.array((0.5, 0.5, 0.5)), axis=1).T + tvec)

        #return np.linalg.norm((tvec + 0.5)/10) #old version

    elif method == 'simtri':
        dx, dy = compute_edge_len(rvec, tvec, K, dist)

        range_x = K[0, 0]*0.1/dx
        range_y = K[1, 1]*0.1/dy
        return (range_x + range_y)/2


def rotmat_to_quaternion(rmat):
    """Computes the quaternion (qw, qx, qy, qz) from a rotation matrix

        Args:
            rmat : A rotation matrix

        Returns:
            Q: A Quaternion (qw, qx, qy, qz)
            """

    Q = np.zeros(4, dtype=float)
    Q[0] = 0.5 * np.sqrt(rmat[0,0] + rmat[1,1] + rmat[2,2]+1)
    Q[1] = (rmat[2,1] - rmat[1,2])/(4*Q[0])
    Q[2] = (rmat[0,2] - rmat[2,0])/(4*Q[0])
    Q[3] = (rmat[1,0] - rmat[0,1])/(4*Q[0])

    return Q


def rotvec_to_quaternion(rvec):
    """compute the quaternion (qw, qx, qy, qz) from a rotation vector

    Args:
        rvec: A rotation vector from cv2.solvePnP

    Returns:
        Q: A Quaternin (qw, qx, qy, qz)
        """

    Q = np.zeros(4, dtype=float)
    angle = np.linalg.norm(rvec)
    ax = rvec/angle

    Q[0] = np.cos(angle/2)
    Q[1] = ax[0]*np.sin(angle/2)
    Q[2] = ax[1]*np.sin(angle/2)
    Q[3] = ax[2]*np.sin(angle/2)

    return Q


def solve_projection(p1, p2, p3, K, dist, image, image_thresh, last_position, mode='debug'):
    """Solve the projection from the 3 cube vertices and find the other ones in the image. This function uses the last
    position of the target to perform continuous tracking.

        Args:
            p1: The vertex at the intersection of the two axis
            p2: One vertex of the cube
            p3: One other vertex of the cube ((p1,p2,p3) should define a face of the cube)
            K: The intrinsic camera matrix
            dist: The distortion coefficeints of the camera if available. If no distortion is assumed, use []
            image: The image to draw on
            image_thresh: A thresholded version of the image
            last_position: The last position of the cube. If last position is (0,0) assumes it is unknown
            mode: Either 'debug' or 'test'. Regulates the amount of figures

        Returns:
            range: The range between the target and the camera
            azim: The azimuth of the target (in degrees)
            elev: The elevation of the target (in degrees)
            quat: The rotation quaternion of the target
            cm_pos: The position of the center of mass on the image
            image: The processed image
    """
    p1_3d = (0.0, 0.0, 0.0)
    p2_3d = (0.0, 1.0, 0.0)
    p3_3d = (1.0, 0.0, 0.0)
    p12_3d = (0.0, 0.5, 0.0)
    p13_3d = (0.5, 0.0, 0.0)

    # These points are the 3 3D vertices of the cube that we assume are known. p12 and p13 are on the middle of the edges

    p12_temp = (np.asarray(p1)+np.asarray(p2))/2
    p12 = (int(p12_temp[0]), int(p12_temp[1]))

    p13_temp = (np.asarray(p1)+np.asarray(p3))/2
    p13 = (int(p13_temp[0]), int(p13_temp[1]))
    retval, rvec, tvec = cv2.solvePnP(np.asarray((p1_3d, p2_3d, p3_3d, p12_3d, p13_3d), dtype=float), np.asarray((p1,p2,p3, p12, p13),dtype=float), K, dist)

    range_mean = compute_range(tvec, rvec, K, dist)

    # After solving the projection, we project the unknown 3D points

    p4_3d = (0.0, 0.0, 1.0)
    p5_3d = (1.0, 1.0, 0.0)
    p6_3d = (0.0, 1.0, 1.0)
    p7_3d = (1.0, 0.0, 1.0)
    p8_3d = (1.0, 1.0, 1.0)
    cm_3d = (0.5, 0.5, 0.5)

    P_3d = np.asarray((p4_3d, p5_3d, p6_3d, p7_3d, p8_3d, cm_3d), dtype=float)
    projected_points, _ = cv2.projectPoints(P_3d, rvec, tvec, K, dist)

    cm_pos = (int(projected_points[5, 0, 0]), int(projected_points[5, 0, 1]))

    cube_points = np.concatenate((np.reshape(projected_points,(6,2)),np.expand_dims(np.asarray(p1), axis=1).T, np.expand_dims(np.asarray(p2), axis=1).T, np.expand_dims(np.asarray(p3), axis=1).T)).astype(np.float32)

    # We use convex hull of projected points for continuous tracking

    conv = cv2.convexHull(cube_points)
    conv = conv.astype(int)
    image_temp = image.copy()
    image_temp = cv2.polylines(image_temp, [conv], True, (0, 255, 255), 1)
    hull_mask = cv2.fillConvexPoly(np.zeros(image.shape), conv, (0, 255, 255), 1)
    hull_mask = hull_mask[:,:,2]

    thresh_diff_img = image_thresh - np.multiply(image_thresh/255,hull_mask)
    thresh_diff = np.sum(thresh_diff_img/255)
    if mode=='debug':
        print(np.sum(thresh_diff))
        print(np.sum(image_thresh/255))
        cv2.imshow("convex hull", image_temp)

    azim, elev = coo_to_azim_elev(cm_pos[0], cm_pos[1], K)

    pos_diff = np.linalg.norm(np.asarray(cm_pos) - np.asarray(last_position))

    # Continuous tracking : if the new position is not good enough, try the other projection
    if (np.sum(np.asarray(last_position) == [0,0])) and (np.sum(np.asarray(last_position) != [0,0]) and (pos_diff > 40)) or (thresh_diff/np.sum(image_thresh/255) > 0.01):

        p12_temp = (np.asarray(p1) + np.asarray(p3)) / 2
        p12 = (int(p12_temp[0]), int(p12_temp[1]))

        p13_temp = (np.asarray(p1) + np.asarray(p2)) / 2
        p13 = (int(p13_temp[0]), int(p13_temp[1]))
        retval, rvec2, tvec2 = cv2.solvePnP(np.asarray((p1_3d, p2_3d, p3_3d, p12_3d, p13_3d), dtype=float),
                                          np.asarray((p1, p3, p2, p12, p13), dtype=float), K, dist)

        range_mean2 = compute_range(tvec2, rvec2, K, dist)

        projected_points2, _ = cv2.projectPoints(P_3d, rvec2, tvec2, K, dist)

        cm_pos2 = (int(projected_points2[5, 0, 0]), int(projected_points2[5, 0, 1]))

        cube_points2 = np.concatenate((np.reshape(projected_points2, (6, 2)), np.expand_dims(np.asarray(p1), axis=1).T,
                                      np.expand_dims(np.asarray(p2), axis=1).T,
                                      np.expand_dims(np.asarray(p3), axis=1).T)).astype(np.float32)

        conv2 = cv2.convexHull(cube_points2)
        conv2 = conv2.astype(int)
        image_temp2 = image.copy()
        image_temp2 = cv2.polylines(image_temp2, [conv2], True, (0, 255, 255), 1)
        hull_mask2 = cv2.fillConvexPoly(np.zeros(image.shape), conv2, (0, 255, 255), 1)
        hull_mask2 = hull_mask2[:, :, 2]

        thresh_diff_img2 = image_thresh - np.multiply(image_thresh / 255, hull_mask2)
        thresh_diff2 = np.sum(thresh_diff_img2 / 255)
        if mode=='debug':
            print(np.sum(thresh_diff2))
            print(np.sum(image_thresh / 255))
            cv2.imshow("convex hull 2", thresh_diff_img2)

        azim2, elev2 = coo_to_azim_elev(int(projected_points2[5, 0, 0]), int(projected_points2[5, 0, 1]), K)

        pos_diff2 = np.linalg.norm(np.asarray(cm_pos2) - np.asarray(last_position))
        if np.sum(np.asarray(last_position) != [0, 0]) and (pos_diff2 > 40):
            print('Warning : cube is moving very fast or a bad position was computed')
            print('pos diff :', pos_diff)

        # We compare the quality of the two projections and keep the best one
        if ((pos_diff2 < pos_diff) and (np.sum(np.asarray(last_position) != [0,0]))) or thresh_diff2 < thresh_diff:
            print('Cube was inverted to keep coherent position')
            azim = azim2
            elev = elev2
            projected_points = projected_points2
            range_mean = range_mean2
            cm_pos = cm_pos2
            rvec = rvec2
            image_temp = image_temp2

    image = image_temp
    if mode=='debug' or mode=='test':
        for point in projected_points:

            image = cv2.circle(image,(int(point[0, 0]), int(point[0, 1])), 10, (0, 255, 0))

        image = cv2.circle(image, cm_pos,10,(0,0,255))
        image = cv2.circle(image, last_position,10,(0,0,255))
        image = cv2.circle(image, p12, 10, (0, 255, 255))
        image = cv2.circle(image, p13, 10, (0, 255, 255))
        image = cv2.line(image, p1, p2, (0, 0, 255), thickness=5)
        image = cv2.line(image, p1, p3, (0, 255, 0), thickness=5)
        image = cv2.line(image, p1, (int(projected_points[0, 0, 0]), int(projected_points[0, 0, 1])), (255, 0, 0), thickness=5)

        if mode=='debug':
          cv2.imshow("projected points", image)

    quat = rotvec_to_quaternion(rvec)

    return range_mean, azim, elev, quat, cm_pos, image


def img_analysis(image, last_position, mode='debug', K_mat=None, dist_mat=None):
    """Performs the whole image analysis process to compute the range, azimuth, elevation and pose of the target from
    an image. The image is first processed by removing the green background if it is RGB. Then it is converted to grayscale,
    thresholded and we perform a small opening to remove noise. We then apply a Canny edge detector and keep only
    the outer edges detected. We apply the cv2.Houghlines function to this canny edge detection to find the edges.
    We then apply find_main_axes, find_cube_vertices and solve_projection to reconstruct the target.

        Args:
            image: An RGB or Grayscale image
            last_position: The last position of the target on the image. If it is (0,0), assumes it is unknown
            mode: Either 'debug' or 'test'. Regulates the amount of figures
        Returns:
            range: The range between the camera and the target
            azim: The azimuth of the target (in degrees)
            elev: The elevation of the target (in degrees)
            quat: The rotation quaternion of the target
            cm_point: The position of the center of mass of the target
            image_with_centerpoint: The image with the reconstructed vertices
            cube_found: A boolean, True if everything went fine
    """

    # Preprocessing
    image_no_green = remove_green(image)

    gray_img = cv2.cvtColor(image_no_green, cv2.COLOR_BGR2GRAY)

    eq_img = hist_eq(gray_img)
    eq_img = gray_img
    ret, eq_img_thresh = cv2.threshold(eq_img, 18, 255, cv2.THRESH_BINARY)
    eq_img_thresh = cv2.morphologyEx(eq_img_thresh, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

    im_filled = eq_img_thresh.copy()
    h,w = eq_img_thresh.shape[:2]
    mask = np.zeros((h+2,w+2), np.uint8)

    im_filled = cv2.floodFill(im_filled,mask,(0,0), 255)
    inv_im_filled = 255 - im_filled[1]

    eq_img_thresh = eq_img_thresh + inv_im_filled

    if mode=='debug':
        cv2.imshow('original_img', image)
        cv2.imshow('gray_img', gray_img)
        cv2.imshow('eq_img', eq_img)
        cv2.imshow('eq_img_thresh', eq_img_thresh)

    canny = cv2.Canny(eq_img_thresh.copy(), 30, 200)
    outer_canny = np.zeros(canny.shape, dtype=np.uint8)

    # Keep only the outer part of edge map

    for row_num, row in enumerate(canny):
        if sum(row):
            left_border = np.argmax(row)
            right_border = np.argmax(np.flip(row,0))

            outer_canny[row_num,left_border] = 255
            outer_canny[row_num, -right_border] =255

    for col_num, col in enumerate(canny.T):
        if sum(col):
            left_border = np.argmax(col)
            right_border = np.argmax(np.flip(col,0))

            outer_canny[left_border, col_num] = 255
            outer_canny[-right_border, col_num] =255

    if mode=='debug':
        cv2.imshow("canny", canny)
        cv2.imshow("outer_canny", outer_canny)

    # Find lines in edge map
    lines = cv2.HoughLines(outer_canny, 1, np.pi/180, 40)

    if lines is not None:
        if mode=='debug' or mode=='test':
            lines_img = image.copy()
            for line in lines:
                rho = line[:,0]
                theta = line[:,1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))

                cv2.line(lines_img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("many_houghlines", lines_img)

        angles = lines[:, :, 1]
        rhos = lines[:, :, 0]

        if mode=='debug':
            plt.figure()
            plt.scatter(rhos,angles, marker='x', linewidths=0.1)
            plt.title(r'Hough Lines parameters : $\rho$ vs $\theta$')
            plt.xlabel(r'$\rho$ [px]')
            plt.ylabel(r'$\theta$ [rad]')
            plt.show()

        axes_mode = 'gmm'
        main_dirs = find_main_axes(angles, rhos, mode=axes_mode)

        # The output of find_main_axis is different depending of the mode, so we handle each case separately.
        # In both cases, we want to keep only two couples rho/theta, i.e. two lines

        if axes_mode =='fast':
            main_rhos = np.zeros(main_dirs.shape)
            for cnt, direction in enumerate(main_dirs):

                angle_diff = angles - direction

                main_rhos[cnt] = rhos[np.argmin(np.absolute(angle_diff))]

        elif axes_mode =='gmm':
            main_rhos = main_dirs[:,1]
            main_dirs = main_dirs[:,0]

            cnt = main_dirs.shape[0]

            dil_img = cv2.morphologyEx(eq_img_thresh, cv2.MORPH_DILATE, kernel=np.ones((5, 5), np.uint8))

            stop=False
            for i in range(cnt):
                for j in range(1,cnt):
                    if i==j:
                        continue
                    else:

                        if np.sin(main_dirs[i]) == 0 or np.sin(main_dirs[j]) == 0:
                            print("Division by 0 incoming, skipping this line")
                            continue

                        m0 = -np.cos(main_dirs[i]) / np.sin(main_dirs[i])
                        h0 = main_rhos[i] / np.sin(main_dirs[i])

                        m1 = -np.cos(main_dirs[j]) / np.sin(main_dirs[j])
                        h1 = main_rhos[j] / np.sin(main_dirs[j])

                        x_inter = int((h1 - h0) / (m0 - m1))
                        y_inter = int(m0 * x_inter + h0)

                        if x_inter >0 and y_inter >0 and x_inter < eq_img_thresh.shape[1] and y_inter < eq_img_thresh.shape[0]:
                            if dil_img[y_inter,x_inter] !=0:
                                print('found ok vertex')
                                stop=True
                        if stop:
                            print(i)
                            print(j)
                            break
                if stop:
                    break

            dir0 = main_dirs[i]
            rho0 = main_rhos[i]
            dir1 = main_dirs[j]
            rho1 = main_rhos[j]

            main_dirs = np.asarray((dir0, dir1))
            main_rhos = np.asarray((rho0, rho1))
            # main_dirs and main_rhos are the parameters of the two lines of choice

        if mode=='debug' or mode=='test':
            for i in range(2):
                rho = main_rhos[i]
                theta = main_dirs[i]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))

                cv2.line(image, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

            if mode=='debug':
                cv2.imshow("houghlines", image)

    else:
        print("no lines found, reduce threshold")
        return 0,0,0,0,(0,0),[], False

    if mode=='debug':
        cv2.imshow('thresholded image', eq_img_thresh)

    v0,v1,v2,image = find_cube_vertices(image, canny, main_dirs, main_rhos, mode)
    if image is None:
        print("aborting")
        return 0,0,0,0,(0,0),[], False


    if K_mat is None:
        # Default matrix parameters
        foc_mm = 12
        sens_w = 6.9 #Pointgrey
        sens_h = 5.5
        #sens_w = 11.3 # Basler
        #sens_h= 11.3
        f_x = foc_mm/sens_w*image.shape[1]
        f_y = foc_mm/sens_h*image.shape[0]
        K = np.array([[f_x, 0, image.shape[1]/2],[0, f_y, image.shape[0]/2],[0,0,1]])
    else:
        K = np.array(eval(K_mat))

    if dist_mat is None:
        # Default distortion coefficients (no distortion)
        dist = np.asarray([])
    else:
        dist = np.array(eval(dist_mat))

    range_mean, azim, elev, quat, cm_pos, image_proc = solve_projection(v0,v1,v2,K, dist, image, eq_img_thresh, last_position, mode)

    if mode=='debug' or mode=='test':
        image_with_centerpoint = cv2.circle(image_proc,(image.shape[1]/2, image.shape[0]/2), 15, (255, 0 ,0))

        cv2.imshow("circle2", image_with_centerpoint)

        print("Results:")
        print("Range : {:f} m".format(range_mean))
        print("Azimuth : {:f} deg, Elevation : {:f} deg".format(azim, elev))
        print("Rotation quaternion : {:f} {:f} {:f} {:f}".format(quat[0], quat[1], quat[2], quat[3]))

        if mode=='debug':
            cv2.waitKey(0)
        else:
            cv2.waitKey(50)

    return range_mean, azim, elev, quat, cm_pos, image_with_centerpoint, True
