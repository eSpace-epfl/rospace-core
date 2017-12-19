import sys, os
import numpy as np
import cv2
from sklearn import mixture
def hist_eq(image):
    """return image with equalize histogram"""
    eq_img = cv2.equalizeHist(image)

    return eq_img

def find_main_axes(angles, mode='fast'):
    """ Fit the directions of Hough lines to a GMM and return the main directions"""
    if mode=='gmm':
        g = mixture.GaussianMixture(n_components=2)

        g.fit(angles)

        return g.means_

    if mode =='fast':
        angles_diff = np.zeros((len(angles), len(angles)))

        for row, angle in enumerate(angles):
            angles_diff[row] = np.squeeze(np.abs(np.abs(angles - angle) - np.pi/2))

        main_dirs_idx = np.unravel_index(angles_diff.argmin(), angles_diff.shape)
        main_dirs =  np.asarray((angles[main_dirs_idx[0]], angles[main_dirs_idx[1]]))

        return main_dirs

def find_vertex(img_thresh, m, h, x_inter, y_inter, debug =True):
    """Find the vertices in the cube from the intersection point and the line parameters (m,h)"""
    cube_px_y, cube_px_x = np.nonzero(img_thresh)
    temp = img_thresh
    dist_px = np.sqrt(np.square(cube_px_x - x_inter) + np.square(cube_px_y - y_inter))
    dist_line = np.sqrt((h + m * cube_px_x - cube_px_y) ** 2 / (m ** 2 + 1))

    thresh = 3
    num_close = 0
    while num_close < 50:
        num_close = np.sum(dist_line < thresh)
        thresh += 1

    closest_to_line = np.argsort(dist_line)[:num_close]
    if debug:
        for pt in closest_to_line:
            temp = cv2.circle(temp, (cube_px_x[pt], cube_px_y[pt]), 10, 80)

    max_d = np.max(dist_px[closest_to_line])

    further_from_inter = np.argmin(abs(dist_px - max_d))

    return cube_px_x[further_from_inter], cube_px_y[further_from_inter]

def find_cube_vertices(image, img_thresh,dirs, rhos, debug=True):
    """Find the 3 vertices of the cubesat based on the houghlines"""

    m0 = -np.cos(dirs[0])/np.sin(dirs[0])
    h0 = rhos[0]/np.sin(dirs[0])

    m1 = -np.cos(dirs[1])/np.sin(dirs[1])
    h1 = rhos[1]/np.sin(dirs[1])

    x_inter = (h1-h0)/(m0-m1)
    y_inter = m0*x_inter+h0

    v0_x, v0_y = find_vertex(img_thresh, m0, h0, x_inter, y_inter, debug)
    v1_x, v1_y = find_vertex(img_thresh, m1, h1, x_inter, y_inter, debug)

    image_vert = image
    if debug:
        image_vert = cv2.circle(image_vert, (v1_x, v1_y), 10, (0,255,255))
        image_vert = cv2.circle(image_vert, (v0_x, v0_y), 10, (0,255,255))
        image_vert = cv2.circle(image_vert, (x_inter, y_inter), 10, (0,255,255))
        cv2.imshow("vertices", image_vert)
    v0 = (v0_x, v0_y)
    v1 = (v1_x, v1_y)
    v_inter = (int(x_inter[0]), int(y_inter[0]))

    return v_inter, v0, v1, image_vert
    
def find_center_of_mass(bw_image):
    """ compute the center of mass of the blob in bw_image"""
    non_zero = np.where(bw_image != 0)

    y_ = int(round(np.mean(non_zero[0]))) # mean value of row
    x_ = int(round(np.mean(non_zero[1]))) # mean value of col

    return y_, x_

def coo_to_azim_elev(x, y, K):
    """ Compute the azimuth and elevation angle (in degrees)of a point in image, requires the intrinsic camera matrix K"""
    P = np.asarray((x, y, 1)).T
    P_ics = np.linalg.inv(K).dot(P)
    azim = P_ics[0]*180/np.pi
    elev = P_ics[1]*180/np.pi

    return azim, elev

def compute_edge_len(rvec, tvec, K, dist):
    """return the approxmate length of the edge of the cube in #pixel"""
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
     Fast method is based on the translation vector and returns the distance to the center of the cube
     Similar triangle method is more computationally expensive and returns the average distance to the detected face"""
    if method == 'fast':
        return np.linalg.norm((tvec + 0.5)/10)
    elif method == 'simtri':
        dx, dy = compute_edge_len(rvec, tvec, K, dist)

        range_x = K[0, 0]*0.1/dx
        range_y = K[1, 1]*0.1/dy
        return (range_x + range_y)/2

def rotmat_to_quaternion(rmat):
    """compute the quaternion (qw, qx, qy, qz) from a rotation matrix"""
    Q = np.zeros(4, dtype=float)
    Q[0] = 0.5 * np.sqrt(rmat[0,0] + rmat[1,1] + rmat[2,2]+1)
    Q[1] = (rmat[2,1] - rmat[1,2])/(4*Q[0])
    Q[2] = (rmat[0,2] - rmat[2,0])/(4*Q[0])
    Q[3] = (rmat[1,0] - rmat[0,1])/(4*Q[0])


    return Q

def rotvec_to_quaternion(rvec):
    """compute the quaternion (qw, qx, qy, qz) from a rotation vector"""

    Q = np.zeros(4, dtype=float)
    angle = np.linalg.norm(rvec)
    ax = rvec/angle

    Q[0] = np.cos(angle/2)
    Q[1] = ax[0]*np.sin(angle/2)
    Q[2] = ax[1]*np.sin(angle/2)
    Q[3] = ax[2]*np.sin(angle/2)

    return Q


def solve_projection(p1, p2, p3, K, dist, image, last_position, debug=True):
    """Solve the projection from the 3 cube vertices and find the other ones in image"""
    p1_3d = (0.0, 0.0, 0.0)
    p2_3d = (0.0, 1.0, 0.0)
    p3_3d = (1.0, 0.0, 0.0)
    p11_3d = (0.0, 0.5, 0.0)

    p11_temp = (np.asarray(p1)+np.asarray(p2))/2
    p11 = (p11_temp[0], p11_temp[1])
    retval, rvec, tvec = cv2.solvePnP(np.asarray((p1_3d, p2_3d, p3_3d, p11_3d), dtype=float), np.asarray((p1,p2,p3, p11),dtype=float), K, dist)

    range_mean = compute_range(tvec, rvec, K, dist)

    p4_3d = (0.0, 0.0, 1.0)
    p5_3d = (1.0, 1.0, 0.0)
    p6_3d = (0.0, 1.0, 1.0)
    p7_3d = (1.0, 0.0, 1.0)
    p8_3d = (1.0, 1.0, 1.0)
    cm_3d = (0.5, 0.5, 0.5)

    P_3d = np.asarray((p4_3d, p5_3d, p6_3d, p7_3d, p8_3d, cm_3d), dtype=float)
    projected_points, _ = cv2.projectPoints(P_3d, rvec, tvec, K, dist)

    azim, elev = coo_to_azim_elev(int(projected_points[5, 0, 0]), int(projected_points[5, 0, 1]), K)

    pos_diff = np.abs((np.asarray((range_mean, azim, elev)) - last_position)/np.asarray((range_mean, azim, elev)))
    if np.sum(last_position != [0,0,0]) and np.sum(pos_diff > np.asarray([0.05, 3,3])):
        print('Cube was inverted to keep coherent position')

        p11_temp = (np.asarray(p1) + np.asarray(p3)) / 2
        p11 = (p11_temp[0], p11_temp[1])
        retval, rvec, tvec = cv2.solvePnP(np.asarray((p1_3d, p2_3d, p3_3d, p11_3d), dtype=float),
                                          np.asarray((p1, p3, p2, p11), dtype=float), K, dist)

        range_mean = compute_range(tvec, rvec, K, dist)

        projected_points, _ = cv2.projectPoints(P_3d, rvec, tvec, K, dist)

        azim, elev = coo_to_azim_elev(int(projected_points[5, 0, 0]), int(projected_points[5, 0, 1]), K)
        pos_diff = np.abs(np.asarray((range_mean, azim, elev)) - last_position)
        if np.sum(last_position != [0, 0, 0]) and np.sum(pos_diff > np.asarray([0.05, 3,3])):
            print('Warning : cube is moving very fast or a bad position was computed')
            print('pos diff :', pos_diff)

    if debug:
        for point in projected_points:
            #print(point)
            image = cv2.circle(image,(int(point[0, 0]),int(point[0, 1])), 10, (0, 255, 0))

        image = cv2.line(image, p1, p2, (0, 0, 255), thickness=5)
        image = cv2.line(image, p1, p3, (0, 255, 0), thickness=5)
        image = cv2.line(image, p1, (int(projected_points[0, 0, 0]), int(projected_points[0, 0, 1])), (255, 0, 0), thickness=5)

        cv2.imshow("projected points", image)

    quat = rotvec_to_quaternion(rvec)

    return range_mean, azim, elev, quat

def img_analysis(image, last_position, debug=True):
    """Performs the whole image analysis chain"""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eq_img = hist_eq(gray_img)

    if debug:
        cv2.imshow('original_img', image)
        cv2.imshow('gray_img', gray_img)
        cv2.imshow('eq_img', eq_img)

    ret, eq_img_thresh = cv2.threshold(eq_img, 80, 255, cv2.THRESH_BINARY)
    eq_img_thresh = cv2.morphologyEx(eq_img_thresh, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

    canny = cv2.Canny(eq_img_thresh.copy(), 30, 200)
    outer_canny = np.zeros(canny.shape, dtype=np.uint8)

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

    if debug:
        cv2.imshow("canny", canny)
        cv2.imshow("outer_canny", outer_canny)

    lines = cv2.HoughLines(outer_canny, 1, np.pi/180, 60)

    if lines is not None:
        if debug:
            lines_img = image
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

            cv2.imshow("houghlines", lines_img)

        angles = lines[:, :, 1]
        rhos =lines[:, :, 0]
        #print(rhos)

        main_dirs = find_main_axes(angles)
        main_rhos = np.zeros(main_dirs.shape)
        cnt = 0
        for direction in main_dirs:

            angle_diff = angles - direction

            main_rhos[cnt] = rhos[np.argmin(np.absolute(angle_diff))]

            cnt += 1

        #print(main_dirs)
        #print(main_rhos)

        if debug:
            for i in range(2):
                rho = main_rhos[i]
                theta = main_dirs[i]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))

                cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("houghlines", image)

    else:
        print("no lines found, reduce threshold")

        return 0,0,0,0, False

    if debug:
        cv2.imshow('thresholded image', eq_img_thresh)

    v0,v1,v2,image = find_cube_vertices(image, canny, main_dirs, main_rhos, debug)

    foc_mm = 12
    sens_w = 6.9
    sens_h = 5.5
    f_x = foc_mm/sens_w*image.shape[1]
    f_y = foc_mm/sens_h*image.shape[0]
    K = np.array([[f_x, 0, image.shape[1]/2],[0, f_y, image.shape[0]/2],[0,0,1]])

    range_mean, azim, elev, quat = solve_projection(v0,v1,v2,K, np.asarray([]), image, np.asarray(last_position), debug)

    if debug:
        image_with_centerpoint = cv2.circle(image,(image.shape[1]/2, image.shape[0]/2), 15, (255, 0 ,0))

        cv2.imshow("circle2", image_with_centerpoint)

        print("Results:")
        print("Range : {:f} m".format(range_mean))
        print("Azimuth : {:f} deg, Elevation : {:f} deg".format(azim, elev))
        print("Rotation quaternion : {:f} {:f} {:f} {:f}".format(quat[0], quat[1], quat[2], quat[3]))

        cv2.waitKey(0)

    return range_mean, azim, elev, quat, True
