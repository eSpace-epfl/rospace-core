import sys, os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import mixture
def hist_eq(image):
    """return image with equalize histogram"""
    eq_img = cv2.equalizeHist(image)

    return eq_img

def find_main_axes(angles):
    """ Fit the directions of Hough lines to a GMM and return the main directions"""
    g = mixture.GMM(n_components=2)

    g.fit(angles)

    return g.means_

def find_vertex(img_thresh, m, h, x_inter, y_inter):
    """Find the vertices in the cube from the intersection point and the line parameters (m,h)"""
    cube_px_y, cube_px_x = np.nonzero(img_thresh)
    temp = img_thresh
    dist_px = np.sqrt(np.square(cube_px_x - x_inter) + np.square(cube_px_y - y_inter))
    dist_line = np.sqrt((h + m * cube_px_x - cube_px_y) ** 2 / (m ** 2 + 1))

    thresh = 1
    num_close = 0
    while num_close < 50:
        num_close = np.sum(dist_line < thresh)
        thresh += 1

    #print("number of points near line", num_close)

    closest_to_line = np.argsort(dist_line)[:num_close]
    for pt in closest_to_line:
        temp = cv2.circle(temp, (cube_px_x[pt], cube_px_y[pt]), 10, 80)

    max_d = np.max(dist_px[closest_to_line])

    further_from_inter = np.argmin(abs(dist_px - max_d))

    #cv2.imshow("debug {:d}".format(num_close), temp)
    return cube_px_x[further_from_inter], cube_px_y[further_from_inter]

def find_cube_vertices(image, img_thresh,dirs, rhos):
    """Find the 3 vertices of the cubesat based on the houghlines"""

    m0 = -np.cos(dirs[0])/np.sin(dirs[0])
    h0 = rhos[0]/np.sin(dirs[0])

    m1 = -np.cos(dirs[1])/np.sin(dirs[1])
    h1 = rhos[1]/np.sin(dirs[1])

    x_inter = (h1-h0)/(m0-m1)
    y_inter = m0*x_inter+h0

    v0_x, v0_y = find_vertex(img_thresh, m0, h0, x_inter, y_inter)
    v1_x, v1_y = find_vertex(img_thresh, m1, h1, x_inter, y_inter)

    image_vert = cv2.circle(image, (v1_x, v1_y), 10, (0,255,255))
    image_vert = cv2.circle(image_vert, (v0_x, v0_y), 10, (0,255,255))
    image_vert = cv2.circle(image_vert, (x_inter, y_inter), 10, (0,255,255))
    cv2.imshow("vertices", image_vert)
    v0 = (v0_x, v0_y)
    v1 = (v1_x, v1_y)
    v_inter = (x_inter, y_inter)

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
    #print(P)
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

def solve_projection(p1, p2, p3, K, dist, image):
    """Solve the projection from the 3 cube vertices and find the other ones in image"""
    p1_3d = (0.0, 0.0, 0.0)
    p2_3d = (0.0, 1.0, 0.0)
    p3_3d = (1.0, 0.0, 0.0)
    retval, rvec, tvec = cv2.solvePnP(np.asarray((p1_3d, p2_3d, p3_3d), dtype=float), np.asarray((p1,p2,p3),dtype=float), K, dist)
    #print("rvec", rvec)
    #print("tvec", tvec)

    range_mean = compute_range(tvec, rvec, K, dist)

    p4_3d = (0.0, 0.0, 1.0)
    p5_3d = (1.0, 1.0, 0.0)
    p6_3d = (0.0, 1.0, 1.0)
    p7_3d = (1.0, 0.0, 1.0)
    p8_3d = (1.0, 1.0, 1.0)
    cm_3d = (0.5, 0.5, 0.5)

    P_3d = np.asarray((p4_3d, p5_3d, p6_3d, p7_3d, p8_3d, cm_3d), dtype=float)
    projected_points, _ = cv2.projectPoints(P_3d, rvec, tvec, K, dist)

    for point in projected_points:
        #print(point)
        image = cv2.circle(image,(int(point[0, 0]),int(point[0, 1])), 10, (0, 255, 0))
    #projected_cm = cv2.projectPoints(np.asarray(cm_3d, dtype=float), rvec, tvec, K, dist)
    azim, elev = coo_to_azim_elev(int(projected_points[5, 0, 0]), int(projected_points[5, 0, 1]), K)

    image = cv2.line(image, p1, p2, (0, 0, 255), thickness=5)
    image = cv2.line(image, p1, p3, (0, 255, 0), thickness=5)
    image = cv2.line(image, p1, (int(projected_points[0, 0, 0]), int(projected_points[0, 0, 1])), (255, 0, 0), thickness=5)

    #image = cv2.circle(image, (int(projected_cm[0,0]), int(projected_cm[0,1])), 15, (0,126,126))
    cv2.imshow("projected points", image)

    return range_mean, azim, elev, rvec*180/np.pi

def img_analysis(image):
    """Performs the whole image analysis chain"""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eq_img = hist_eq(gray_img)

    eq_img_filt = cv2.bilateralFilter(eq_img.copy(), 11, 17, 17)

    cv2.imshow('eq_img',eq_img)
    cv2.imshow('eq_img_filt',eq_img_filt)

    canny = cv2.Canny(eq_img_filt.copy(), 30, 200)

    cv2.imshow("canny", canny)

    lines = cv2.HoughLines(canny, 1, np.pi/180, 80)

    if lines is not None:
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




    sobelx = np.absolute(cv2.Sobel(eq_img, cv2.CV_64F, 1, 0, ksize=3))
    sobely = np.absolute(cv2.Sobel(eq_img, cv2.CV_64F, 0, 1, ksize=3))

    edge_dir = np.arctan2(sobely, sobelx)
    edge_int = np.sqrt(sobelx**2 + sobely**2)

    ret, eq_img_thresh = cv2.threshold(eq_img,80,255,cv2.THRESH_BINARY)
    eq_img_thresh_mean = 255-cv2.adaptiveThreshold(eq_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    eq_img_thresh_gaus = 255-cv2.adaptiveThreshold(eq_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    cv2.imshow('thresholded image', eq_img_thresh)

    #v0, v1, v2, image = find_cube_vertices(image, cv2.morphologyEx(eq_img_thresh_gaus,cv2.MORPH_OPEN,np.ones((1,1), np.uint8)),main_dirs, main_rhos)
    v0,v1,v2,image = find_cube_vertices(image, canny, main_dirs, main_rhos)
    #print("vertices : ", v0, v1,v2)
    #cv2.imshow("houghlines", image)

    foc_mm = 12
    sens_w = 6.9
    sens_h = 5.5
    f_x = foc_mm/sens_w*image.shape[1]
    f_y = foc_mm/sens_h*image.shape[0]
    K= np.array([[f_x, 0, image.shape[1]/2],[0, f_y, image.shape[0]/2],[0,0,1]])
    #print(K)

    range_mean, azim, elev, pose = solve_projection(v0,v1,v2,K, np.asarray([]), image)

    y_center, x_center = find_center_of_mass(eq_img_thresh)
    image_with_centerpoint = cv2.circle(image,(image.shape[1]/2, image.shape[0]/2), 15, (255, 0 ,0))

    cv2.imshow("circle2", image_with_centerpoint)

    #azim, elev = coo_to_azim_elev(x_center, y_center,K)

    print("Results:")
    print("Range : {:f} m".format(range_mean))
    print("Azimuth : {:f} deg, Elevation : {:f} deg".format(azim, elev))
    print("Pose relative to image coordinate frame : {:f}, {:f}, {:f} deg".format(pose[0,0], pose[1,0], pose[2,0]))

    cv2.waitKey(0)



    return range_mean, azim, elev, pose
