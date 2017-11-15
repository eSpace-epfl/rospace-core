import sys, os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import mixture
def hist_eq(image):

    #gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #plt.hist(gray_img.ravel())
    #plt.show()

    eq_img = cv2.equalizeHist(image)

    #plt.hist(eq_img.ravel())
    #plt.show()
    #print('saving image')
    #cv2.imwrite('/equalized_cube.png', eq_img)
    #print('saved')
    #cv2.namedWindow('original')
    #cv2.namedWindow('equalized')

    #cv2.imshow('original', gray_img)
    #cv2.imshow('equalized', eq_img)

    #cv2.waitKey(0)

    return eq_img

def find_main_axes(angles):
    g = mixture.GMM(n_components=2)

    g.fit(angles)

    return g.means_


def find_center_of_mass(bw_image):
    """ compute the center of mass of the blob in bw_image"""
    non_zero = np.where(bw_image != 0)

    y_ = int(round(np.mean(non_zero[0]))) # mean value of row
    x_ = int(round(np.mean(non_zero[1]))) # mean value of col

    return y_, x_

def coo_to_azim_elev(x, y, K):
    """ Compute the azimuth and elevation angle of a point in image, requires the intrinsic camera matrix K"""
    P = np.asarray((x,y,1)).T
    print(P)
    P_ics = np.linalg.inv(K).dot(P)
    azim = P_ics[0]
    elev = P_ics[1]
    return azim, elev

def img_analysis(image):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eq_img = hist_eq(gray_img)

    eq_img_filt = cv2.bilateralFilter(eq_img.copy(),11,17,17)

    cv2.imshow('eq_img',eq_img)
    cv2.imshow('eq_img_filt',eq_img_filt)

    canny = cv2.Canny(eq_img_filt.copy(),30,200)

    cv2.imshow("canny", canny)

    lines = cv2.HoughLines(canny,1,np.pi/180,80)

    if lines is not None:
        angles = lines[:,:,1]
        rhos =lines[:,:,0]
        print(rhos)

        main_dirs = find_main_axes(angles)
        main_rhos = np.zeros(main_dirs.shape)
        cnt=0
        for direction in main_dirs:

            angle_diff = angles - direction

            main_rhos[cnt] = rhos[np.argmin(np.absolute(angle_diff))]

            cnt+=1

        print(main_dirs)
        print(main_rhos)
       #angle_thresh = np.pi/180*5
       #main_dirs_num = np.array([0,0])
       #main_dirs = np.array([0.,0.])
       #main_rhos = np.array([0.,0.])
       #for angle in np.arange(5,175,10):
       #    angle_r =angle* np.pi/180
       #
       #    #print(np.logical_and(angles>angle_r -angle_thresh,angles < angle_r + angle_thresh))
       #    num_lines = np.sum(np.logical_and(angles>angle_r -angle_thresh,angles < angle_r + angle_thresh))
       #    if num_lines > np.min(main_dirs_num):
       #        i = np.argmin(main_dirs_num)
       #        print(i)
       #        print(num_lines)
       #        main_dirs_num[i] = num_lines
       #        print(np.mean(angles[np.logical_and(angles>angle_r -angle_thresh,angles < angle_r + angle_thresh)]))
       #        main_dirs[i] = np.mean(angles[np.logical_and(angles>angle_r -angle_thresh,angles < angle_r + angle_thresh)])
       #        main_rhos[i] = np.mean(rhos[np.logical_and(angles>angle_r -angle_thresh,angles < angle_r + angle_thresh)])

        #for i in range(2):
        #    rho = main_rhos[i]
        #    theta = main_dirs[i]
        #    a = np.cos(theta)
        #    b = np.sin(theta)
        #    x0 = a * rho
        #    y0 = b * rho
        #    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        #    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        #
        #    cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        #cv2.imshow("houghlines", image)

        for line in lines:
            rho = line[0][0]
            theta = line [0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        
            cv2.line(image,pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow("houghlines", image)

    else:
        print("no lines found, reduce threshold")




    sobelx = np.absolute(cv2.Sobel(eq_img,cv2.CV_64F,1,0,ksize=3))
    sobely = np.absolute(cv2.Sobel(eq_img,cv2.CV_64F,0,1,ksize=3))

    edge_dir = np.arctan2(sobely,sobelx)
    edge_int = np.sqrt(sobelx**2 + sobely**2)

    #edge_dir[edge_int==0] = np.nan

    #plt.hist(edge_dir[edge_dir!=0].ravel())
    #plt.show()

    #cv2.imshow("h edges", sobelx)
    #cv2.imshow("v edges", sobely)

    #cv2.imshow("edge directions",edge_dir)
    #cv2.imshow("edge intensity", edge_int)

    ret, eq_img_thresh = cv2.threshold(eq_img,80,255,cv2.THRESH_BINARY)
    eq_img_thresh_mean = 255-cv2.adaptiveThreshold(eq_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    eq_img_thresh_gaus = 255-cv2.adaptiveThreshold(eq_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    cv2.imshow('thresholded image', eq_img_thresh)

    y_center, x_center = find_center_of_mass(eq_img_thresh)
    print(y_center, x_center)
    image_with_centerpoint = cv2.circle(image,(image.shape[1]/2, image.shape[0]/2), 15, (255, 0 ,0))
    cv2.imshow("circle2", cv2.circle(image_with_centerpoint,(x_center, y_center), 20, (0, 255, 0)))

    foc_mm = 30
    sens_w = 6.9
    sens_h = 5.5
    f_x = foc_mm/sens_w*image.shape[1]
    f_y = foc_mm/sens_h*image.shape[0]
    K= np.array([[f_x, 0, image.shape[1]/2],[0, f_y, image.shape[0]/2],[0,0,1]])
    print(K)
    azim, elev = coo_to_azim_elev(x_center, y_center,K)
    print(azim, elev)
    #cv2.imshow('thresholded image mean', eq_img_thresh_mean)
    #cv2.imshow('thresholded image gaussian', eq_img_thresh_gaus)



    cv2.waitKey(0)


    d = azim = elev = 0.1

    return d, azim, elev
