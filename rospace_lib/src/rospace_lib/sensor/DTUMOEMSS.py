# @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.
import numpy as np
import scipy as sp
from scipy.interpolate import UnivariateSpline

class DTUMOEMSS(object):
    """
    Model of a 2 axis DTU MOMEMS Sun Sensor
    See for characteristics:
       [1] https://arc.aiaa.org/doi/pdf/10.2514/6.IAC-03-U.2.b.02


    Currently, this ignores discretization errors and temperature drifts and the like.
    """
    def __init__(self):
        # set default transfer function
        self.transfer_func = lambda x: x
        self.noise_func = lambda x: 5

        # maximum view angles
        self.max_val = np.deg2rad(65)
        self.min_val = -np.deg2rad(65)

        # current sun vector in space craft body frame
        self.T_SS_B = np.identity(4)

        # current vector from Sun to space craft body frame in spacecraft body frame
        self.p_B_S = np.zeros(3)

    def set_noise_func(self, samples_x, samples_y):
        self.noise_func = UnivariateSpline(samples_x, samples_y)

        # calculate error over all samples
        samples_y_calc = np.zeros(samples_y.shape)
        for i in range(0, len(samples_x)):
            samples_y_calc[i] = self.noise_func(samples_x[i])

        print np.linalg.norm(samples_y - samples_y_calc)


    def set_transfer_func(self, samples_x, samples_y):
        self.transfer_func = UnivariateSpline(samples_x, samples_y)

        # calculate error over all samples
        samples_y_calc = np.zeros(samples_y.shape)
        for i in range(0, len(samples_x)):
            samples_y_calc[i] = self.transfer_func(samples_x[i])

        print np.linalg.norm(samples_y-samples_y_calc)


    def set_true_value(self, sun_vector):
        self.p_B_S = sun_vector

    def get_value(self):
        response_vector = np.zeros(2)

        print self.T_SS_B
        # get sun vector in sensor frame
        p_SS_S = np.dot(self.T_SS_B, np.append(self.p_B_S, 1))

        # normalize s.t. z = 1
        p_SS_S = p_SS_S[0:3] / p_SS_S[2]


        # calculate angle 1 and 2
        angle_x = np.arctan2(p_SS_S[0], p_SS_S[2])
        angle_y = np.arctan2(p_SS_S[1], p_SS_S[2])

        if(angle_x > self.max_val or
           angle_x < self.min_val or
           angle_y > self.max_val or
           angle_y < self.min_val):
            response_vector[:] = np.nan
            return response_vector

        print angle_x
        # use transfer function to get non-noisy sensor response
        angle_1_response = self.transfer_func(angle_x)
        angle_2_response = self.transfer_func(angle_y)

        print "sdev at angle:", self.noise_func(angle_x)
        # add noise with variance according to the noise function
        angle_1_response = angle_1_response + np.random.normal(0, self.noise_func(angle_x))
        angle_2_response = angle_2_response + np.random.normal(0, self.noise_func(angle_y))

        print angle_1_response

        response_vector[0] = angle_1_response
        response_vector[1] = angle_2_response


        return response_vector

