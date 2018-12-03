# Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
# Copyright (c) 2017, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

import numpy as np
import scipy as sp
from scipy.interpolate import UnivariateSpline


class DTUMOEMSS(object):
    """
    Model of a 2 axis DTU MOMEMS Sun Sensor
    See for characteristics:
       [1] https://arc.aiaa.org/doi/pdf/10.2514/6.IAC-03-U.2.b.02
       [2] https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1929&context=smallsat

    The maximal and minimal view angles of the Sun Sensor are set to +-65 [deg], which are
    the values specified in [1].

    The sun sensor is build out of two one-axis sensors which are perpendicular to each other. In
    a provided sensor frame the first sensor measures the angle between the sun array and z-axis in the z-x plane
    (x-Angle) and has its slit aligned with the y-axis. The second sensor measures the angular separation between
    sun array and z-axis in the y-z plane (y-Angle) and has its slit aligned with the x-axis.

    The z-axis of the sensor frame should be perpendicular to the plane span by the sensor and pointing outwards.

    Currently, this ignores discretization errors and temperature drifts and the like.

    Args:
        T_SS_B (numpy.ndarray): 4x4 array for transformation from spacecraft body frame to sun sensor

    Attributes:
        transfer_func (scipy.interpolate.UnivariateSpline): transfer function created from samples provided by a table
        inv_transfer_func (scipy.interpolate.UnivariateSpline): inverse of transfer function
        (possible as sensor reading is approximately a linear function)
        noise_func (scipy.interpolate.UnivariateSpline): noise function created from samples provided by a table
        max_val (float): maximal value of viewing angle
        min_val (float): minimal value of viewing angle
        T_SS_B (np.ndarray): 4x4 array for transformation from spacecraft body frame to sun sensor
        p_B_S (np.array): 1x3 position vector of the sun in spacecraft body frame
    """
    def __init__(self, T_SS_B):

        self.transfer_func = lambda x: x
        self.inv_transfer_func = lambda x: 1/x

        self.noise_func = lambda x: 5

        self.max_val = np.deg2rad(65)
        self.min_val = -np.deg2rad(65)

        self.T_SS_B = T_SS_B

        self.p_B_S = np.zeros(3)

    def set_noise_func(self, samples_x, samples_y):
        """Sets the noise function of the sensor model using a UnivariateSpline for interpolation between sample points.

        The noise function represents the standard deviation as a function of incident angle of the sun's array.

        Args:
            samples_x (np.array): incident angle given in radians
            samples_y (np.array): standard deviation of sensor response
        """
        self.noise_func = UnivariateSpline(samples_x, samples_y)

    def set_transfer_func(self, samples_x, samples_y):
        """Sets the transfer function of the sensor model using a UnivariateSpline for interpolation between sample points.

        The transfer function represents the relation between sensor output and incident angle of the sun's array.

        Args:
            samples_x (np.array): incident angle given in radians
            samples_y (np.array): sensor response
        """
        self.transfer_func = UnivariateSpline(samples_x, samples_y)

        self.inv_transfer_func = UnivariateSpline(samples_y, samples_x)

    def set_true_value(self, sun_vector):
        """Set the current value of the sun vector as seen by the spacecraft.

        Args:
            sun_vector (np.array): 1x3 position vector of the sun in spacecraft body frame
        """
        self.p_B_S = sun_vector

    def get_value(self):
        """Compute the sensor response for the last true value of the sun sensor, which has been set.

        The response is given as measured angles in the sensor frame.

        If the incident angle is not in the interval specified in the constructor, this method returns NaN values
        as sensor response.

        Returns:
            np.array: 1x2 array with the measured angles in sensor frame as: [x-Angle ; y-Angle]
        """
        response_vector = np.zeros(2)

        # get sun vector in sensor frame
        p_SS_S = np.dot(self.T_SS_B, np.append(self.p_B_S, 1))

        # normalize s.t. z = 1
        p_SS_S = p_SS_S[0:3] / p_SS_S[2]

        # calculate angle 1 and 2
        angle_x = np.arctan2(p_SS_S[0], p_SS_S[2])
        angle_y = np.arctan2(p_SS_S[1], p_SS_S[2])

        # print "Angles: ", angle_x, angle_y

        if(angle_x > self.max_val or
           angle_x < self.min_val or
           angle_y > self.max_val or
           angle_y < self.min_val):
            response_vector[:] = np.nan
            return response_vector

        # use transfer function to get non-noisy sensor response
        angle_1_response = self.transfer_func(angle_x)
        angle_2_response = self.transfer_func(angle_y)

        # add noise with variance according to the noise function
        angle_1_response = angle_1_response + np.random.normal(0, self.noise_func(angle_x))
        angle_2_response = angle_2_response + np.random.normal(0, self.noise_func(angle_y))

        # get angles from response using transfer function:
        angle_1 = self.inv_transfer_func(angle_1_response)
        angle_2 = self.inv_transfer_func(angle_2_response)

        response_vector[0] = angle_1
        response_vector[1] = angle_2

        return response_vector
