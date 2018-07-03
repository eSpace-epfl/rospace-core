#!/usr/bin/env python
#  @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
#  @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import numpy as np


class BDotController(object):
    """Simple implementation of BDOT controller.

    The implementation is specified in the Swisscube Document:
        [1] S3-B-C-ADCS-1-2_BDot_Controller_Code.pdf
    and is available at:
    http://escgesrv1.epfl.ch/06%20-%20Attitude%20control/S3-B-C-ADCS-1-2-Bdot_Controller_Code.pdf

    Attributes:
        lmbda (float): Derivative stage parameter [-]
        A (float): Area surrounded by magnetotorquer coils [m^2]
        K_Bdot (float): Controller gain [Nms]
        n (int): Number of windings per magnetotorquer [-]
        maxCurrent (float): Maximum current allowed through the coils [m^2]
        threshold (float): Desired value of minimum rotational speed [rad/s]
        Bdot_old (numpy.array): Magnetic field derivative from previous time step [T/s]
        B_old (numpy.array): Magnetic field from previous time step [T]
        t_old (float): simulation time of last time step [s]

    """

    def __init__(self):
        self.lmbda = 0.7  # (spelled wrong on purpose)
        self.A = 4.861*10**(-3)
        self.K_Bdot = 1.146 * 10**-2
        self.n = 427
        self.maxCurrent = 50*10**-3

        self.threshold = 0  # tbd

        self.Bdot_old = np.array([0, 0, 0])
        self.B_old = np.array([0, 0, 0])
        self.t_old = 0.0

    def run_controller(self, B_field, w_vec, time):
        """Start the controller.

        The equations for the control algorithm can be found at [1]

        Args:
            B_field (numpy.array): Magnetic field in the spacecraft body frame [T]
            w_vec (numpy.array): current spin vector of spacecraft in body frame [rad/s]
            time (float): current simulation time [s]

        Returns:
            numpy.array: Current current provided for magnetotorques [A]

        """
        # set start values
        if self.t_old == 0.0:
            self.t_old = time
            self.B_old = B_field
            return np.array([0, 0, 0])

        # if not advancing in time return last value
        delta_t = time - self.t_old
        if delta_t == 0.0:
            # return np.array([0,0,0])
            return self.Bdot_old

        # Equation (1) from [1]
        Bdot = (1 - self.lmbda) * self.Bdot_old + self.lmbda * (B_field - self.B_old) / delta_t

        w = np.linalg.norm(w_vec)
        if w >= self.threshold:
            K_s = -1
        else:
            K_s = 1

        K_B = 1 / np.sum(B_field**2)
        current = K_B * Bdot * K_s * self.K_Bdot * (1 / (self.A * self.n))

        I_max = max(abs(current))

        if I_max > self.maxCurrent:

            # limit current
            K_i = self.maxCurrent / I_max
            current = current*K_i

        self.B_old = B_field
        self.Bdot_old = Bdot

        # return currents for torquers
        return current
