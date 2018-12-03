# Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
# Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.
import numpy as np


class ThreeAxisMagnetorquer(object):
    """Creates a three axis magnetorquer.

    The three axis are modeled independently using :class:`OneAxisMagnetorquer`.

    Currently, all three one axis magnetorquer are aligned with one axis of the spacecraft's
    body frame and all have the same physical properties.

    Args:
        windings (int): Number of windings of magnetorquer
        area (float): Some Area [m] ; TODO: Magic number - where is it from?
        radius (float): radius [m]

    """

    def __init__(self, windings, area, radius):
        self._x_torquer = OneAxisMagnetorquer(windings, area, radius, np.array([1, 0, 0]))

        self._y_torquer = OneAxisMagnetorquer(windings, area, radius, np.array([0, 1, 0]))

        self._z_torquer = OneAxisMagnetorquer(windings, area, radius, np.array([0, 0, 1]))

    def set_I(self, I):
        """Set the current current for all three axes.

        Args:
            I (numpy.array): current current in spacecraft body frame [A]

        """
        self._x_torquer.set_I(I[0])
        self._y_torquer.set_I(I[1])
        self._z_torquer.set_I(I[2])

    def set_B_field(self, B):
        """Set current magnetic field.

        The magnetic field provided should be defined in the spacecraft body frame.

        Args:
            B (numpy.array): Magnetic Field [T]

        """
        self._x_torquer.set_B_field(B)
        self._y_torquer.set_B_field(B)
        self._z_torquer.set_B_field(B)

    def get_torque(self):
        """Get the current torque induced by magnetorquer.

        The torque returned is provided in the spacecraft body frame.

        Returns:
            numpy.array: current torque induced by magnetorquer [Nm]

        """
        x_torque = self._x_torquer.get_torque()
        y_torque = self._y_torquer.get_torque()
        z_torque = self._z_torquer.get_torque()

        # all torques in same frame, so combine..
        return x_torque + y_torque + z_torque


class OneAxisMagnetorquer(object):
    """Model of a 1 axis Magnetorquer.

    Modeling is done according to SwissCube Document
        Phase D: Characterization of the SwissCube magnetometers and
        magnetorquer, "S3-D-ADCS-1-7-MM_MT_Characterization.pdf", 2008
        by G. Bourban, M. Noca, P. Kejik, R. Popovic

    Args:
        windings (int): Number of windings of magnetorquer
        area (float): Some Area [m] ; TODO: Magic number - where is it from?
        radius (float): radius [m]
        orient (numpy.array): orientation in spacecraft body frame

    Attributes:
        mu_0 (float): magnetic constant
        n (int): number of windings
        A (float): area
        R (float): radius

    """

    def __init__(self, windings, area, radius, orient):
        # Physical properties
        self.mu_0 = 4.0 * np.pi * 10**(-7)
        self.n = windings  # 427
        self.A = area  # 0.0731**2
        self.R = radius  # 0.0354

        # orientation of magnetorquer
        self._orientation = orient

        # Current B Field in body frame
        self._current_I = 0.0  # Current current [A]
        self._current_B_field = np.zeros(3)  # Current B-field in body frame

    def set_I(self, I):
        """Set current Current.

        Args:
            I (float): Current [A]

        """
        self._current_I = I

    def set_B_field(self, B):
        """Set current magnetic field.

        The magnetic field provided should be defined in the spacecraft body frame.

        Args:
            B (numpy.array): Magnetic Field [T]
        """
        self._current_B_field = B

    def get_torque(self):
        """Get the current torque induced by magnetorquer.

        The torque returned is provided in the spacecraft body frame.

        Returns:
            numpy.array: current torque induced by magnetorquer [Nm]

        """
        # first calculate magnetic dipole moment
        coil_mag_moment = self.n * self._current_I * self.A * self._orientation  # [A*m^2]

        # get resulting torque
        # units = [A*m^2] x [ N * s * C^-1 * m^-1]
        # Note that Ampere can be interpreted as [C* s^-1]
        # The resulting unit is:
        # [C * s^-1 * m^2] x [N * s * C^-1 * m^-1] = [N*m]
        magnetic_torque = np.cross(coil_mag_moment, self._current_B_field)

        return magnetic_torque
