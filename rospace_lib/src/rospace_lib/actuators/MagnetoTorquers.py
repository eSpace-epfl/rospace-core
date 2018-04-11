# @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.
import numpy as np

class ThreeAxisMagnetoTorquer(object):
    def __init__(self):
        self.x_torquer = OneAxisMagnetoTorquer()
        self.x_torquer._orientation = np.array([1, 0, 0])

        self.y_torquer = OneAxisMagnetoTorquer()
        self.y_torquer._orientation = np.array([0, 1, 0])

        self.z_torquer = OneAxisMagnetoTorquer()
        self.z_torquer._orientation = np.array([0, 0, 1])


    def set_I(self, I ):
        self.x_torquer.set_I(I[0])
        self.y_torquer.set_I(I[1])
        self.z_torquer.set_I(I[2])

    def set_B_field(self, B):
        self.x_torquer.set_B_field(B)
        self.y_torquer.set_B_field(B)
        self.z_torquer.set_B_field(B)

    def get_torque(self):
        x_torque = self.x_torquer.get_torque()
        y_torque = self.y_torquer.get_torque()
        z_torque = self.z_torquer.get_torque()

        # all torques in same frame, so combine..
        return x_torque + y_torque + z_torque

class OneAxisMagnetoTorquer(object):
    """
    Model of a 1 axis Magentotorquer
    Modelling is done according to Swisscube Document
        Phase D: Characterization of the SwissCube magnetometers and
        magnetotorquers, "S3-D-ADCS-1-7-MM_MT_Characterization.pdf", 2008
        by G. Bourban, M. Noca, P. Kejik, R. Popovic


    """
    def __init__(self):
        # Physical properties
        self.mu_0 = 4.0 * np.pi * 10**(-7) # magnetic constant
        self.N = 427  #  Windings
        self.A = 0.0731**2 # Some Area [m] TODO: Magic number - where is it from?
        self.R = 0.0354  # Radius [m]

        # orientation of magnetorquer
        self._orientation = np.array([1, 0, 0])

        # Current B Field in body frame
        self._current_I = 0.0  # Current current [A]
        self._current_B_field = np.zeros(3) # Current B-field in body frame


    def set_I(self, I):
        self._current_I = I

    def set_B_field(self, B):
        self._current_B_field = B

    def get_torque(self):
        # first calculate magnetic dipole moment
        coil_mag_moment = self.N * self._current_I * self.A * self._orientation # [A*m^2]


        # get resulting torque
        # units = [A*m^2] x [ N * s * C^-1 * m^-1]
        # Note that Ampere can be interpreted as [C* s^-1]
        # The resulting unit is:
        # [C * s^-1 * m^2] x [N * s * C^-1 * m^-1] = [N*m]
        magnetic_torque = np.cross(coil_mag_moment, self._current_B_field)

        return magnetic_torque

