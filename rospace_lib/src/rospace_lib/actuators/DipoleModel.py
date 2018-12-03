# Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

import math
import numpy as np


class DipoleModel(object):
    """Class which adds hysteresis rods and bar magnets to satellite & computes their dipole moment vector.

    This class uses a simplified model for the calculation of the hysteresis rods.
    Only one hysteresis loop per rod is created to model the hysteresis response.
    Also the initial magnetizing phase is neglected and not modeled.

    They Hysteresis model is based on:
        [1] 'A Magnetic Hysteresis Model' by T.W. Flatley and D.A. Henretty

    Attributes:
        _H_hyst_last (numpy.array): Magnetic field strength induced in hysteresis rods at last attitude integration step
        _B_hyst_last(numpy.array): Magnetic density induced in hysteresis rods at last attitude integration step
        _Bs (numpy.array): Saturation induction of hysteresis rods
        _Hc (numpy.array): Coercive force of hysteresis rod
        _k (numpy.array): Hysteresis rod constant determined from a set of magnetic hysteresis parameters
        _volume (numpy.array): Volume of hysteresis rod
        _direction_hyst (numpy.array): Dipole moment direction of hysteresis rods
        _m_hyst (numpy.array): Dipole moment vector of hysteresis rods
        _m_barMag (numpy.array): Dipole moment vector of bar magnets

    """

    def __init__(self):
        self._H_hyst_last = np.empty([0, 1])

        self._B_hyst_last = np.empty([0, 1])

        self._Bs = np.empty([0, 1])

        self._Hc = np.empty([0, 1])

        self._k = np.empty([0, 1])

        self._volume = np.empty([0, 1])

        self._direction_hyst = np.empty([0, 3])

        self._m_hyst = np.empty([0, 3])

        self._m_barMag = np.empty([0, 3])

        self.MU_0 = 4e-7 * math.pi  # permeability of free space
        self.TWO_DIV_PI = 2. / math.pi  # value for 2 / pi used often in calculation

    def addBarMagnet(self, direction, m):
        """Add Bar Magnet to satellite.

        Args:
            direction (numpy.array): Direction of dipole moment vector
            m (float): Dipole moment in [Am^2]

        """
        norm = np.linalg.norm(direction)
        if norm != 1.0:
            direction = direction / norm

        self._m_barMag = np.append(self._m_barMag, [m*direction], axis=0)

    def addHysteresis(self, direction, volume, Hc, Bs, Br):
        """Add hysteresis rod to satellite.

        Args:
            direction (numpy.array): Direction of dipole moment vector of rod
            volume (float): Volume of hysteresis rod in [m^3]
            Hc (float): Coercive force of hysteresis rod in [A/m]
            Bs (float): Saturation induction of hysteresis rod in [Tesla]
            Br (float): Remanence of hysteresis rod in [Tesla]

        """
        self._Hc = np.insert(self._Hc, self._Hc.size, Hc, axis=0)
        self._Bs = np.insert(self._Bs, self._Bs.size, Bs, axis=0)
        _k = 1 / Hc * math.tan(math.pi * 0.5 * Br / Bs)
        self._k = np.insert(self._k, self._k.size, _k, axis=0)

        # normalize if necessary
        norm = np.linalg.norm(direction)
        if norm != 1.0:
            direction = direction / norm

        self._direction_hyst = np.append(self._direction_hyst, [direction], axis=0)
        self._volume = np.insert(self._volume, self._volume.size, volume, axis=0)

        self._H_hyst_last = np.append(self._H_hyst_last, [[0.]], axis=0)
        self._B_hyst_last = np.append(self._B_hyst_last, [[0.]], axis=0)

    def initializeHysteresisModel(self, B_field):
        """Initialize induced magnetic density in hysteresis rods._m_barMag

        Args:
            B_field (numpy.array): magnetic flux density of magnetic field in satellite frame
        """
        self._compute_m_hyst(B_field)

    def getDipoleVectors(self, B_field):
        """Compute dipole moment vectors of satellite's hysteresis rods and bar magnets.

        Args:
            B_field (numpy.array): magnetic flux density of magnetic field in satellite frame

        Returns:
            numpy.array: array with the dimensions of n x 3. First entries in array are
                         hysteresis rods' dipole moments and last are bar magnets'

        """
        self._compute_m_hyst(B_field)
        return np.append(self._m_hyst, self._m_barMag, axis=0)

    def _compute_m_hyst(self, B_field):
        """Compute the magnetic dipole vector of hysteresis rods (Internal method).

        This method also stores the last values for the magnetic field strength
        and magnetic density induced in the hysteresis rods.

        The induced magnetic density is computed based on eq. (15) in [1]

        Args:
            B_field (numpy.array): magnetic flux density of magnetic field in satellite frame

        """
        _H = B_field / self.MU_0

        H_hyst = np.dot(self._direction_hyst, _H)[:, None]

        # define if on left or right hysteresis curve depending on dH/dt
        sign = np.where(H_hyst > self._H_hyst_last, -1., 1.)

        # keep old value if magnetic field did not change
        change = np.where(H_hyst == self._H_hyst_last, 0., 1.)

        B_hyst = self.TWO_DIV_PI * self._Bs * np.arctan(self._k * (H_hyst + (sign * self._Hc)))
        B_hyst = B_hyst * change + self._B_hyst_last * (1 - change)  # use old B_hyst if H didnt change

        self._B_hyst_last = B_hyst
        self._H_hyst_last = H_hyst
        self._m_hyst = (B_hyst * self._volume / self.MU_0) * self._direction_hyst
