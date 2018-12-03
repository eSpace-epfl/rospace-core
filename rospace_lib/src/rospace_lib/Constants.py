# Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

"""Constants and Helper Classes"""

import numpy as np
import datetime


def R_x(q):
    """Calculates a rotation about the x-axis

    Args:
        q (float): Angle in Radians

    Returns: Rotation Matrix in x

    """
    return np.array([[1, 0, 0], [0, np.cos(q), -np.sin(q)], [0, np.sin(q), np.cos(q)]])


def R_y(q):
    """Calculates a rotation about the y-axis

    Args:
        q (float): Angle in Radians

    Returns: Rotation Matrix in y

    """
    return np.array([[np.cos(q), 0, np.sin(q)], [0, 1, 0], [-np.sin(q), 0, np.cos(q)]])


def R_z(q):
    """Calculates a rotation about the z-axis

    Args:
        q (float): Angle in Radians

    Returns: Rotation Matrix in z

    """
    return np.array([[np.cos(q), -np.sin(q), 0], [np.sin(q), np.cos(q), 0], [0, 0, 1]])


mu_earth = 3.986004418e14 / 1e9
'''Standard Gravitational Prameter of Earth [km^3 s^-2]'''

R_earth = 6378.1363
'''Radius of earth [km]'''

J_2 = 1.08262668 * 10 ** (-3.0)
'''Second dynamic form factor of earth.'''

J2000_jd = 2451545.0
'''Julian Date of J2000 Epoch'''

J2000_date = datetime.datetime(2000, 1, 1, 11, 59, 0)
'''UTC datetime of J2000 Epoch'''
