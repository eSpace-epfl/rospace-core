# Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...

from rospace_lib.actuators import OneAxisMagnetorquer


class MagnetorquersTest(unittest.TestCase):

    def setUp(self):
        # create torque with SwissCube settings aligned with x-axis:
        self.torquer = OneAxisMagnetorquer(windings=427, area=0.0731**2,
                                           radius=0.0354, orient=np.array([1, 0, 0]))

    def test_dipole_moment(self):
        """Test computation of dipole moment.

        With a unit magnetic field oriented in positive y-direction and a
        current of 22.7 mA the moment created should yield approximately 51.8 mNm
        around the z-axis.
        """
        B_field = np.array([0, 1, 0])
        current = 22.7*10**-3  # 95% current

        self.torquer.set_B_field(B_field)
        self.torquer.set_I(current)

        # this should yield roughly 51.8 mNm magnetic moment around z
        resulting_torque = self.torquer.get_torque()
        self.assertAlmostEqual(51.8*10**-3, resulting_torque[2], 3)

        # and zero for the other axes
        self.assertAlmostEqual(0, resulting_torque[0], 6)
        self.assertAlmostEqual(0, resulting_torque[1], 6)


if __name__ == '__main__':
    unittest.main()
