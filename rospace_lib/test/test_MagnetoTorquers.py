# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

# Note: these tests are quite preliminary....

import unittest
import sys
import os
from copy import deepcopy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src/")  # hack...
from rospace_lib.actuators import *

class MagnetoTorquersTest(unittest.TestCase):


    def test_dipole_moment(self):
        torquer = OneAxisMagnetoTorquer()
        torquer._orientation = np.array([1, 0, 0]) # Torquer aligned with x-axis
        B_field = np.array([0,1, 0]) # unit b-field in y
        I = 22.7*10**-3  # 95% current (22.7 mA)

        torquer.set_B_field(B_field)
        torquer.set_I(I)

        # this should yield roughly 51.8 mNm magnetic moment around z
        resulting_torque = torquer.get_torque()
        self.assertAlmostEqual(51.8*10**-3, resulting_torque[2], 3)
        # and zero for the other axes
        self.assertAlmostEqual(0,resulting_torque[0], 6)
        self.assertAlmostEqual(0, resulting_torque[1], 6)



if __name__ == '__main__':
    unittest.main()

