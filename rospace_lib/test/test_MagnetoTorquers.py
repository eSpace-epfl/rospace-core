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
        B_field = np.array([10,0, 0])
        I = 22.7*10**-3

        torquer.set_B_field(B_field)
        torquer.set_I(I)

        print torquer.get_torque()



if __name__ == '__main__':
    unittest.main()

