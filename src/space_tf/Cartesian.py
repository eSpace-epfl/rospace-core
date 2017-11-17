#   Classes for storing cartesian coordinates with epoch and frame information
#   Author: Michael Pantic, michael.pantic@gmail.com
#   License: TBD
import datetime
import numpy as np
from BaseState import *
from Constants import *


class CartesianFrame:
    UNDEF = "UNDEF"
    TEME = "TEME"
    ITRF = "ITRF"
    LVLH = "LVLH"


class Cartesian(BaseState):
    """
    Holds a space craft state vector in cartesian coordinates.
    Includes transformations to initialize from other coordinate types
    """

    def __init__(self):
        super(Cartesian, self).__init__()

        self.R = np.array([0, 0, 0])
        '''numpy.array: Position in [km]'''

        self.V = np.array([0, 0, 0])
        '''numpy.array: Velocity in [km/s]'''

        self.frame = CartesianFrame.UNDEF

    def from_keporb(self, keporb):
        """Sets cartesian state from Keplerian Orbital elements.

        Args:
            keporb (space_tf.KepOrbElem):  Spacecraft position

        """
        p = keporb.a * (1 - keporb.e ** 2)

        # Position in perifocal frame, then rotate to proper orbital plane
        R_per = np.array([p * np.cos(keporb.v), p * np.sin(keporb.v), 0]) / (1.0 + keporb.e * np.cos(keporb.v))
        self.R = R_z(keporb.O).dot(R_x(keporb.i)).dot(R_z(keporb.w)).dot(R_per)

        # speed in perifocal frame, then rotate
        V_per = np.array([-np.sin(keporb.v), keporb.e + np.cos(keporb.v), 0]) * np.sqrt(Constants.mu_earth / p)
        self.V = R_z(keporb.O).dot(R_x(keporb.i)).dot(R_z(keporb.w)).dot(V_per)

    def get_lof(self):
        """Calculates local orbital frame

        Calculates the 3 base vectors (i,j,k) in the current frame.
        Base vector i is aligned with the line between spacecraft and center of earth
        Base vector j is aligned with the instantaneous velocity vector
        Base vector k is crossproduct(i,j).

        Returns:
            numpy.array: 3x3 Matrix containing base vectors i,j,k in current frame

        """
        # calculates base vectors of LOF in current frame
        # calculate 3 basis vectors
        i = self.R / np.linalg.norm(self.R)
        j = self.V / np.linalg.norm(self.V)
        k = np.cross(i, j)

        B = np.identity(3)
        B[0, 0:3] = i
        B[1, 0:3] = j
        B[2, 0:3] = k

        return B


class CartesianTEME(Cartesian):
    def __init__(self):
        super(CartesianTEME, self).__init__()
        self.frame = CartesianFrame.TEME


class CartesianITRF(Cartesian):
    def __init__(self):
        super(CartesianITRF, self).__init__()
        self.frame = CartesianFrame.ITRF


class CartesianLVLH(Cartesian):
    """Class that holds relative coordinates in the local vertical local horizontal frame.

    Allows to calculate and store R-bar/V-bar/H-bar.

    """

    @property
    def rbar(self):
        return self.R[0]

    @rbar.setter
    def rbar(self, value):
        self.R[0] = value

    @property
    def vbar(self):
        return self.R[1]

    @vbar.setter
    def vbar(self, value):
        self.R[1] = value

    @property
    def hbar(self):
        return self.R[2]

    @hbar.setter
    def hbar(self, value):
        self.R[2] = value

    def __init__(self):
        super(CartesianLVLH, self).__init__()
        self.frame = CartesianFrame.LVLH

    def from_keporb(self, keporb):
        raise RuntimeError("Method currently not supported")

    def from_cartesian_pair(self, chaser, target):
        """Initializes relative coordinates in target lvlh frame.

        Args:
            chaser (Cartesian): Chaser state
            target (Cartesian): Target state (base for LVLH)
        """

        # calculate target lvlh
        R_lvlh = target.get_lof()

        # get vector from target to chaser in TEME in [km]
        self.R = (chaser.R - target.R)

        # rotate into lvlh
        self.R = R_lvlh.dot(self.R)
