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

    def from_lvlh_frame(self, target, lvlh_chaser):
        """Initialize cartesian frame from LVLH and target cartesian
  	
        Args:
            target (space_tf.Cartesian):  Target Cartesian position
            lvlh_chaser (space_tf.CartesianLVLH):  Chaser LVLH Cartesian position
        """
        # Get rotation matrix from target absolute position
        R_TL_T = target.get_lof()

        R_T_TL = np.linalg.inv(R_TL_T)

        p_TL = lvlh_chaser.R
        v_TL = lvlh_chaser.V

        self.R = target.R + R_T_TL.dot(p_TL)
        self.V = target.V + R_T_TL.dot(v_TL)

    def from_keporb(self, keporb):
        """Sets cartesian state from Keplerian Orbital elements.

        Args:
            keporb (space_tf.KepOrbElem):  Spacecraft position

        """
        p = keporb.a - (keporb.a * keporb.e) * keporb.e

        # Position in perifocal frame, then rotate to proper orbital plane
        R_per = np.array([p * np.cos(keporb.v), p * np.sin(keporb.v), 0.0]) / (1.0 + keporb.e * np.cos(keporb.v))
        self.R = R_z(keporb.O).dot(R_x(keporb.i)).dot(R_z(keporb.w)).dot(R_per)

        # speed in perifocal frame, then rotate
        V_per = np.array([-np.sin(keporb.v), keporb.e + np.cos(keporb.v), 0.0]) * np.sqrt(Constants.mu_earth / p)
        self.V = R_z(keporb.O).dot(R_x(keporb.i)).dot(R_z(keporb.w)).dot(V_per)

    def get_lof(self):
        """Calculates local orbital frame

        Calculates the 3 base vectors (i,j,k) in the current frame.
        Base vector i is aligned with the line between spacecraft and center of earth
        Base vector j is the negative crossproduct(i,k)
        Base vector k is aligned with the orbit angular momentum

        Returns:
            numpy.array: 3x3 Matrix containing base vectors i,j,k in current frame

        """
        # calculates base vectors of LOF in current frame
        # calculate 3 basis vectors
        i = self.R / np.linalg.norm(self.R)
        H = np.cross(self.R, self.V)
        k = H / np.linalg.norm(H)
        j = -np.cross(i, k)

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

        Two main definitions are available in literature, one with the
        R vector pointing towards the Earth while the other pointing outwards.

        According to:
            Fehse W. et al, Automated Rendezvous and Docking of Spacecraft

        vector R points towards the Earth, and is referred as x-axis. The y-axis is then
        the direction of angular momentum, and z-axis is obtained by cross product.
        In this reference, however, only circular orbits are considered.

        According to:
            Alfriend K.T. et al, Spacecraft Formation Flying

        vector R points outwards, and is again referred as x-axis. In this case the y-axis
        is obtained by negative cross product of x-axis and z-axis, where the z-axis is
        in the direction of the angular momentum. In this case also elliptical orbits are
        considered.

        There the conversion is implemented according to Alfriend et al.

        Args:
            chaser (Cartesian): Chaser state
            target (Cartesian): Target state (base for LVLH)
        """

        # calculate target lvlh
        R_TL_T = target.get_lof()

        # get vector from target to chaser in TEME in [km]
        p_T_C = (chaser.R - target.R)

        # get the relative velocity of chaser w.r.t target in TEME [km/s]
        v_T_C = chaser.V - target.V

        # get chaser position and velocity in target LVLH frame
        p_TL_C = R_TL_T.dot(p_T_C)
        v_TL_C = R_TL_T.dot(v_T_C)
        self.R = p_TL_C
        self.V = v_TL_C


class CartesianLVC(Cartesian):
    """ Class for the Local-Vertical-Curvilinear reference frame

    Can it be useful? (mpantic:)Yes. To do later ;-)

    """

    def from_lvlh_frame(self, target, lvlh_chaser):
        """
        Args:
            target:
            lvlh_chaser:
        """

        # Convert chaser to cartesian
        cart_chaser = Cartesian()
        cart_chaser.from_lvlh_frame(target, lvlh_chaser)

        # Convert both in perifocal frame


        # Evaluate radius difference
        

    def from_cartesian_pair(self, target, chaser):
        pass
