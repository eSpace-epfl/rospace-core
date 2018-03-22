# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Classes for storing cartesian coordinates with epoch and frame information"""
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
    """ Class for the Local-Vertical-Curvilinear reference frame.

        Make sense when the distances from the target are significant. The radial distance is the
        difference between the two radii at a certain instant in time. The along-track distance is defined
        as the difference between the true anomalies, to give a consistent measure of the distance between
        target and chaser. Finally the out-of-plane distance is also defined as the angle between the chaser
        position and the target plane. Everything is then normalized to have values within a similar range.

    """

    def __init__(self):
        super(CartesianLVC, self).__init__()

        self.dR = 0
        self.dV = 0
        self.dH = 0

        self.frame = CartesianFrame.UNDEF

    def from_keporb(self, chaser, target):
        """
            Given chaser and target (mean) orbital elements, evaluate radial distance and true anomaly difference.

        Args:
            chaser (KepOrbElem)
            target (KepOrbElem)
        """

        # Evaluate cartesian from keplerian
        chaser_cartesian = Cartesian()
        chaser_cartesian.from_keporb(chaser)
        target_cartesian = Cartesian()
        target_cartesian.from_keporb(target)

        # Evaluate along-track distance in terms of true anomaly difference
        dv = chaser.v + chaser.w - target.v - target.w

        # Evaluate radial distance w.r.t the position of the target
        p_T = target.a * (1.0 - target.e**2)
        r_T = p_T/(1 + target.e * np.cos(target.v))
        p_C = chaser.a * (1.0 - chaser.e**2)
        r_C = p_C/(1 + chaser.e * np.cos(chaser.v))
        dr = r_C - r_T

        # Evaluate plane angular distance to define the out-of-plane difference
        p_C_TEM = chaser_cartesian.R
        h_T_TEM = np.cross(target_cartesian.R, target_cartesian.V)
        e_p_C_TEM = p_C_TEM / np.linalg.norm(p_C_TEM)
        e_h_T_TEM = h_T_TEM / np.linalg.norm(h_T_TEM)
        dh = np.pi/2.0 - np.arccos(np.dot(e_p_C_TEM, e_h_T_TEM))

        # Assign variables
        self.dR = dr / r_T
        self.dV = dv / (2*np.pi)
        self.dH = dh / (2*np.pi)

    def from_qns(self):
        # TODO
        pass