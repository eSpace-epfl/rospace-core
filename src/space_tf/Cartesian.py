#   Classes for storing cartesian coordinates with epoch and frame information
#   Author: Michael Pantic, michael.pantic@gmail.com
#   License: TBD
import datetime
import numpy as np
from BaseState import *
from Constants import *

class CartesianFrame:
    UNDEF = 0
    TEME = 1
    ITRF = 2
    LVLH = 3



class Cartesian(BaseState):
    def __init__(self):
        super(Cartesian, self).__init__()

        self.R = np.array([0, 0, 0])  # position
        self.V = np.array([0, 0, 0])  # velocity
        self.epochJD = 0  # julian date epoch
        self.frame = CartesianFrame.UNDEF  # frame type

    def setEpochFromDate(self, date):
        J2000 = 2451545.0
        J2000_date = datetime.datetime(2000, 1, 1, 11, 59, 0)  # UTC time of J2000
        delta = date - J2000_date
        self.epochJD = J2000 + delta.total_seconds() / (60.0 * 60 * 24)


    def from_keporb(self, keporb):
        # Calculates cartesian coordinates based on current orbital elements
        p = keporb.a * (1 - keporb.e ** 2)

        # Position in perifocal frame, then rotate to proper orbital plane
        R_per = np.array([p * np.cos(keporb.v), p * np.sin(keporb.v), 0]) / (1.0 + keporb.e * np.cos(keporb.v))
        self.R = R_z(keporb.O).dot(R_x(keporb.i)).dot(R_z(keporb.w)).dot(R_per)

        # speed in perifocal frame, then rotate
        V_per = np.array([-np.sin(keporb.v), keporb.e + np.cos(keporb.v), 0]) * np.sqrt(Constants.mu_earth / p)
        self.V = R_z(keporb.O).dot(R_x(keporb.i)).dot(R_z(keporb.w)).dot(V_per)

    def get_lof(self):
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

    def from_cartesian_pair(self, chaser, target):

        # calculate target lvlh
        R_lvlh = target.get_lof()

        # get vector from target to chaser in TEME in [km]
        self.R = (chaser.R - target.R)

        # rotate into lvlh
        self.R = R_lvlh.dot(self.R)
