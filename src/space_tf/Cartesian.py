#   Classes for storing cartesian coordinates with epoch and frame information
#   Author: Michael Pantic, michael.pantic@gmail.com
#   License: TBD
import datetime
import numpy as np

class CartesianFrame:
    UNDEF = 0
    TEME = 1
    ITRF = 2


class Cartesian(object):
    def __init__(self):
        self.R = np.array([0,0,0])    # position
        self.V = np.array([0,0,0])    # velocity
        self.epochJD = 0              # julian date epoch
        self.frame = CartesianFrame.UNDEF # frame type

    def setEpochFromDate(self, date):
        J2000 = 2451545.0
        J2000_date = datetime.datetime(2000,1,1,11,59,0) #UTC time of J2000
        delta = date - J2000_date
        self.epochJD = J2000 + delta.total_seconds()/(60.0*60*24)


class CartesianTEME(Cartesian):
    def __init__(self):
        super(CartesianTEME, self).__init__()
        self.frame = CartesianFrame.TEME


class CartesianITRF(Cartesian):
    def __init__(self):
        super(CartesianITRF, self).__init__()
        self.frame = CartesianFrame.ITRF
