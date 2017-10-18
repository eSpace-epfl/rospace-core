#   Converter between various earth centered coordinats
#   Author: Michael Pantic, michael.pantic@gmail.com
#    Author: Jean Noel Pittet  (Conversion Function ITRF-TEME)
#   Author: Christophe Paccolat (original oe-tools)
#   License: TBD

import numpy as np
import datetime
import copy
from sgp4.propagation import _gstime as greenwichSiderealTime
from Cartesian import *
from OrbitalElements import *
from SphericalEarth import *
from Constants import *

class Converter:
    # Generic main convert method
    @staticmethod
    def convert(source, target):
        if isinstance(source, Cartesian) and isinstance(target, Cartesian):
            Converter.convertCartesianToCartesian(source,target)

        elif isinstance(source, Cartesian) and isinstance(target, SphericalEarth):
            # create itrf
            itrf = CartesianITRF()
            #convert whatever cartesian frame to ITRF
            Converter.convert(source, itrf)
            Converter.convertITRFtoSpherical(itrf, target)

        elif isinstance(source, OrbitalElements) and isinstance(target, Cartesian):
            source.toCartesian(target)
            target.frame = CartesianFrame.TEME

        elif isinstance(source, Cartesian) and isinstance(target, OrbitalElements):
            # crate TEME Cartesian
            teme = CartesianTEME()

            # Convert whatever cartesian frame to TEME
            Converter.convert(source, teme)

            # Init orbital elements from teme cartesian
            target.fromCartesian(teme)

        else:
            raise Exception("Not implemented conversion")


    @staticmethod
    def convertCartesianToCartesian(source,target):
        if(source.frame == target.frame):
            # copy all data to target object
            target.__dict__.update(source.__dict__)

        elif(source.frame == CartesianFrame.TEME and
        target.frame == CartesianFrame.ITRF):
            Converter.convertTEMEtoITRF(source, target)
        else:
            raise Exception("Not implemented cartesian conversion")


    @staticmethod
    def convertTEMEtoITRF(source, target):
        # from Code of previous student ....xy...
        xp = 0.000
        yp = 0.000

        # TODO: Not sure if this converts correctly, as it does not
        # incorporate the mean equinox direction?

        # Rotate around z based on sidereal time in greenwich
        gmst1982 = greenwichSiderealTime(target.epochJD)
        R_gmst = R_z(gmst1982)

        # Calculation rotation of polar motion (not in used at the moment)
        W = R_x(-yp).dot(R_y(-xp))

        # Do rotations
        target.R = R_gmst.T.dot(W.T).dot(source.R.reshape(3,1))
        target.V = R_gmst.T.dot(W.T).dot(source.V.reshape(3,1))
        target.epochJD = source.epochJD

    @staticmethod
    def convertITRFtoSpherical(source, target):
        target.lat = np.arctan(source.R[2] / np.sqrt(source.R[1] ** 2 + source.R[0] ** 2))[0,0]
        target.lon = np.arctan2(source.R[1], source.R[0])[0,0]
        
    @staticmethod
    def fromOEMessage(msg):
        tf_obj = OrbitalElements()

        tf_obj.i = np.deg2rad(msg.inclination)
        tf_obj.w = np.deg2rad(msg.arg_perigee)
        tf_obj.omega = np.deg2rad(msg.raan)
        tf_obj.t = np.deg2rad(msg.true_anomaly)
        tf_obj.a = msg.semimajoraxis
        tf_obj.e = msg.eccentricity
        return tf_obj


