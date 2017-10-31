#   Class to hold relative orbital elements (ROE)
#   Author: Michael Pantic, michael.pantic@gmail.com
#   License: TBD
# 	Literature: [1] New State Transition Matrices for Relative Motion of
#                   Spacecraft Formations in Perturbed Orbits,
#                   A.W. Koenig, T. Guffanti, S. D'Amico
#                   AIAA 2016-5635
#
#               [2] Improved Maneuver-free approach to angles-only
#                   navigation for space rendezvous,
#                   J. Sullivan, A.W. Koenig, S. D'Amico
#                   AAS 16-530
#
#   Note: This class uses the quasi-nonsingular formulation!
#         It is well-defined for almost all configurations of chaser and target
#         EXCEPT: Target is in equatorial orbit (target.i=0)
#
import sys
import numpy as np

from . import Constants
from . import OrbitalElements


class QNSRelOrbElements:
    def __init__(self):
        self.dA = 0  #
        self.dL = 0
        self.dEx = 0
        self.dEy = 0
        self.dIx = 0
        self.dIy = 0

    def from_absolute(self, target, chaser):
        # Calculate relative coordinates based on
        # target and chaser orbital elements

        # make sure mean anomaly is populated
        target.update_m()
        chaser.update_m()

        # shorter writing
        t = target
        c = chaser

        if abs(t.i) < sys.float_info.epsilon:
            raise ValueError("Target in equitorial orbit. QNS not well-defined.")

        # calculate values
        self.dA = (float(t.a) - c.a) / c.a  # a_t sometimes happens to be a int
        self.dL = (t.m + t.w) - (c.m + c.w) + (t.omega - c.omega) * np.cos(c.i)
        self.dEx = t.e * np.cos(t.w) - c.e * np.cos(c.w)
        self.dEy = t.e * np.sin(t.w) - c.e * np.sin(c.w)
        self.dIx = t.i - c.i
        self.dIy = (t.omega - c.omega) * np.sin(c.i)

    def to_absolute(self, chaser):
        # calculate target absolute orbital elements
        # based on state and chaser absolute orbital elements

        target = OrbitalElements()
        chaser.update_m()
        u_c = chaser.m + chaser.w

        # calculate absolute orbital elements.
        target.a = chaser.a * (1.0 + self.dA)
        e_c_c_wc = chaser.e * np.cos(chaser.w) + self.dEx
        e_c_s_wc = chaser.e * np.sin(chaser.w) + self.dEy
        target.i = chaser.i + self.dIx
        target.omega = self.dIy * np.sin(chaser.i) + chaser.omega

        target.e = np.sqrt(e_c_c_wc ** 2 + e_c_s_wc ** 2)
        target.w = np.arctan2(e_c_s_wc, e_c_c_wc)

        u_t = u_c + (self.dL - np.cos(chaser.i) * (target.omega - chaser.omega))
        target.m = u_t - target.w
        target.update_E()  # calculate Eccentric Anomaly
        target.update_t()  # calculate true anomaly

        return target
