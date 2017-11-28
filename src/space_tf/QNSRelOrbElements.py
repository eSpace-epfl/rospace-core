"""
Module that contains Classes for Relative Orbital Elements (ROE)

References:

    [1] New State Transition Matrices for Relative Motion of Spacecraft Formations in Perturbed Orbits,
        A. Koenig, T. Guffanti, S. D'Amico, AIAA 2016-5635
    [2] Improved Maneuver-free approach to angles-only navigation for space rendezvous,
        J. Sullivan, A.W. Koenig, S. D'Amico, AAS 16-530

"""

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
from BaseState import *


class QNSRelOrbElements(BaseState):
    """ Holds Relative Orbital Elements in the Quasi-Non Singular Formulation

    IMPORTANT: Formulation is not unique if target is in equatorial orbit (ie. target.i =  0)

    """
    def __init__(self):
        super(QNSRelOrbElements, self).__init__()
        self.dA = 0
        self.dL = 0
        self.dEx = 0
        self.dEy = 0
        self.dIx = 0
        self.dIy = 0

    def from_vector(self, vector):
        """Method to initialize from numpy vector [dA,dL,dEx,dEy,dIx,dIy]"""
        self.dA = vector[0]
        self.dL = vector[1]
        self.dEx = vector[2]
        self.dEy = vector[3]
        self.dIx = vector[4]
        self.dIy = vector[5]

    def as_vector(self):
        """Initializes from a numpy vector [dA,dL,dEx,dEy,dIx,dIy]"""
        vector = np.zeros([6, 1])
        vector[0] = self.dA
        vector[1] = self.dL
        vector[2] = self.dEx
        vector[3] = self.dEy
        vector[4] = self.dIx
        vector[5] = self.dIy
        return vector

    def from_keporb(self, target, chaser):
        """Calculate relative orbital elements based on target and chaser absolute elements.

        Note:
            In order to calculate mean relative orbital elements, use mean elements for target and chaser!

        Args:
            target (KepOrbElem): Mean Orbital Elements of target
            chaser (KepOrbElem): Mean Orbital Elements of chaser

        """

        # shorter writing
        t = target
        c = chaser

        if abs(t.i) < sys.float_info.epsilon:
            raise ValueError("Target in equitorial orbit. QNS not well-defined.")

        if target.frame != chaser.frame:
            raise ValueError("Chaser and Target must be in same frame")

        # calculate values
        self.time = target.time
        self.frame = target.frame
        self.dA = (float(t.a) - c.a) / c.a  # a_t sometimes happens to be a int
        self.dL = (t.m + t.w) - (c.m + c.w) + (t.O - c.O) * np.cos(c.i)
        self.dEx = t.e * np.cos(t.w) - c.e * np.cos(c.w)
        self.dEy = t.e * np.sin(t.w) - c.e * np.sin(c.w)
        self.dIx = t.i - c.i
        self.dIy = (t.O - c.O) * np.sin(c.i)
