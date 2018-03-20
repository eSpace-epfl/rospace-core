"""
Module that contains Classes for Relative Orbital Elements (ROE)

References:

    [1] New State Transition Matrices for Relative Motion of Spacecraft Formations in Perturbed Orbits,
        A. Koenig, T. Guffanti, S. D'Amico, AIAA 2016-5635
    [2] Improved Maneuver-free approach to angles-only navigation for space rendezvous,
        J. Sullivan, A.W. Koenig, S. D'Amico, AAS 16-530

    Author: Michael Pantic
    License: TBD
"""
import sys
import numpy as np

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
        self.dA = np.asscalar(vector[0])
        self.dL = np.asscalar(vector[1])
        self.dEx = np.asscalar(vector[2])
        self.dEy = np.asscalar(vector[3])
        self.dIx = np.asscalar(vector[4])
        self.dIy = np.asscalar(vector[5])

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
        dl_pre = (t.m + t.w) - (c.m + c.w)


        if dl_pre > np.pi:
            dl_pre -= 2*np.pi

        if dl_pre < -np.pi:
            dl_pre += 2*np.pi

        self.time = target.time
        self.frame = target.frame
        self.dA = (float(t.a) - c.a) / c.a  # a_t sometimes happens to be a int
        self.dL = dl_pre + (t.O - c.O) * np.cos(c.i)
        self.dEx = t.e * np.cos(t.w) - c.e * np.cos(c.w)
        self.dEy = t.e * np.sin(t.w) - c.e * np.sin(c.w)
        self.dIx = t.i - c.i
        self.dIy = (t.O - c.O) * np.sin(c.i)

    def as_scaled(self, a_chaser):
        scaled = QNSRelOrbElements()
        scaled.dA = 1000.0 * a_chaser * self.dA
        scaled.dL = 1000.0 * a_chaser * self.dL
        scaled.dEx = 1000.0 * a_chaser * self.dEx
        scaled.dEy = 1000.0 * a_chaser * self.dEy
        scaled.dIx = 1000.0 * a_chaser * self.dIx
        scaled.dIy = 1000.0 * a_chaser * self.dIy
        return scaled

    def __str__(self):
        return "dA: {0}, dL: {1}, dIx: {2}, dIy: {3}, dEx: {4}, dEy: {5}".format(self.dA,
                                                                                 self.dL,
                                                                                 self.dIx,
                                                                                 self.dIy,
                                                                                 self.dEx,
                                                                                 self.dEy)






