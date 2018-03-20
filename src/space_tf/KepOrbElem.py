# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""
Module that contains Classes for Keplerian Orbital Elements.

References:

    [1] Orbital Mechanics for Engineering Students, 3rd Edition, Howard D. Curtis, ISBN 978-0-08-097747-8
    [2] Analytical Mechanics of Space Systems, H. Schaub and J.L. Junkins, AIAA Education Series, 2003
    [3] New State Transition Matrices for Relative Motion of Spacecraft Formations in Perturbed Orbits,
        A. Koenig, T. Guffanti, S. D'Amico, AIAA 2016-5635
"""
import orekit

from orekit.pyhelpers import setup_orekit_curdir
import inspect
import thread
from threading import Thread
from org.orekit.propagation.semianalytical.dsst import DSSTPropagator
from java.util import Arrays
from java.util import Collections
from java.util import ArrayList
from java.util import HashSet
from orekit import JArray_double
from orekit import JArray


from org.orekit.attitudes import BodyCenterPointing
from org.orekit.bodies import  OneAxisEllipsoid
from org.orekit.frames import  FramesFactory
from org.orekit.data import DataProvidersManager, ZipJarCrawler
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.orbits import KeplerianOrbit
from org.orekit.utils import Constants as Cst
from org.orekit.propagation.analytical import EcksteinHechlerPropagator
from org.orekit.propagation.analytical.tle import TLEPropagator
from org.orekit.propagation.conversion import FiniteDifferencePropagatorConverter
from org.orekit.propagation.conversion import TLEPropagatorBuilder
from datetime import datetime
from org.orekit.forces.gravity.potential import UnnormalizedSphericalHarmonicsProvider
from org.orekit.propagation.semianalytical.dsst.forces import DSSTZonal
from org.orekit.propagation.semianalytical.dsst.forces import DSSTForceModel
from org.orekit.propagation.semianalytical.dsst.utilities import AuxiliaryElements

from org.orekit.propagation import SpacecraftState
from org.orekit.orbits import OrbitType, PositionAngle, EquinoctialOrbit
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.utils import IERSConventions


from . import *
from threading import RLock, Lock
import numpy as np

class KepOrbElem(BaseState):

    vm = None
    provider = None
    lock = Lock()

    """
    Class that contains a spacecraft state in Keplerian orbital elements.
    Use this class to hold _mean_ orbital elements.
    """
    def __init__(self):
        super(KepOrbElem, self).__init__()

        self.a = 0
        '''Semimajor axis in [km]'''

        self.e = 0
        '''Eccentricity'''

        self.O = 0
        '''Right Ascending of the Ascending Node (RAAN) [rad]'''

        self.w = 0
        '''Argument of Perigeum [rad]'''

        self.i = 0
        '''Inclination [rad]'''

        self.period = 0
        '''Orbital Period. Might not always be set.'''

        # Internal members
        self._lock = RLock() # Lock for thread safety
        self._orekit_lock = Lock()

        self._E = None  # eccentric anomaly
        self._m = None  # mean anomaly
        self._v = None  # true anomaly

    @property
    def E(self):
        """Eccentric Anomaly [rad]"""
        self._lock.acquire()
        if self._E is None:
            self._sync_anomalies()
        self._lock.release()
        return self._E

    @E.setter
    def E(self, value):
        self._lock.acquire()
        self._E = value
        self._m = None
        self._v = None
        self._sync_anomalies()
        self._lock.release()

    @property
    def m(self):
        """Mean Anomaly [rad]"""
        self._lock.acquire()
        if self._m is None:
            self._sync_anomalies()
        self._lock.release()
        return self._m

    @m.setter
    def m(self, value):
        self._lock.acquire()
        self._E = None
        self._m = value
        self._v = None
        self._sync_anomalies()
        self._lock.release()

    @property
    def v(self):
        """True Anomaly [rad]"""
        self._lock.acquire()
        if self._v is None:
            self._sync_anomalies()
        self._lock.release()
        return self._v

    @v.setter
    def v(self, value):

        self._lock.acquire()
        # Reset other anomalies (the are not valid anymore)s
        self._E = None
        self._m = None

        # Set true anomaly
        self._v = value

        # sync anomalies
        self._sync_anomalies()
        self._lock.release()

    def _sync_anomalies(self):
        """ Calculates unknown anomalies from given anomalies.

            To calculate v from m and vice versa, the eccentric anomaly is needed.
            Function is run recursively until all anomalies are set and valid

        """
        if self._E is None and self._m is None and self._v is None:
            raise ValueError("No anomaly set. Cannot sync.")

        if self._E is not None and \
                self._m is not None \
                and self._v is not None:
            return

        if self._v is None and self._E is not None:
            self._calc_v_from_E()

        if self._E is None and self._v is not None:
            self._calc_E_from_v()

        if self._E is None and self._m is not None:
            self._calc_E_from_m()

        if self._m is None and self._E is not None:
            self._calc_m_from_E()

        # recursively run until everything is set
        self._sync_anomalies()

    def _calc_E_from_v(self):
        """Calculates Eccentric anomaly from true anomaly.

        Closed form, analytical solution.
        Prerequisites: e and v
        """
        self._E = np.arctan2(np.sqrt(1.0-self.e**2)*np.sin(self._v), self.e+np.cos(self._v))

        if self._E < 0:
            self._E = self._E + np.pi * 2.0

    def _calc_v_from_E(self):
        """Calculates True anomaly from Eccentric anomaly.

        Closed form, analytical solution.
        Prerequisites: e and E
        """
        self._v = 2.0 * np.arctan2(np.sqrt(1.0 + self.e) * np.sin(self._E / 2.0),
                                 np.sqrt(1.0 - self.e) * np.cos(self._E / 2.0))

        if self._v < 0:
            self._v = self._v + np.pi * 2.0

    def _calc_E_from_m(self):
        """Calculates Eccentric anomaly from Mean anomaly

        Uses a Newton-Raphson iteration to solve Kepler's Equation.
        Source: Algorithm 3.1 in [1]

        Prerequisites: m and e

        """
        if self._m < np.pi:
            self._E = self._m + self.e / 2.0
        else:
            self._E = self._m - self.e / 2.0

        max_int = 20  # maximum number of iterations

        while max_int > 1:
            fE = self._E - self.e * np.sin(self._E) - self._m
            fpE = 1.0 - self.e * np.cos(self._E)
            ratio = fE / fpE
            max_int = max_int - 1

            # check if ratio is small enough
            if abs(ratio) > 1e-15:
                self._E = self._E - ratio
            else:
                break

        if self._E < 0:
            self._E = self._E + np.pi * 2.0

    def _calc_m_from_E(self):
        """Calculates Mean anomaly Eccentric anomaly using Kepler's Equation.

        Prerequisites: E and e
        """
        self._m = self._E - self.e*np.sin(self._E)

        if self._m < 0:
            self._m = self._m + np.pi * 2.0

    def from_tle(self, i_tle, omega_tle, e_tle, m_tle, w_tle, mean_motion_tle):
        """ Initializes object from TLE elements (TLE = Two Line Elements)

        Args:
            i_tle: Inclination [rad]
            omega_tle: RAAN [rad]
            e_tle: Eccentricity
            m_tle: Mean Anomaly [rad]
            w_tle: Arg of Perigee [rad]
            mean_motion_tle: Mean Motion [rad/s]

        """
        # assign known data
        self.i = i_tle
        self.e = e_tle
        self._m = m_tle
        self.w = w_tle
        self.O = omega_tle
        self.period = 1.0 / mean_motion_tle * 60 * 60 * 24

        # update unknown data in proper order
        self.a = np.power(self.period ** 2 * Constants.mu_earth / (4 * np.pi ** 2), 1 / 3.0)

    def from_message(self, msg):
        """Initialize object from a ROS-Message that uses Degrees"""
        self.i = np.deg2rad(msg.inclination)
        self.w = np.deg2rad(msg.arg_perigee)
        self.O = np.deg2rad(msg.raan)
        self.a = msg.semimajoraxis
        self.e = msg.eccentricity

        # assign latest!
        self.v = np.deg2rad(msg.true_anomaly)

    def from_cartesian(self, cart):
        """Initialize object from a cartesian state vector (pos + speed).

        Source: Algorithm 4.1 in [1]

        Args:
            cart (Cartesian): Cartesian State Vector with R and V set.

        """
        K = np.array([0, 0, 1.0])  # 3rd basis vector

        # 1. Calc distance
        r = np.linalg.norm(cart.R, ord=2)

        # 2. Calc  speed
        v = np.linalg.norm(cart.V, ord=2)

        # 3. Calc radial velocity
        v_r = np.dot(cart.R.flat, cart.V.flat) / r

        # 4. Calc specific angular momentum
        H = np.cross(cart.R.flat, cart.V.flat)

        # 5. Calc magnitude of specific angular momentum
        h = np.linalg.norm(H, ord=2)

        # 6. Calc inclination
        self.i = np.arccos(H[2] / h)

        # 7. Calculate Node line
        N = np.cross(K, H)

        # 8. Calculate magnitude of N
        n = np.linalg.norm(N, ord=2)

        # 9. calculate RAAN
        self.O = np.arccos(N[0] / n)
        if N[1] < 0:
            self.O = 2 * np.pi - self.O

        # 10. calculate eccentricity vector  / 11. Calc eccentricity
        E = 1 / Constants.mu_earth * ((v ** 2 - Constants.mu_earth / r) * cart.R.flat - r * v_r * cart.V.flat)
        self.e = np.linalg.norm(E, ord=2)

        # direct form:
        # self.e = 1 / Constants.mu_earth * np.sqrt(
        #    (2 * Constants.mu_earth - r * v ** 2) * r * v_r ** 2 + (Constants.mu_earth - r * v ** 2) ** 2)

        # 11. Calculate arg. of perigee
        P = E / (n * self.e)

        self.w = np.arccos(np.dot(N, P))
        if E[2] < 0:
            self.w = 2 * np.pi - self.w

        # 12. Calculate the true anomaly
        # p2 = np.log(self.e)+np.log(r)
        self.v = np.arccos(np.dot(E, cart.R.flat) / (self.e*r))
        if v_r < 0:
            self.v = 2 * np.pi - self.v

        # 13. Calculate semimajor axis
        rp = h ** 2 / Constants.mu_earth * 1 / (1 + self.e)
        ra = h ** 2 / Constants.mu_earth * 1 / (1 - self.e)
        self.a = 0.5 * (rp + ra)

        # 14. Calculate period (in [s])
        self.period = 2 * np.pi / np.sqrt(Constants.mu_earth) * (pow(self.a, 1.5))

    def from_qns_relative(self, qns, chaser):
        """Initialize object using Quasi-Non-Singular Relative Orbital Elements
        and the corresponding absolute coordinates.

        Source: Can be calculated based on the definition of QNS in [3]
        Args:
            qns (QNSRelOrbElements): Quasi-Non-Singular Relative Orbital Elements
            chaser (KepOrbElem): Absolute coordinates (to which qns are relative)

        """
        u_c = chaser.m + chaser.w

        # calculate absolute orbital elements.
        self.a = chaser.a * (1.0 + qns.dA)
        e_c_c_wc = chaser.e * np.cos(chaser.w) + qns.dEx
        e_c_s_wc = chaser.e * np.sin(chaser.w) + qns.dEy
        self.i = chaser.i + qns.dIx
        self.O = qns.dIy * np.sin(chaser.i) + chaser.O

        self.e = np.sqrt(e_c_c_wc ** 2 + e_c_s_wc ** 2)
        self.w = np.arctan2(e_c_s_wc, e_c_c_wc)

        u_t = u_c + (qns.dL - np.cos(chaser.i) * (self.O - chaser.O))
        m = u_t - self.w

        self._v = None
        self._E = None

        if m < 0:
            self._m = m + 2.0*np.pi
        else:
            self._m = m

    def as_array_true(self):
        """Return as numpy array [a,v,e,w,i,O] """
        return np.array([[self.a, self.v, self.e, self.w, self.i, self.O]]).T

    def as_array_mean(self):
        """Return as numpy array [a,m,e,w,i,O] """
        return np.array([[self.a, self.m, self.e, self.w, self.i, self.O]]).T

    def __str__(self):
        return "a: {0}, e: {1}, i: {2}, w: {3}, O: {4}, v: {5}, E: {6}, M: {7}".format(self.a,
                                                                                       self.e,
                                                                                       self.i,
                                                                                       self.w,
                                                                                       self.O,
                                                                                       self.v,
                                                                                       self.E,
                                                                                       self.m)

    def from_osc_elems(self, osc, mode="ore"):
        """ Initialize mean elements from a set of osculating elements"""
        self.osc_elems_transformation_redir(osc, True, mode)

    def osc_elems_transformation_redir(self,other, dir, mode):

        if mode == "ore":
            self.osc_elems_transformation_ore(other,dir)
        elif mode == "schaub":
            self.osc_elems_transformation(other, dir)
        elif mode == "null":
            self.osc_elems_transformation_null(other, dir)
        else:
            raise("Osculating transform mode not known")

    def osc_elems_transformation_null(self,other, dir):
        self.a = other.a
        self.e = other.e
        self.v = other.v
        self.i = other.i
        self.O = other.O
        self.w = other.w


    def osc_elems_transformation_ore(self, other, dir):

        self._orekit_lock.acquire()
        if KepOrbElem.vm is None:
            KepOrbElem.vm =  orekit.initVM()
            setup_orekit_curdir()
            KepOrbElem.provider = GravityFieldFactory.getUnnormalizedProvider(6,6)

        KepOrbElem.vm.attachCurrentThread()

        utc = TimeScalesFactory.getUTC()



        orekit_date = AbsoluteDate(2017,
                                   1,
                                   1,
                                   12,
                                   1,
                                   1.0,
                                   utc)
        inertialFrame = FramesFactory.getEME2000()
        a = float(other.a)
        e = float(other.e)
        i = float(other.i)
        w = float(other.w)
        O = float(other.O)
        v = float(other.v)

        initialOrbit = KeplerianOrbit(a * 1000.0, e, i, w, O, v,
                                      PositionAngle.TRUE,
                                      inertialFrame, orekit_date, Constants.mu_earth*1e9)

        initialState = SpacecraftState(initialOrbit, 1.0)

        #zonal_forces= DSSTZonal(provider,2,1,5)
        zonal_forces = DSSTZonal(KepOrbElem.provider, 6, 4, 6)
        forces = ArrayList()
        forces.add(zonal_forces)
        try:
            equinoctial = None
            if dir:

                equinoctial = DSSTPropagator.computeMeanState(initialState, None, forces)
            else:

                equinoctial = DSSTPropagator.computeOsculatingState(initialState, None, forces)

            newOrbit = KeplerianOrbit(equinoctial.getOrbit())

            self.a = newOrbit.getA()/1000.0
            self.e = newOrbit.getE()
            self.i = newOrbit.getI()
            self.w = newOrbit.getPerigeeArgument()
            self.O = newOrbit.getRightAscensionOfAscendingNode()
            self.v = newOrbit.getAnomaly(PositionAngle.TRUE)

            # correct ranges

            if self.i < 0:
                self.i += 2*np.pi

            if self.w < 0:
                self.w += 2*np.pi

            if self.O < 0:
                self.O += 2*np.pi

            if self.v < 0:
                self.v += 2*np.pi


        finally:
            self._orekit_lock.release()


    def osc_elems_transformation(self, other, dir):
        """ Approximation to convert between mean and osculating orbital elements.

        Note that this is a first order approximation and does not map 1 to 1.
        Also, it might not be well behaving for close to circular orbits.

        Source: Appendix G in [2]


        Args:
            other: Other orbital elements
            dir: if dir == True: other are osculating, this are mean elements,
                 if dir == False: other are mean, this are osculating elements

        """
        eta = np.sqrt(1.0 - other.e ** 2)
        gma_2 = Constants.J_2 / 2.0 * (Constants.R_earth / other.a) ** 2  # G.296

        if dir: # if we are mapping osc to mean, switch sign of gma_2
            gma_2 = -gma_2  # G.297

        gma_2_p = gma_2 / eta ** 4  # G.298
        c_i = np.cos(other.i)
        c_v = np.cos(other.v)
        a_r = (1.0 + other.e * np.cos(other.v)) / (eta ** 2)  # G.301

        # calculate osculating semi-major axis based on series expansion
        a_1 = (3.0 * c_i ** 2 - 1) * (a_r ** 3 - 1.0 / eta ** 3)
        a_2 = 3.0 * (1 - c_i ** 2) * a_r ** 3 * np.cos(2.0 * other.w + 2.0 * other.v)
        self.a = other.a + other.a * gma_2 * (a_1 + a_2)  # G.302

        # calculate intermediate for d_e
        d_e1 = gma_2_p / 8.0 * other.e * eta ** 2 * (
        1.0 - 11 * c_i ** 2 - 40.0 * ((c_i ** 4) / (1.0 - 5.0 * c_i ** 2))) * np.cos(2.0 * other.w)  # G.303
        fe_1 = (3.0 * c_i ** 2 - 1.0) / eta ** 6
        e_1 = other.e * eta + other.e / (1.0 + eta) + 3 * c_v + 3 * other.e * c_v ** 2 + other.e ** 2 * c_v ** 3
        fe_2 = 3.0 * (1.0 - c_i ** 2) / eta ** 6
        fe_3 = gma_2_p * (1.0 - c_i ** 2)
        e_2 = (other.e + 3 * c_v + 3 * other.e * c_v ** 2 + other.e ** 2 * c_v ** 3)*np.cos(2*other.w+2*other.v)
        e_3 = 3 * np.cos(2.0 * other.w + other.v) + np.cos(2.0 * other.w + 3.0 * other.v)

        d_e = d_e1 + (eta ** 2 / 2.0) * (gma_2 * (fe_1 * e_1 + fe_2 * e_2) - fe_3 * e_3)  # G.304

        fi_1 = gma_2_p / 2.0 * c_i * np.sqrt(1.0 - c_i ** 2)
        i_1 = 3.0 * np.cos(2.0 * other.w + 2.0 * other.v) + 3.0 * other.e * np.cos(
            2.0 * other.w + other.v) + other.e * np.cos(2.0 * other.w + 3.0 * other.v)
        d_i = (other.e * d_e1) / (eta ** 2 * np.tan(other.i)) + fi_1 * i_1


        # formula G.306
        if other.m > 2*np.pi:
            other.m = other.m - 2*np.pi

        if other.w > 2*np.pi:
            other.w = other.w - 2*np.pi

        if other.O > 2 * np.pi:
            other.O = other.O - 2 * np.pi

        if other.w < 0:
            other.w = other.w + 2 * np.pi

        if other.O < 0:
            other.O = other.O + 2 * np.pi

        MwO = other.m + other.w + other.O + \
              gma_2_p / 8.0 * eta ** 3 * (1.0 - 11.0 * c_i ** 2 - 40.0 * (c_i ** 4 / (1.0 - 5.0 * c_i ** 2))) \
              - gma_2_p / 16.0 * (2.0 + other.e ** 2 - 11 * (2.0 + 3.0 * other.e ** 2) * c_i ** 2 \
                                  - 40.0 * (2.0 + 5.0 * other.e ** 2) * (
                                  c_i ** 4 / (1.0 - 5.0 * c_i ** 2)) - 400.0 * other.e ** 2 * (
                                  c_i ** 6 / (1.0 - 5.0 * c_i ** 2) ** 2)) \
              + gma_2_p / 4.0 * (-6.0 * (1.0 - 5.0 * c_i ** 2) * (other.v - other.m + other.e * np.sin(other.v)) \
                                 + (3.0 - 5.0 * c_i ** 2) * (
                                 3.0 * np.sin(2.0 * other.w + 2.0 * other.v) + 3.0 * other.e * np.sin(
                                     2.0 * other.w + other.v) \
                                 + other.e * np.sin(2 * other.w + 3 * other.v))) \
              - gma_2_p / 8.0 * other.e ** 2 * c_i * (
        11.0 + 80.0 * (c_i ** 2 / (1.0 - 5.0 * c_i ** 2)) + 200 * (c_i ** 4 / (1.0 - 5.0 * c_i ** 2) ** 2)) \
              - gma_2_p / 2.0 * c_i * (6.0 * (other.v - other.m + other.e * np.sin(other.v)) \
                                       - 3.0 * np.sin(2.0 * other.w + 2.0 * other.v) - 3.0 * other.e * np.sin(
            2.0 * other.w + other.v) - other.e * np.sin(2.0 * other.w + 3.0 * other.v))



        # formula G.307
        edM = gma_2_p / 8.0 * other.e * eta ** 3 * (1.0 - 11.0 * c_i ** 2 - 40.0 * (c_i ** 4 / (1 - 5 * c_i ** 2))) \
              - gma_2_p / 4.0 * eta ** 3 * (2.0 * (3.0 * c_i ** 2 - 1.0) * ((a_r * eta) ** 2 + a_r + 1) * np.sin(other.v) \
                                          + 3.0 * (1.0 - c_i ** 2) * (
                                          (-(a_r * eta) ** 2 - a_r + 1) * np.sin(2.0 * other.w + other.v) \
                                          + (((a_r * eta) ** 2 + a_r + 1.0 / 3.0) * np.sin(2.0 * other.w + 3.0 * other.v))))

        # formula G.308
        dO = -gma_2_p / 8.0 * other.e ** 2 * c_i * (
        11.0 + 80.0 * (c_i ** 2 / (1 - 5 * c_i ** 2)) + 200.0 * (c_i ** 4 / (1 - 5 * c_i ** 2) ** 2)) \
             - gma_2_p / 2.0 * c_i * (
        6.0 * (other.v - other.m + other.e * np.sin(other.v)) - 3.0 * np.sin(2.0 * other.w + 2.0 * other.v) \
        - 3.0 * other.e * np.sin(2.0 * other.w + other.v) - other.e * np.sin(2.0 * other.w + 3.0 * other.v))

        d_1 = (other.e + d_e) * np.sin(other.m) + edM * np.cos(other.m)  # G.309
        d_2 = (other.e + d_e) * np.cos(other.m) - edM * np.sin(other.m)  # G.310

        m = np.arctan2(d_1, d_2)  # G.311
        if m<0:
            m=2*np.pi+m

        self.e = np.sqrt(d_1 ** 2 + d_2 ** 2)  # G.312

        d_3 = (np.sin(other.i / 2.0) + np.cos(other.i / 2.0) * d_i / 2.0) * np.sin(other.O) + np.sin(
            other.i / 2.0) * dO * np.cos(other.O)  # G.313
        d_4 = (np.sin(other.i / 2.0) + np.cos(other.i / 2.0) * d_i / 2.0) * np.cos(other.O) - np.sin(
            other.i / 2.0) * dO * np.sin(other.O)  # G.314

        self.O = np.arctan2(d_3, d_4)  # G.315

        if self.O < 0:
            self.O = 2*np.pi+self.O

        self.i = 2 * np.arcsin(np.sqrt(d_3 ** 2 + d_4 ** 2))  # G.316
        self.w = MwO - m - self.O

        while self.w > 2*np.pi:
            self.w = self.w - 2*np.pi

        self.m = m  # assign m latest, as other properties might be used for ocnversion!


class OscKepOrbElem(KepOrbElem):
    """
    Class that contains a spacecraft state in Keplerian orbital elements.
    Use this class to hold _osculating_ orbital elements.
    """

    def __init__(self):
        super(OscKepOrbElem, self).__init__()

    def from_mean_elems(self, mean, mode="ore"):
        """ Initialize osculating elements from a set of mean elements"""
        self.osc_elems_transformation_redir(mean, False, mode)
