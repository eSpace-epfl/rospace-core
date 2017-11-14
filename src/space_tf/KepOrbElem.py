# Base Class for Keplerian orbital elements in different formulations

# References:
#
# [1] Orbital Mechanics for Engineering Students, 3rd Edition, Howard D. Curtis, ISBN 978-0-08-097747-8
# [2] Analytical Mechanics of Space Systems, H. Schaub and J.L. Junkins, AIAA Education Series, 2003
#
#



from BaseState import *
from Constants import *
import numpy as np
from threading import RLock


# Base without anomaly
class KepOrbElem(BaseState):
    def __init__(self):
        super(KepOrbElem, self).__init__()
        self.a = 0
        self.e = 0
        self.O = 0
        self.w = 0
        self.i = 0
        self.period = 0

        self.lock = RLock()
        self._E = None  # eccentric anomaly
        self._m = None  # mean anomaly
        self._v = None  # true anomaly

    @property
    def E(self):
        self.lock.acquire()
        if self._E is None:
            self.sync_anomalies()
        self.lock.release()
        return self._E

    @E.setter
    def E(self, value):
        self.lock.acquire()
        self._E = value
        self._m = None
        self._v = None
        self.sync_anomalies()
        self.lock.release()

    @property
    def m(self):
        self.lock.acquire()
        if self._m is None:
            self.sync_anomalies()
        self.lock.release()
        return self._m

    @m.setter
    def m(self, value):
        self.lock.acquire()
        self._E = None
        self._m = value
        self._v = None
        self.sync_anomalies()
        self.lock.release()

    @property
    def v(self):
        self.lock.acquire()
        if self._v is None:
            self.sync_anomalies()
        self.lock.release()
        return self._v

    @v.setter
    def v(self, value):
        self.lock.acquire()
        self._E = None
        self._m = None
        self._v = value
        self.sync_anomalies()
        self.lock.release()

    def sync_anomalies(self):
        if self._E is None and self._m is None and self._v is None:
            raise ValueError("No anomaly set. Cannot sync.")

        if self._E is not None and \
                self._m is not None \
                and self._v is not None:
            return

        if self._v is None and self._E is not None:
            self.calc_v_from_E()

        if self._E is None and self._v is not None:
            self.calc_E_from_v()

        if self._E is None and self._m is not None:
            self.calc_E_from_m()

        if self._m is None and self._E is not None:
            self.calc_m_from_E()

        # recursively run until everything is set
        self.sync_anomalies()

    def calc_E_from_v(self):
        self._E = np.arctan2(np.sqrt(1.0-self.e**2)*np.sin(self._v),self.e+np.cos(self._v))

        if self._E < 0:
            self._E = self._E + np.pi * 2.0

    def calc_v_from_E(self):
        # Calculates v
        # Prerequisities: e and E
        self._v = 2.0 * np.arctan2(np.sqrt(1.0 + self.e) * np.sin(self._E / 2.0),
                                 np.sqrt(1.0 - self.e) * np.cos(self._E / 2.0))

        if self._v < 0:
            self._v = self._v + np.pi * 2.0


    def calc_E_from_m(self):
        # Calculates mean eccentricity
        # Prerequisites: m and e
        # Source: Algorithm 3.1 from [1]
        # Uses Newton-Raphson iteration
        if self._m < np.pi:
            self._E = self._m + self.e / 2.0
        else:
            self._E = self._m - self.e / 2.0

        max_int = 20
        cur_int = 0

        while True:
            fE = self._E - self.e * np.sin(self._E) - self._m
            fpE = 1.0 - self.e * np.cos(self._E)
            ratio = fE / fpE
            if abs(ratio) > 1e-15 and cur_int < max_int:
                self._E = self._E - ratio
                cur_int = cur_int + 1
            else:
                break

        if self._E < 0:
            self._E = self._E + np.pi * 2.0

    def calc_m_from_E(self):
        self._m = self._E - self.e*np.sin(self._E)

        if self._m < 0:
            self._m = self._m + np.pi * 2.0

    def calc_m_from_v(self):
        # Calculates m based on series expansion
        # Prerequisites: v
        # Source: Todo: Celestial Mechanics by W.M. Smart
        self._m = self._v
        self._m = self._m - 2 * self.e * np.sin(self._v)
        self._m = self._m + (0.75 * self.e ** 2 + 0.125 * self.e ** 4) * np.sin(2 * self._v)
        self._m = self._m - (1.0 / 3.0) * self.e ** 3 * np.sin(3 * self._v)
        self._m = self._m + (5.0 / 32.0) * self.e ** 4 * np.sin(4 * self._v)

    def from_tle(self, i_tle, omega_tle, e_tle, m_tle, w_tle, mean_motion_tle):
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
        self.i = np.deg2rad(msg.inclination)
        self.w = np.deg2rad(msg.arg_perigee)
        self.O = np.deg2rad(msg.raan)
        self.a = msg.semimajoraxis
        self.e = msg.eccentricity

        # assign latest!
        self._v = np.deg2rad(msg.true_anomaly)

    def from_cartesian(self, cart):
        # Calculates and initalizes keplerian elements from speed and position
        # Algorithm 4.1, Chapter 4 in [1]
        K = np.array([0, 0, 1])  # 3rd basis vector

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
        # e = np.linalg.norm(E,ord=2)

        # direct form:
        self.e = 1 / Constants.mu_earth * np.sqrt(
            (2 * Constants.mu_earth - r * v ** 2) * r * v_r ** 2 + (Constants.mu_earth - r * v ** 2) ** 2)

        # 11. Calculate arg. of perigee
        self.w = np.arccos(np.dot(N, E / (n * self.e)))
        if E[2] < 0:
            self.w = 2 * np.pi - self.w

        # 12. Calculate the true anomaly
        self.v = np.arccos(np.dot(E, cart.R.flat) / (self.e * r))
        if v_r < 0:
            self.v = 2 * np.pi - self.v

        # 13. Calculate semimajor axis
        rp = h ** 2 / Constants.mu_earth * 1 / (1 + self.e)
        ra = h ** 2 / Constants.mu_earth * 1 / (1 - self.e)
        self.a = 0.5 * (rp + ra)

        # 14. Calculate period (in [s])
        self.period = 2 * np.pi / np.sqrt(Constants.mu_earth) * (pow(self.a, 1.5))

    def from_qns_relative(self, qns, chaser):
        # calculate target absolute orbital elements
        # based on state and chaser absolute orbital elements

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
        # correct angles!
        if m > 2*np.pi:
            self._m = m - 2*np.pi
        elif m<0:
            self._m = m + 2*np.pi
        else:
            self._m = m

    def as_array_true(self):
        return np.array([[self.a, self.v, self.e, self.w, self.i, self.O]]).T

    def as_array_mean(self):
        return np.array([[self.a, self.m, self.e, self.w, self.i, self.O]]).T

    def from_osc_elems(self, osc):
        self.osc_elems_transformation(osc, True)


    def osc_elems_transformation(self, other, dir):
        # if dir == True =>  osc to mean
        # if dir == False => mean to osc
        # Reference:
        # Appendix G in [2]
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
        MwO = other.m + other.w + other.O + \
              gma_2_p / 8.0 * eta ** 3 * (1.0 - 11.0 * c_i ** 2 - 40.0 * (c_i ** 4 / (1.0 - 5.0 * c_i ** 2))) \
              - gma_2_p / 16.0 * (2.0 + other.e ** 2 - 11 * (2.0 + 3.0 * other.e ** 2) * c_i ** 2 \
                                  - 40.0 * (2.0 + 5.0 * other.e ** 2) * (
                                  c_i ** 4 / (1.0 - 5.0 * c_i ** 2)) - 400.0 * other.e * (
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
        edM = gma_2 / 8.0 * other.e * eta ** 3 * (1.0 - 11.0 * c_i ** 2 - 40.0 * (c_i ** 4 / (1 - 5 * c_i ** 2))) \
              - gma_2 / 4.0 * eta ** 3 * (2.0 * (3.0 * c_i ** 2 - 1.0) * ((a_r * eta) ** 2 + a_r + 1) * np.sin(other.v) \
                                          + 3.0 * (1.0 - c_i ** 2) * (
                                          (-(a_r * eta) ** 2 - a_r + 1) * np.sin(2.0 * other.w + other.v) \
                                          + (
                                          ((a_r * eta) ** 2 + a_r + 1.0 / 3.0) * np.sin(2.0 * other.w + 3.0 * other.v))))

        # formula G.308
        dO = -gma_2_p / 8.0 * other.e ** 2 * c_i * (
        11.0 + 80.0 * (c_i ** 2 / (1 - 5 * c_i ** 2)) + 200.0 * (c_i ** 4 / (1 - 5 * c_i ** 2) ** 2)) \
             - gma_2_p / 2.0 * c_i * (
        6.0 * (other.v - other.m + other.e * np.sin(other.v)) - 3.0 * np.sin(2.0 * other.w + 2.0 * other.v) \
        - 3.0 * other.e * np.sin(2.0 * other.w + other.v) - other.e * np.sin(2.0 * other.w + 3.0 * other.v))

        d_1 = (other.e + d_e) * np.sin(other.m) + edM * np.cos(other.m)  # G.309
        d_2 = (other.e + d_e) * np.cos(other.m) - edM * np.sin(other.m)  # G.310

        m = np.arctan2(d_1, d_2)  # G.311
        self.e = np.sqrt(d_1 ** 2 + d_2 ** 2)  # G.312

        d_3 = (np.sin(other.i / 2.0) + np.cos(other.i / 2.0) * d_i / 2.0) * np.sin(other.O) + np.sin(
            other.i / 2.0) * dO * np.cos(other.O)  # G.313
        d_4 = (np.sin(other.i / 2.0) + np.cos(other.i / 2.0) * d_i / 2.0) * np.cos(other.O) - np.sin(
            other.i / 2.0) * dO * np.sin(other.O)  # G.314

        self.O = np.arctan2(d_3, d_4)  # G.315
        self.i = 2 * np.arcsin(np.sqrt(d_3 ** 2 + d_4 ** 2))  # G.316
        self.w = MwO - m - self.O

        self.m = m  # assign m latest, as other properties might be used for ocnversion!


class OscKepOrbElem(KepOrbElem):

    def __init__(self):
        super(OscKepOrbElem, self).__init__()

    def from_mean_elems(self, mean):
        self.osc_elems_transformation(mean, False)

