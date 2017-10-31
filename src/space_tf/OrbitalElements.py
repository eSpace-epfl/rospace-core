#   Class to hold keplerian orbital elements and convert to/from cartesian
#   Author: Michael Pantic, michael.pantic@gmail.com
#   License: TBD
#     Sources/Literature: [1] Orbital Mechanics for Engineering Students by Howard Curtis

import numpy as np

from Constants import *


class OrbitalElements:
    def __init__(self):
        self.i = 0 # inclination
        self.e = 0 # eccentricity
        self.w = 0 # arg of perigee
        self.omega  = 0 # raan
        self.t = 0 # true anomaly
        self.m = 0 # mean anomaly
        self.a = 0 # semimjaor axis
        self.period = 0 # period
        self.mean_motion = 0 # mean motion
        self.E = 0 # mean eccentricity

    def update_m(self):
        # Calculates m based on series expansion
        # Prerequisites: t
        # Source: Todo: Celestial Mechanics by W.M. Smart
        self.m = self.t
        self.m = self.m - 2*self.e*np.sin(self.t)
        self.m = self.m + (0.75*self.e**2 + 0.125*self.e**4)*np.sin(2*self.t)
        self.m = self.m - 1.0/3.0*self.e**3*np.sin(3*self.t)
        self.m = self.m + 5.0/32.0*self.e**4*np.sin(4*self.t)

    def update_E(self):
        # Calculates mean eccentricity
        # Prerequisites: m and e
        # Source: Algorithm 3.1 from [1]
        # Uses Newton-Raphson iteration
        if self.m < np.pi:
            self.E = self.m+self.e/2.0
        else:
            self.E = self.m-self.e/2.0

        max_int = 10
        cur_int = 0

        while True:
            fE = self.E - self.e * np.sin(self.E) - self.m
            fpE = 1.0 - self.e * np.cos(self.E)
            ratio = fE/fpE
            if abs(ratio) > 1e-15 and cur_int < max_int:
                self.E = self.E - ratio
                cur_int = cur_int+1
            else:
                break

    def update_t(self):
        # Calculates true anomaly
        # Prerequisities: e and E
        self.t = 2*np.arctan2(np.sqrt(1+self.e)*np.sin(self.E/2.0),
                              np.sqrt(1-self.e)*np.cos(self.E/2.0))

    def update_a(self):
        # Calculates semimajor axis based on period
        self.a = np.power(self.period**2*Constants.mu_earth / (4*np.pi**2),1/3.0)

    def fromTLE(self, i_tle, omega_tle, e_tle, m_tle, w_tle, mean_motion_tle):
        #assign known data
        self.i = i_tle
        self.e = e_tle
        self.m = m_tle
        self.w = w_tle
        self.omega = omega_tle
        self.mean_motion = mean_motion_tle
        self.period = 1.0/mean_motion_tle*60*60*24


        #update unknown data in proper order
        self.update_E()
        self.update_t()
        self.update_a()

    def asArray(self):
        return np.array([self.e, self.i, self.omega, self.w, self.a,  self.t])


    def toCartesian(self, cart):
        #Calculates cartesian coordinates based on current orbital elements
        p = self.a*(1-self.e**2)

        # Position in perifocal frame, then rotate to proper orbital plane
        R_per = np.array([p*np.cos(self.t), p*np.sin(self.t), 0]) / (1.0+self.e*np.cos(self.t))
        cart.R = R_z(self.omega).dot(R_x(self.i)).dot(R_z(self.w)).dot(R_per)

        # speed in perifocal frame, then rotate
        V_per = np.array([-np.sin(self.t), self.e+np.cos(self.t), 0]) * np.sqrt(Constants.mu_earth/p)
        cart.V = R_z(self.omega).dot(R_x(self.i)).dot(R_z(self.w)).dot(V_per)


    def fromCartesian(self, cart):
        # Calculates and initalizes keplerian elements from speed and position
        # Algorithm 4.1, Chapter 4 in [1]
        K = np.array([0,0,1]) # 3rd basis vector

        # 1. Calc distance
        r = np.linalg.norm(cart.R,ord=2)

        # 2. Calc  speed
        v = np.linalg.norm(cart.V,ord=2)

        # 3. Calc radial velocity
        v_r = np.dot(cart.R.flat,cart.V.flat)/r

        # 4. Calc specific angular momentum
        H = np.cross(cart.R.flat, cart.V.flat)

        # 5. Calc magnitude of specific angular momentum
        h = np.linalg.norm(H,ord=2)

        # 6. Calc inclination
        self.i = np.arccos(H[2]/h)

        # 7. Calculate Node line
        N = np.cross(K, H)

        # 8. Calculate magnitude of N
        n = np.linalg.norm(N, ord=2)

        # 9. calculate RAAN
        self.omega = np.arccos(N[0]/n)
        if N[1] < 0:
            self.omega = 2*np.pi-self.omega

        # 10. calculate eccentricity vector  / 11. Calc eccentricity
        E = 1/Constants.mu_earth*((v**2-Constants.mu_earth/r)*cart.R.flat-r*v_r*cart.V.flat)
        # e = np.linalg.norm(E,ord=2)

        # direct form:
        self.e = 1/Constants.mu_earth*np.sqrt( (2*Constants.mu_earth-r*v**2)*r*v_r**2+(Constants.mu_earth-r*v**2)**2)

        # 11. Calculate arg. of perigee
        self.w = np.arccos(np.dot(N,E/(n*self.e)))
        if E[2] < 0:
            self.w = 2*np.pi-self.w

        # 12. Calculate the true anomaly
        self.t = np.arccos(np.dot(E,cart.R.flat)/(self.e*r))
        if v_r < 0:
            self.t = 2*np.pi-self.t

        # 13. Calculate semimajor axis
        rp = h**2/Constants.mu_earth * 1/(1+self.e)
        ra = h**2/Constants.mu_earth * 1/(1-self.e)
        self.a = 0.5 * (rp+ra)

        # 14. Calculate period (in [s])
        self.period = 2*np.pi/np.sqrt(Constants.mu_earth)*(pow(self.a,1.5))
