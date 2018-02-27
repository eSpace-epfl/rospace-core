"""
Relative Navigation Filter based on UKF/filterpy.

    [1] New State Transition Matrices for Relative Motion of Spacecraft Formations in Perturbed Orbits,
        A. Koenig, T. Guffanti, S. D'Amico, AIAA 2016-5635
    [2] Improved Maneuver-free approach to angles-only navigation for space rendezvous,
        J. Sullivan, A.W. Koenig, S. D'Amico, AAS 16-530

    Prerequisities:
        filterpy
        numpy
        scipy
        orbdyn-tools (EPFL)

    Author: Michael Pantic
"""

import time
import numpy as np
import scipy
import rospy
from threading import Lock
from copy import copy

from tf import transformations
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints, MerweScaledSigmaPoints

import space_tf as stf
from RelativeOrbitalSTM import *
from space_msgs.msg import FilterState


class UKFRelativeOrbitalFilter:
    """ Relative navigation filter.

    Configuration values (for init)
        x:  Initial Filter state (size depends on options)
        P:  Initial Filter covariance (size of x ^2 )
        Q:  Process model noise (same size as P)
        R:  Measurement noise (2x2 for AON, 3x3 for ARN)

        mode:   Defines which mean-to-osculating transform is used
                Possible  values are:
                "ore" - DSST/Orekit function
                "schaub" - 1st order approximation
                "null"  - no transformation

        enable_emp: True/False if empirical accelerations are co-estimated
        enable_bias: True/False if biases are co-estimated
        output_debug: True/False if debug message should be output


    Filter state sizes:
        Relative Orbital Elements   6
        AON Biases                  +2 (if activated)
        Range Bias                  +1 (if activated and supplied)
        Empirical accelerations     +3 (if activated)
        Total:                      12

    """

    def __init__(self, x, P, Q, R, mode="ore", enable_emp=True, enable_bias=True, output_debug = False):

        self.n_total = len(x)  # total number of dimensions
        self.n_bias = len(np.diag(R))   # total number of biases to estimate
        self.n_sensor = len(np.diag(R)) # total number of sensor measurements

        if not enable_bias:
            self.n_bias = 0

        self.emp = enable_emp
        self.bias = enable_bias
        self.output_debug = output_debug


        # Set up filterpy UKF
        self.ukf = UnscentedKalmanFilter(
            self.n_total,
            self.n_sensor,
            120,
            lambda x : self.hx(x),
            lambda x, dt: self.fx(x,dt),
            MerweScaledSigmaPoints(self.n_total, 1e-3, 2, 0),
            scipy.linalg.cholesky,
            None,
            None,
            None,
            None)

        # set up debug publisher
        if self.output_debug:
            self.debug_pub = rospy.Publisher("state", FilterState, queue_size=10)

        # set initial filter states
        self.ukf.x = x
        self.ukf.P = P
        self.ukf.R = R
        self.ukf.Q = np.diag(np.concatenate([np.diag(Q), np.zeros(self.n_bias + 3)]))

        # set timestamp of initial filter state
        self.t_ukf = 0.0
        self.integration_step = 100.0

        self.mode = mode
        self.stm_matrix = RelativeOrbitalSTM()

        # time constant of empirical accelerations model
        self.tau_emp = 400
        self.t_oe_c = 0.0

        # variables to store noisy chaser state
        # and target states (target is only used for evaluation)
        self.O_c = stf.KepOrbElem()
        self.osc_O_c = stf.OscKepOrbElem()
        self.osc_O_t = stf.OscKepOrbElem()

        # variables to store exact chaser state
        self.O_c_exact = stf.KepOrbElem()
        self.osc_O_c_exact = stf.OscKepOrbElem()

        self.R_J2K_CB = None

        # constant term for J2
        self.J2_term = (3.0 / 4.0) * (stf.Constants.J_2 * (stf.Constants.R_earth ** 2) * np.sqrt(stf.Constants.mu_earth))
        self.update_lock = Lock()

        if self.output_debug:
            print "SETUP DONE"



    def fx(self, x, dt):
        """" State transition function / Process model
        Calculates the next state according to dt and current state.

        Simple brute-force integrator (as the function is very smooth)
        """

        t_step = self.integration_step

        while dt > 0.0:

            if dt < t_step:
                t_step = dt

            # process model for the ROE
            dx = self.get_dx_roe(x, t_step)
            x[0:6] = x[0:6] + dx[0:6]

            # influence of empirical accelerations
            b_oe = self.stm_matrix.get_B_oe(self.O_c.as_array_true())
            x[0:6] = x[0:6] + np.dot(b_oe * dt, x[6 + self.n_bias:6 + self.n_bias + 3])

            # no bias process model ;-)

            # emprical accelerations process model
            if self.emp:

                # decrease function
                psi = lambda tau: np.exp(-dt / tau)
                d_x_emp = np.array([1, 1, 1]) * psi(self.tau_emp)

                x[6 + self.n_bias:6 + self.n_bias + 3] = x[6+self.n_bias:6+self.n_bias+3] * d_x_emp

            dt -= t_step

        return x

    def get_dx_roe(self, x, delta_t):
        """ Implementation of Equation 10 of [2]
            Calculates derivative of current mean roe state x
        """

        # chaser variables
        a_s = self.O_c.a
        e_s = self.O_c.e
        w_s = self.O_c.w
        i_s = self.O_c.i

        c_ws = np.cos(w_s)
        s_ws = np.sin(w_s)
        c_is = np.cos(i_s)
        s_is = np.sin(i_s)

        # calculate target variables from current state
        a_t = a_s * (1 + x[0])
        e_t_c_wt = e_s * c_ws + x[2]
        e_t_s_wt = e_s * s_ws + x[3]
        i_t = i_s + x[4]

        e_t = np.sqrt(e_t_c_wt ** 2 + e_t_s_wt ** 2)
        c_it = np.cos(i_t)
        s_it = np.sin(i_t)

        Q_t = (5.0 * c_it ** 2 - 1.0)
        Q_s = (5.0 * c_is ** 2 - 1.0)

        # part1
        p1 = np.zeros([6])
        p1[1] = np.sqrt(1.0 - e_t ** 2) * (3.0 * c_it ** 2 - 1) + Q_t - 2.0 * c_it * c_is
        p1[2] = -e_t_s_wt * Q_t
        p1[3] = e_t_c_wt * Q_t
        p1[5] = -2.0 * c_it * s_is
        m_p1 = (self.J2_term * np.power(a_t, -3.5)) / ((1 - e_t ** 2) ** 2)

        # part2
        p2 = np.zeros([6])
        p2[1] = (1.0 + np.sqrt(1.0 - e_s ** 2)) * (3.0 * c_is ** 2 - 1.0)
        p2[2] = -e_s * s_ws * Q_s
        p2[3] = e_s * c_ws * Q_s
        p2[5] = -2.0 * c_is * s_is
        m_p2 = -(self.J2_term * np.power(a_s, -3.5)) / ((1 - e_s ** 2) ** 2)

        # part 3
        p3 = np.zeros([6])
        p3[1] = np.power(a_t, -1.5) - np.power(a_s, -1.5)
        m_p3 = np.sqrt(stf.Constants.mu_earth)

        d_x_oe = (m_p1 * p1 + m_p2 * p2 + m_p3 * p3) * delta_t

        return d_x_oe


    def hx(self, x):
        """" Measurement Function
        Converts state vector x into a measurement vector
        """

        # check if range measurements are active
        has_range = self.n_bias == 3

        # calculate estimated  target orbital elements based on current ROE state
        O_t_est = self.get_target_oe(x[0:6])

        # get osculating state vectors
        osc_O_t_est = stf.OscKepOrbElem()
        osc_O_t_est.from_mean_elems(O_t_est, self.mode)

        S_c = stf.Cartesian()
        S_c.from_keporb(self.osc_O_c)

        S_t = stf.Cartesian()
        S_t.from_keporb(osc_O_t_est)

        # calculate observation and add estimated negative bias
        if has_range:
            bias = x[6:9]
        else:
            bias = x[6:8]

        r_J2K = (S_t.R - S_c.R) * 1000
        r_CB = np.dot(self.R_J2K_CB[0:3, 0:3].T, r_J2K)

        # calculate angles
        alpha = np.arcsin(r_CB[1] / np.linalg.norm(r_CB)) + bias[0]
        eps = np.arctan2(r_CB[0], r_CB[2]) + bias[1]

        # return estimated measurement
        if has_range:
            r = np.linalg.norm(r_CB) + bias[2]
            return np.array([alpha, eps, r])
        else:
            return np.array([alpha, eps])

    def get_target_oe(self, x):
        """ Calculates target orbital element based
            on the relative orbital elements and the current
            chaser location
        """

        delta_c_t = stf.QNSRelOrbElements()
        delta_c_t.from_vector(copy(x[0:6]))

        O_t = stf.KepOrbElem()
        O_t.from_qns_relative(delta_c_t, self.O_c)

        return O_t

    def callback_aon(self,target_oe, chaser_oe, meas_msg):
        """ Performs the update step"""

        # fix asynchronous problems if a seperate predict step isused
        self.update_lock.acquire()

        osc_O_c = stf.OscKepOrbElem()
        osc_O_c.from_message(chaser_oe.position)

        osc_O_t = stf.OscKepOrbElem()
        osc_O_t.from_message(target_oe.position)

        # store exact values for evaluation
        self.osc_O_c_exact = osc_O_c
        self.O_c_exact = stf.KepOrbElem()
        self.O_c_exact.from_osc_elems(self.osc_O_c_exact, "ore") # always use ore for evaluation


        # add noise to mock state estimation error
        S_c_noisy = stf.Cartesian()
        S_c_noisy.from_keporb(osc_O_c)

        S_c_noisy.R += np.random.normal([0, 0, 0], [1/3.0, 1/3.0, 1/3.0])/1000.0
        S_c_noisy.V += np.random.normal([0,0,0],[1/300.0, 1/300.0, 1/300.0])/1000.0

        self.osc_O_c.from_cartesian(S_c_noisy)
        self.osc_O_t = osc_O_t

        self.O_c = stf.KepOrbElem()
        self.O_c.from_osc_elems(osc_O_c, self.mode)

        # store orientation
        self.R_J2K_CB = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                           chaser_oe.orientation.y,
                                                           chaser_oe.orientation.z,
                                                           chaser_oe.orientation.w])

        # get current time
        t_msg = float(meas_msg.header.stamp.secs) + float(meas_msg.header.stamp.nsecs)*1e-9


        # perform predict step up to current message time
        dt = t_msg - self.t_ukf

        if dt < 0.0:
            rospy.logerr("DT in Callback smaller than 0 : %f", dt)
            return

        self.ukf.predict(dt)

        # publish filter state hat(x) after predict
        if self.output_debug:
            self.publish_filter_state()

        self.t_ukf = t_msg

        rospy.loginfo("Executed predict with dt %f", dt)

        # perform update step if message is valid
        if meas_msg.valid:

            z = np.zeros(self.n_sensor)

            if self.n_sensor == 3:
                    z[2] = meas_msg.value.range

            z[0] = meas_msg.value.azimut
            z[1] = meas_msg.value.elevation

            # get pre-fit residual
            residual_pre = np.array(self.hx(self.ukf.x)) - z

            # perform UKF update step
            self.ukf.update(z)

            # get post-fit residual
            residual_post = np.array(self.hx(self.ukf.x)) - z

            # output post update filter state debug message
            if self.output_debug:
                self.publish_filter_state(z, residual_pre, residual_post)
        self.update_lock.release()



    def publish_filter_state(self, meas=None, pre_residual=None, post_residual=None):
        """ Publishes internal filter state (almost all data)"""

        fs = FilterState()
        fs.header.stamp = rospy.Time.from_seconds(self.t_ukf)


        # set state
        fs.x = self.ukf.x
        fs.chaser_mean_a = self.O_c.a * 1000.0
        fs.roe_scaled = self.ukf.x[0:6]*self.O_c.a * 1000
        fs.std_roe = np.sqrt(np.diag(self.ukf.P[0:6,0:6]))*(self.O_c.a * 1000.0)
        fs.P = self.ukf.P.flatten("C")

        #if there is a measurement, set it
        if meas is not None:
            fs.measurement = meas
            fs.pre_residual = pre_residual
            fs.post_residual = post_residual



        # calculate differences (est is for estimated)
        O_t_est = self.get_target_oe(self.ukf.x)

        # osculating orbital elements of target, estimated
        osc_O_t_est = stf.OscKepOrbElem()
        osc_O_t_est.from_mean_elems(O_t_est, self.mode)

        # state vector of target, estimated
        S_t_est = stf.Cartesian()
        S_t_est.from_keporb(osc_O_t_est)

        # state vector of target, true
        S_t_true = stf.Cartesian()
        S_t_true.from_keporb(self.osc_O_t)

        # state vector of chaser, true
        S_c_true = stf.Cartesian()
        S_c_true.from_keporb(self.osc_O_c_exact)

                # mean orbital elements of target, true
        O_t_true = stf.KepOrbElem()
        O_t_true.from_osc_elems(self.osc_O_t, "ore") # always use ore for evaluation, best accuracy

        # calculate true current ROE, scaled
        true_roe = stf.QNSRelOrbElements()
        true_roe.from_keporb(O_t_true, self.O_c_exact)
        fs.roe_scaled_true = true_roe.as_scaled(self.O_c_exact.a).as_vector()

        # set target/target estimated and chaser positions
        fs.chaser_true_R = S_c_true.R
        fs.chaser_true_V = S_c_true.V
        fs.target_true_R = S_t_true.R
        fs.target_true_V = S_t_true.V
        fs.target_est_R = S_t_est.R
        fs.target_est_V = S_t_est.V

        # calculate LVLH difference as seen from CHASER with LVLH origin (THIS IS FLIPPED ON PURPOSE)
        diff_R_lvlh = stf.CartesianLVLH()
        diff_R_lvlh.from_cartesian_pair(S_t_est, S_c_true)

        target_lvlh_true = stf.CartesianLVLH()
        target_lvlh_true.from_cartesian_pair(S_t_true, S_c_true)

        fs.diff_R_lvlh_est = diff_R_lvlh.R
        fs.diff_V_lvlh_est = diff_R_lvlh.V

        fs.diff_R_lvlh_true = target_lvlh_true.R
        fs.diff_V_lvlh_true = target_lvlh_true.V

        fs.diff_lvlh = np.linalg.norm(S_t_true.R-S_t_est.R)

        self.debug_pub.publish(fs)


    def get_state(self):
        """ Getter to get current target estimate"""

        self.update_lock.acquire()
        try:
            return [self.t_ukf, self.ukf.x[0:6] * self.O_c.a * 1000, self.ukf.P[0:6, 0:6] * (self.O_c.a * 1000.0) ** 2]
            #return [self.ukf.x , self.ukf.P * (self.oe_c.a * 1000.0) ** 2]
        finally:
            self.update_lock.release()


