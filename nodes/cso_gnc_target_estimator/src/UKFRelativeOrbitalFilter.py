import numpy as np
import rospy

from tf import transformations
import space_tf as stf
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from RelativeOrbitalSTM import *
import time
from threading import Lock
from copy import copy
from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints, MerweScaledSigmaPoints
import scipy
from space_msgs.msg import FilterState
class UKFRelativeOrbitalFilter:
    # used references:
    # [1] the other one... TODO

    # [2] Improved maneuver-free approach to angles only navigation
    #       for space rendez-vous" by Sullivan/Koenig/D'Amico

    def __init__(self, x, P, Q, R, mode="ore", enable_emp=True, enable_bias=True, augment_range = False):

        self.n_total = len(x)
        self.n_bias = len(np.diag(R))
        self.n_sensor = len(np.diag(R))

        if not enable_bias:
            self.n_bias= 0

        self.emp = enable_emp
        self.bias = enable_bias
        self.augment_range = augment_range

        self.ukf = UnscentedKalmanFilter(
            self.n_total,
            self.n_sensor,
            120,
            lambda x : self.hx_aonr(x),
            lambda x, dt: self.fx(x,dt),
            MerweScaledSigmaPoints(self.n_total,1e-3,2,0),
            scipy.linalg.cholesky,
            None,
            None, #lambda sigma, weights: self.obs_mean(sigma, weights),
            None,
            None) #lambda a, b: self.obs_residual(a, b))




        self.debug_pub = rospy.Publisher("state", FilterState, queue_size=10)
        self.ukf.x = x
        self.ukf.P = P


        self.mode = mode

        self.stm_matrix = RelativeOrbitalSTM()
        self.ukf.Q = np.diag(np.concatenate([np.diag(Q), np.zeros(self.n_bias + 3)]))
        print np.diag(self.ukf.Q[0:6,0:6])

        self.ukf.R = R
        self.t_ukf = 0.0
        self.t_oe_c = 0.0

        self.tau_b = 9999999999999
        self.tau_emp = 400

        self.oe_c = stf.KepOrbElem()
        self.oe_c_osc = stf.OscKepOrbElem()

        self.oe_c_osc_exact = stf.KepOrbElem()

        self.oe_t_osc = stf.OscKepOrbElem()
        self.R_body = None

        # constant term
        self.J2_term = (3.0 / 4.0) * (stf.Constants.J_2 * (stf.Constants.R_earth ** 2) * np.sqrt(stf.Constants.mu_earth))
        self.update_lock = Lock()
        print "SETUP DONE"

    def hx_aonr(self, x):

        has_range = self.n_bias == 3

        """" Measurement Function
        Converts state vector x into a measurement vector
        """
        oe_t = self.get_target_oe(x[0:6])

        # get osculating state vectors
        target_osc_oe = stf.OscKepOrbElem()
        target_osc_oe.from_mean_elems(oe_t, self.mode)



        cart_c = stf.Cartesian()
        cart_c.from_keporb(self.oe_c_osc)

        cart_t = stf.Cartesian()
        cart_t.from_keporb(target_osc_oe)

        # calculate observation
        if has_range:
            bias = x[6:9]
        else:
            bias = x[6:8]

        p_teme = (cart_t.R - cart_c.R) * 1000
        p_body = np.dot(self.R_body[0:3, 0:3].T, p_teme)

        # calculate angles
        alpha = np.arcsin(p_body[1] / np.linalg.norm(p_body)) + bias[0]
        #alpha = np.arctan2(p_body[2], p_body[1]) + bias[9]
        eps = np.arctan2(p_body[0], p_body[2]) + bias[1]

        if has_range:
            r = np.linalg.norm(p_body) + bias[2]
            return np.array([alpha, eps, r])
        else:
            return np.array([alpha, eps])



    def fx(self, x, dt):
        """" State transition function
        Calculates the next state according to dt and current state
        """

        t_step = 100.0
        while dt > 0.0:

            if dt <t_step:
                t_step = dt

            dx = self.get_Dx(x, t_step, self.tau_b, self.tau_emp)
            b_oe = self.stm_matrix.get_B_oe(self.oe_c.as_array_true())

            x[0:6] = x[0:6] + dx[0:6]

            if self.bias:
                x[6:6+self.n_bias] = x[6:6+self.n_bias] #* dx[6:6+self.n_bias]

            if self.emp:
                x[0:6] = x[0:6] + np.dot(b_oe*dt, x[6+self.n_bias:6+self.n_bias+3])
                x[6 + self.n_bias:6 + self.n_bias + 3] =x[6+self.n_bias:6+self.n_bias+3] * dx[6+self.n_bias:6+self.n_bias+3]
            dt -= t_step

        return x


    def get_target_oe(self, x):
        rel_elem = stf.QNSRelOrbElements()
        rel_elem.from_vector(copy(x[0:6]))

        oe_t = stf.KepOrbElem()
        oe_t.from_qns_relative(rel_elem, self.oe_c)

        return oe_t


    def get_Dx(self,x,delta_t, tau_b, tau_emp):
        # Implementation of Equation 10 of [2]
        # variables
        # chaser variables
        a_s = self.oe_c.a
        e_s = self.oe_c.e
        w_s = self.oe_c.w
        i_s = self.oe_c.i

        c_ws = np.cos(w_s)
        s_ws = np.sin(w_s)
        c_is = np.cos(i_s)
        s_is = np.sin(i_s)

        # target variables
        a_t = a_s*(1+x[0])
        e_t_c_wt = e_s*c_ws + x[2]
        e_t_s_wt = e_s*s_ws + x[3]
        i_t = i_s + x[4]

        e_t = np.sqrt(e_t_c_wt**2+e_t_s_wt**2)
        c_it = np.cos(i_t)
        s_it = np.sin(i_t)

        Q_t = (5.0* c_it**2 - 1.0)
        Q_s = (5.0 * c_is ** 2 - 1.0)

        # part1
        p1 = np.zeros([6])
        p1[1] = np.sqrt(1.0-e_t**2) * (3.0*c_it**2 - 1) + Q_t - 2.0*c_it*c_is
        p1[2] = -e_t_s_wt * Q_t
        p1[3] = e_t_c_wt * Q_t
        p1[5] = -2.0 * c_it * s_is
        m_p1 = (self.J2_term * np.power(a_t, -3.5))/((1-e_t**2)**2)

        # part2
        p2 = np.zeros([6])
        p2[1] = (1.0 + np.sqrt(1.0-e_s**2))*(3.0*c_is**2-1.0)
        p2[2] = -e_s * s_ws * Q_s
        p2[3] =  e_s * c_ws * Q_s
        p2[5] = -2.0 * c_is * s_is
        m_p2 = -(self.J2_term * np.power(a_s, -3.5))/((1-e_s**2)**2)

        # part 3
        p3 = np.zeros([6])
        p3[1] = np.power(a_t, -1.5) - np.power(a_s, -1.5)
        m_p3 = np.sqrt(stf.Constants.mu_earth)

        d_x_oe = (m_p1 * p1 + m_p2 * p2 + m_p3 * p3)*delta_t

        psi = lambda tau: np.exp(-delta_t/tau)

        d_x_B = np.ones(self.n_bias)* psi(tau_b)
        d_x_emp = np.array([1, 1, 1]) * psi(tau_emp)

        d_x = np.zeros(self.n_total)
        d_x[0:6] = d_x_oe

        if self.bias:
            d_x[6:6+self.n_bias] = d_x_B

        if self.emp:
            d_x[6+self.n_bias:6+self.n_bias+3] = d_x_emp

        return d_x

    # Performs measurement update
    def callback_aon(self,target_oe, chaser_oe, meas_msg):

        self.update_lock.acquire()


        oe_c_osc = stf.OscKepOrbElem()
        oe_c_osc.from_message(chaser_oe.position)

        oe_t_osc = stf.OscKepOrbElem()
        oe_t_osc.from_message(target_oe.position)

        self.oe_c_osc_exact = oe_c_osc

        # add noise
        cart_chaser_noise = stf.Cartesian()
        cart_chaser_noise.from_keporb(oe_c_osc)

        cart_chaser_noise.R += np.random.normal([0, 0, 0], [1/3.0, 1/3.0, 1/3.0])/1000.0
        cart_chaser_noise.V += np.random.normal([0,0,0],[1/300.0, 1/300.0, 1/300.0])/1000.0

        self.oe_c_osc.from_cartesian(cart_chaser_noise)
        self.oe_t_osc = oe_t_osc

        self.oe_c = stf.KepOrbElem()
        self.oe_c.from_osc_elems(oe_c_osc, self.mode)

        self.oe_c_exact = stf.KepOrbElem()
        self.oe_c_exact.from_osc_elems(self.oe_c_osc_exact, "ore")

        # store orientation
        self.R_body = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                         chaser_oe.orientation.y,
                                                         chaser_oe.orientation.z,
                                                         chaser_oe.orientation.w])

        t_msg = float(meas_msg.header.stamp.secs) + float(meas_msg.header.stamp.nsecs)*1e-9


        # advance state to time of current measurement
        dt = t_msg -self.t_ukf

        if dt < 0.0:
            rospy.logerr("DT in Callback smaller than 0 : %f", dt)
            return

        self.ukf.predict(dt)
        self.publish_filter_state()
        rospy.loginfo("Executed predict with dt %f", dt)
        self.t_ukf = t_msg


        if meas_msg.valid:
            # augment range
            target_est_oe = self.get_target_oe(self.ukf.x)
            target_est_osc = stf.OscKepOrbElem()
            target_est_osc.from_mean_elems(target_est_oe, self.mode)
            target_est_cart = stf.Cartesian()
            target_est_cart.from_keporb(target_est_osc)


            chaser_true_cart = stf.Cartesian()
            chaser_true_cart.from_keporb(self.oe_c_osc)

            z = np.zeros(self.n_sensor)

            if self.n_sensor == 3:
                    z[2] = meas_msg.value.range

            z[0] = meas_msg.value.azimut
            z[1] = meas_msg.value.elevation

            residual_pre = np.array(self.hx_aonr(self.ukf.x))-z

            self.ukf.update(z)

            residual_post = np.array(self.hx_aonr(self.ukf.x))-z

            self.publish_filter_state(z, residual_pre, residual_post)
        self.update_lock.release()



    def publish_filter_state(self, meas=None, pre_residual=None, post_residual=None):
        fs = FilterState()
        fs.header.stamp = rospy.Time.from_seconds(self.t_ukf)


        # set state
        fs.x = self.ukf.x
        fs.chaser_mean_a = self.oe_c.a*1000.0
        fs.roe_scaled = self.ukf.x[0:6]*self.oe_c.a*1000
        fs.std_roe = np.sqrt(np.diag(self.ukf.P[0:6,0:6]))*(self.oe_c.a*1000.0)
        fs.P = self.ukf.P.flatten("C")

        #if there is a measurement, set it
        if meas is not None:
            fs.measurement = meas
            fs.pre_residual = pre_residual
            fs.post_residual = post_residual



        # calculate differences
        target_est_oe = self.get_target_oe(self.ukf.x)
        target_est_osc = stf.OscKepOrbElem()
        target_est_osc.from_mean_elems(target_est_oe, self.mode)
        target_est_cart = stf.Cartesian()
        target_est_cart.from_keporb(target_est_osc)

        target_true_cart = stf.Cartesian()
        target_true_cart.from_keporb(self.oe_t_osc)

        chaser_true_cart = stf.Cartesian()
        chaser_true_cart.from_keporb(self.oe_c_osc_exact)

        # calculate true current ROE, scaled
        true_roe = stf.QNSRelOrbElements()
        target_true_mean = stf.KepOrbElem()
        target_true_mean.from_osc_elems(self.oe_t_osc, "ore") # always use ore for evaluation, best accuracy

        true_roe.from_keporb(target_true_mean, self.oe_c_exact)

        fs.roe_scaled_true = true_roe.as_scaled(self.oe_c_exact.a).as_vector()

        # set target/target estimated and chaser positions
        fs.chaser_true_R = chaser_true_cart.R
        fs.chaser_true_V = chaser_true_cart.V
        fs.target_true_R = target_true_cart.R
        fs.target_true_V = target_true_cart.V
        fs.target_est_R = target_est_cart.R
        fs.target_est_V = target_est_cart.V

        # calculate LVLH position with CHASER as LVLH origin (THIS IS FLIPPED ON PURPOSE)
        target_lvlh_est = stf.CartesianLVLH()
        target_lvlh_est.from_cartesian_pair( target_est_cart, chaser_true_cart)

        target_lvlh_true = stf.CartesianLVLH()
        target_lvlh_true.from_cartesian_pair( target_true_cart, chaser_true_cart)

        fs.diff_R_lvlh_est = target_lvlh_est.R
        fs.diff_V_lvlh_est = target_lvlh_est.V

        fs.diff_R_lvlh_true = target_lvlh_true.R
        fs.diff_V_lvlh_true = target_lvlh_true.V

        fs.diff_lvlh = np.linalg.norm(target_true_cart.R-target_est_cart.R)

        #print self.mode+" "+str(np.linalg.norm(target_true_cart.R-target_est_cart.R))

        self.debug_pub.publish(fs)

    def get_state(self):
        self.update_lock.acquire()
        try:
            return [self.t_ukf, self.ukf.x[0:6]*self.oe_c.a*1000, self.ukf.P[0:6,0:6]*(self.oe_c.a*1000.0)**2]
            #return [self.ukf.x , self.ukf.P * (self.oe_c.a * 1000.0) ** 2]
        finally:
            self.update_lock.release()


