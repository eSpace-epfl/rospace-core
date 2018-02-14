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
class UKFRelativeOrbitalFilter:
    # used references:
    # [1] the other one... TODO

    # [2] Improved maneuver-free approach to angles only navigation
    #       for space rendez-vous" by Sullivan/Koenig/D'Amico

    def __init__(self, x, P, Q, R, mode):

        self.ukf = UnscentedKalmanFilter(
            12,
            3,
            120,
            lambda x: self.hx(x),
            lambda x, dt: self.fx(x,dt),
            MerweScaledSigmaPoints(12,1e-3,2,0),
            scipy.linalg.cholesky,
            None,
            None, #lambda sigma, weights: self.obs_mean(sigma, weights),
            None,
            None) #lambda a, b: self.obs_residual(a, b))


        

        self.ukf.x = np.concatenate([x, [0,0,0], [0,0,0]])

        self.ukf.P = np.zeros((12,12))
        self.ukf.P[0:6,0:6] = P
        self.ukf.P[6:9,6:9] = np.diag([1e-6, 1e-6, 1e-1])
        self.ukf.P[9:12, 9:12] = np.diag([1e-14, 1e-14, 1e-14])
        self.mode = mode
        self.stm_matrix = RelativeOrbitalSTM()
        self.ukf.Q = np.diag(np.diag(self.ukf.P)) /50000.0

        self.ukf.R = R
        self.t_ukf = 0.0
        self.t_oe_c = 0.0

        self.tau_b = 27000
        self.tau_emp = 400

        self.oe_c = stf.KepOrbElem()
        self.oe_c_osc = stf.OscKepOrbElem()
        self.oe_t_osc = stf.OscKepOrbElem()
        self.R_body = None

        # constant term
        self.J2_term = (3.0 / 4.0) * (stf.Constants.J_2 * (stf.Constants.R_earth ** 2) * np.sqrt(stf.Constants.mu_earth))
        self.update_lock = Lock()

    def obs_mean(self,sigma,weights):

        sum_sin = np.zeros(2)

        sum_cos = np.zeros(2)
        mean = np.zeros(2)

        for i in range(0, len(sigma)):
            s = sigma[i]
            sum_sin[0] += np.sin(s[0])*weights[i]
            sum_sin[1] += np.sin(s[1])*weights[i]
            sum_cos[0] += np.cos(s[0])*weights[i]
            sum_cos[1] += np.cos(s[1])*weights[i]
            print i, weights[i]

        mean[0] = np.arctan2(sum_sin[0], sum_cos[0])
        mean[1] = np.arctan2(sum_sin[1], sum_cos[1])
        return mean

    def obs_residual(self, a, b):
        c = np.zeros(a.shape)

        for i in range(0, len(c)):
            c[i] = a[i] - b[i]
            if c[i] > np.pi:
                c[i] -= 2 * np.pi
            if c[i] < -np.pi:
                c[i] = 2 * np.pi
        return c

    def calc_angle(self, target_osc_oe, chaser_osc_oe, bias):

        cart_c = stf.Cartesian()
        cart_c.from_keporb(chaser_osc_oe)

        cart_t = stf.Cartesian()
        cart_t.from_keporb(target_osc_oe)

        # calculate observation
        p_teme = (cart_t.R - cart_c.R) * 1000
        p_body = np.dot(self.R_body[0:3, 0:3].T, p_teme)
        r = np.linalg.norm(p_body)+ bias[2]

        # calculate angles
        alpha = np.arcsin(p_body[1] / np.linalg.norm(p_body))+ bias[0]
        eps = np.arctan(p_body[0]/p_body[2])+ bias[1]

        return np.array([alpha, eps, r])

    def hx_aon(self, x):
        """" Measurement Function
        Converts state vector x into a measurement vector
        """
        oe_t = self.get_target_oe(x[0:6])

        # get osculating state vectors
        target_osc_oe = stf.OscKepOrbElem()
        target_osc_oe.from_mean_elems(oe_t, self.mode)

        bias = x[6:8]

        cart_c = stf.Cartesian()
        cart_c.from_keporb(self.oe_c_osc)

        cart_t = stf.Cartesian()
        cart_t.from_keporb(target_osc_oe)

        # calculate observation
        p_teme = (cart_t.R - cart_c.R) * 1000
        p_body = np.dot(self.R_body[0:3, 0:3].T, p_teme)

        # calculate angles
        alpha = np.arcsin(p_body[1] / np.linalg.norm(p_body)) + bias[0]
        eps = np.arctan(p_body[0] / p_body[2]) + bias[1]

        return np.array([alpha, eps])

    def hx_aonr(self, x):
        """" Measurement Function
        Converts state vector x into a measurement vector
        """
        oe_t = self.get_target_oe(x[0:6])

        # get osculating state vectors
        target_osc_oe = stf.OscKepOrbElem()
        target_osc_oe.from_mean_elems(oe_t, self.mode)

        bias = x[6:9]

        cart_c = stf.Cartesian()
        cart_c.from_keporb(self.oe_c_osc)

        cart_t = stf.Cartesian()
        cart_t.from_keporb(target_osc_oe)

        # calculate observation
        p_teme = (cart_t.R - cart_c.R) * 1000
        p_body = np.dot(self.R_body[0:3, 0:3].T, p_teme)
        r = np.linalg.norm(p_body) + bias[2]

        # calculate angles
        alpha = np.arcsin(p_body[1] / np.linalg.norm(p_body)) + bias[0]
        eps = np.arctan(p_body[0] / p_body[2]) + bias[1]

        return np.array([alpha, eps, r])



    def fx(self, x, dt):
        """" State transition function
        Calculates the next state according to dt and current state
        """
        x_old = x.copy()
        t_step = 10.0
        while dt > 10.0:
            dx = self.get_Dx(x, t_step, self.tau_b, self.tau_emp)
            b_oe = self.stm_matrix.get_B_oe(self.oe_c.as_array_true())

            x[0:6] = x[0:6] + dx[0:6]+ np.dot(b_oe*dt, x[9:12])
            x[6:9] = x[6:9] * dx[6:9]
            x[9:12] = x[9:12] * dx[9:12]
            dt -= t_step


        dx = self.get_Dx(x, dt, self.tau_b, self.tau_emp)
        b_oe = self.stm_matrix.get_B_oe(self.oe_c.as_array_true())

        x[0:6] = x[0:6] + dx[0:6] + np.dot(b_oe*dt, x[9:12])
        x[6:9] = x[6:9] * dx[6:9]
        x[9:12] = x[9:12] * dx[9:12]

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

        d_x_B = np.array([1,1,1]) * psi(tau_b)
        d_x_emp = np.array([1, 1, 1]) * psi(tau_emp)

        d_x = np.zeros(12)
        d_x[0:6] = d_x_oe
        d_x[6:9] = d_x_B
        d_x[9:12] = d_x_emp
        return d_x


    # Performs prediction step
    def callback_state(self, target_oe, chaser_oe):
        self.update_lock.acquire()
        # get current time
        t_msg = float(chaser_oe.header.stamp.secs) + float(chaser_oe.header.stamp.nsecs)*1e-9

        oe_c_osc = stf.OscKepOrbElem()
        oe_c_osc.from_message(chaser_oe.position)

        oe_t_osc = stf.OscKepOrbElem()
        oe_t_osc.from_message(target_oe.position)

        self.oe_c_osc = oe_c_osc
        self.oe_t_osc = oe_t_osc

        self.oe_c = stf.KepOrbElem()
        self.oe_c.from_osc_elems(oe_c_osc, self.mode)



        dt = t_msg - self.t_ukf
        # if self.t_ukf == 0.0 and dt > 0.0:
          #  self.ukf.predict(dt)
          #  rospy.loginfo("Executed predict with dt %f (state0)", dt)
          #  self.t_ukf = t_msg
        # elif dt > 15: # update every 10s
           # self.ukf.predict(dt)
           # rospy.loginfo("Executed predict with dt %f (state)", dt)
           # self.t_ukf = t_msg
         #pass




        # store orientation
        self.R_body = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                    chaser_oe.orientation.y,
                                                    chaser_oe.orientation.z,
                                                    chaser_oe.orientation.w])

        self.t_oe_c = t_msg
        # update target location
        self.update_lock.release()


    # Performs measurement update
    def callback_aon(self,target_oe, chaser_oe, meas_msg):
        self.update_lock.acquire()


        oe_c_osc = stf.OscKepOrbElem()
        oe_c_osc.from_message(chaser_oe.position)

        oe_t_osc = stf.OscKepOrbElem()
        oe_t_osc.from_message(target_oe.position)

        self.oe_c_osc = oe_c_osc
        self.oe_t_osc = oe_t_osc

        self.oe_c = stf.KepOrbElem()
        self.oe_c.from_osc_elems(oe_c_osc, self.mode)

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
        rospy.loginfo("Executed predict with dt %f", dt)
        self.t_ukf = t_msg


        # perform measurement update
        z = np.zeros(3)
        z[0] = meas_msg.value.azimut
        z[1] = meas_msg.value.elevation
        z[2] = meas_msg.range

        self.ukf.update(z)

        hx = self.hx(self.ukf.x)

        diff = hx-z
        rospy.loginfo("Executed update with z_diff " + str(diff))

        print self.ukf.x[6:9]
        print self.ukf.x[9:12]

        target_est_oe =self.get_target_oe(self.ukf.x+0.01)
        target_est_osc = stf.OscKepOrbElem()
        target_est_osc.from_mean_elems(target_est_oe, self.mode)
        target_est_cart = stf.Cartesian()
        target_est_cart.from_keporb(target_est_osc)

        target_true_cart = stf.Cartesian()
        target_true_cart.from_keporb(self.oe_t_osc)

        chaser_true_cart = stf.Cartesian()
        chaser_true_cart.from_keporb(self.oe_c_osc)

        print np.linalg.norm(chaser_true_cart.R - target_true_cart.R)
        print np.linalg.norm(chaser_true_cart.R - target_est_cart.R)
        print np.linalg.norm(target_true_cart.R - target_est_cart.R)


        self.update_lock.release()

    def get_state(self):
        self.update_lock.acquire()
        try:
            return [self.t_ukf, self.ukf.x[0:6]*self.oe_c.a*1000, self.ukf.P[0:6,0:6]*(self.oe_c.a*1000.0)**2]
            #return [self.ukf.x , self.ukf.P * (self.oe_c.a * 1000.0) ** 2]
        finally:
            self.update_lock.release()


