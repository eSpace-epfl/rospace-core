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
class BaseRelativeOrbitalFilter:
    # used references:
    # [1] the other one... TODO

    # [2] Improved maneuver-free approach to angles only navigation
    #       for space rendez-vous" by Sullivan/Koenig/D'Amico

    def __init__(self):
        #00138646026702
        #self.x = np.array([[0.002, 0.007, 7.0e-7, 1.90e-5, 3.225e-6, -0.0052202]]).T
        #self.x = np.array([[-1.04166667e-05, -1.38888889e-03, 2.08333333e-05, 0, -2.08333333e-05, 0.0]]).T

        self.x = np.array([[-5000, -20000, 0, 0, 0, 0.0]]).T
        self.x = self.x / (7050.0*1000)
        self.P = np.diag([0.01,0.01,0.01,0.01,0.01,0.01])
        self.Q = np.diag([0.01, 0.01, 0.001, 0.001, 0.001, 0.001])
        self.R = np.array([[0.0001, 0],[0, 0.0001]])

        self.t = rospy.Time(0, 0)

        self.time_c = rospy.Time(0, 0)
        self.oe_c = stf.KepOrbElem()
        self.oe_t = stf.KepOrbElem()
        self.has_oe = False

        self.stm = RelativeOrbitalSTM()
        self.R_body = None

        # constant term
        self.J2_term = (3.0 / 4.0) * (stf.Constants.J_2 * (stf.Constants.R_earth ** 2) * np.sqrt(stf.Constants.mu_earth))

        self.residual = np.zeros([2,1])
        self.update_lock = Lock()
        self.pub =  rospy.Publisher("observation", Marker, queue_size=10)

    #calculates target orbital elements based on
    # current filter state and current position of chaser
    def update_target_oe(self):




        rel_elem = stf.QNSRelOrbElements()
        rel_elem.from_vector(self.x)

        self.oe_t = stf.KepOrbElem()
        self.oe_t.from_qns_relative(rel_elem, self.oe_c)


        self.has_oe = True

    print "============"

    def get_H_of_x(self):
        # get cartesian locations of both


        target_osc_oe = stf.OscKepOrbElem()
        target_osc_oe.from_mean_elems(self.oe_t)

        cart_c = stf.Cartesian()
        cart_c.from_keporb(self.oe_c)

        cart_t = stf.Cartesian()
        cart_t.from_keporb(target_osc_oe)

        p_teme = (cart_t.R - cart_c.R) * 1000
        p_body = np.dot(self.R_body[0:3, 0:3].T, p_teme)



        #calculate angles
        alpha = np.arcsin(p_body[1] / np.linalg.norm(p_body))
        eps = np.arctan2(p_body[0], p_body[2])

        # correct quadrant
        # if p_body[0] > 0 and p_body[1] > 0:  # quadrant I
        #     pass
        #     print "Q1"
        # elif p_body[0] < 0 and p_body[1] > 0:  # II
        #     alpha = np.pi - alpha
        #     print "Q2"
        # elif p_body[0] < 0 and p_body[1] < 0:  # III
        #     alpha = np.pi - alpha
        #     print "Q3"
        # elif p_body[0] > 0 and p_body[1] < 0:  # IV
        #     alpha = 2 * np.pi + alpha
        #     print "Q4"

        # publish observation
        msg = Marker()
        msg.header.frame_id = "cso"
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.points.append(Point(0,0,0))
        msg.points.append(Point(p_body[0], p_body[1], p_body[2]))
        msg.scale.x = 100
        msg.scale.y = 200
        msg.color.a = 1.0
        msg.color.g = 1.0

        self.pub.publish(msg)
        print "H(x):", p_body

        z = np.zeros([2,1])
        z[0] = alpha
        z[1] = eps
        return [z, np.linalg.norm(p_body)]

    def get_H(self):

        a_c = self.oe_c.a
        e_c = self.oe_c.e
        w_c = self.oe_c.w
        i_c = self.oe_c.i
        T_c = self.oe_c.v
        s_wc = np.sin(w_c)
        c_wc = np.cos(w_c)

        #puuh this is going to be long.
        #partial derivative from oe to oe_star
        d_oe_oes = np.zeros([6,6])
        eta = np.sqrt(1.0 - e_c ** 2.0)
        xi =(1.0+0.5*e_c**2)/eta**3
        d_oe_oes[0,:] = [1.0,0,0,0,0,0]
        d_oe_oes[1,:] = [0, xi, s_wc*(xi-1.0)/e_c, -c_wc*(xi-1.0)/e_c, 0, -(xi-1.0)/np.tan(i_c)]
        d_oe_oes[2,:] = [0, 0, c_wc/eta**2, s_wc/eta**2, 0, 0]
        d_oe_oes[3,:] = [0, -e_c/eta**3, -s_wc/eta**3, -s_wc/eta**3, 0, e_c/(eta**3*np.tan(i_c))]
        d_oe_oes[4,:] = [0, 0, 0, 0, 1.0, 0]
        d_oe_oes[5, :] = [0, 0, 0, 0, 0, 1.0]

        # partial derivative from oes to RTN
        phi = np.zeros([3,6])
        c_f = np.cos(T_c)
        s_f = np.sin(T_c)

        k = 1 + e_c*c_f
        phi[0,:] = [1.0, 0, -k*c_f, -k*s_f, 0, 0]
        phi[1,:] = [0, 1.0, (k+1.0)*s_f, -(k+1.0)*c_f + e_c/2.0, 0, 0]
        phi[2,:] = [0, 0, 0, 0, np.sin(T_c+w_c), -np.cos(T_c+w_c)]
        r_f = a_c*1000 #orbital radius?!! TODO

        #partial derivate from RTN to camera frame
        d_C_R = np.eye(3)

        #partial derivative from  camera frame to measurement
        d_y_C = np.zeros([2,3])
        [z_ideal, norm_C] = self.get_H_of_x()
        alpha = z_ideal[0]
        eps = z_ideal[1]

        d_y_C[0, :] = [-np.sin(alpha) * np.sin(eps), np.cos(alpha), -np.sin(alpha) * np.cos(eps)]
        d_y_C[1, :] = [(1.0 / np.cos(alpha)) * np.cos(eps), 0, -(1.0 / np.cos(alpha)) * np.sin(eps)]
        d_y_C = (1.0 / norm_C) * d_y_C

        # TODO TEST AND FIX NOTATION
        return d_y_C.dot(d_C_R).dot(r_f*phi).dot(d_oe_oes)


    def get_Dx(self):
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
        a_t = a_s*(1+self.x[0])
        e_t_c_wt = e_s*c_ws + self.x[2]
        e_t_s_wt = e_s*s_ws + self.x[3]
        i_t = i_s + self.x[4]

        e_t = np.sqrt(e_t_c_wt**2+e_t_s_wt**2)
        c_it = np.cos(i_t)
        s_it = np.sin(i_t)

        Q_t = (5.0* c_it**2 - 1.0)
        Q_s = (5.0 * c_is ** 2 - 1.0)

        # part1
        p1 = np.zeros([6, 1])
        p1[1] = np.sqrt(1.0-e_t**2) * (3.0*c_it**2 - 1) + Q_t - 2.0*c_it*c_is
        p1[2] = -e_t_s_wt * Q_t
        p1[3] = e_t_c_wt * Q_t
        p1[5] = -2.0 * c_it * s_is
        m_p1 = (self.J2_term * np.power(a_t, -3.5))/((1-e_t**2)**2)

        # part2
        p2 = np.zeros([6, 1])
        p2[1] = (1.0 + np.sqrt(1.0-e_s**2))*(3.0*c_is**2-1.0)
        p2[2] = -e_s * s_ws * Q_s
        p2[3] =  e_s * c_ws * Q_s
        p2[5] = -2 * c_it * s_is
        m_p2 = (self.J2_term * np.power(a_s, -3.5))/((1-e_s**2)**2)

        # part 3
        p3 = np.zeros([6, 1])
        p3[1] = np.power(a_t, -1.5) - np.power(a_s, -1.5)
        m_p3 = np.sqrt(stf.Constants.mu_earth)

        d_x = m_p1 * p1 - m_p2 * p2 + m_p3 * p3

        return d_x


    def advance_state(self, delta_t):
        # advance state according to timestep from msg
        dx = self.get_Dx() * delta_t  # primitive integrator

        self.x = self.x + dx

        # advance covariance according to timestep
        F = self.stm.get_matrix(self.oe_c.as_array_mean(), delta_t)
        self.P = F.dot(self.P).dot(F.T) + self.Q


    # Performs prediction step
    def callback_state(self, chaser_oe):
        if self.t > chaser_oe.header.stamp:
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!! ignore"
            return

        oe_c_osc = stf.OscKepOrbElem()
        oe_c_osc.from_message(chaser_oe.position)

        self.update_lock.acquire()


        self.oe_c = stf.KepOrbElem()
        self.oe_c.from_osc_elems(oe_c_osc)


        self.time_c = chaser_oe.header.stamp
        # store orientation
        self.R_body = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                    chaser_oe.orientation.y,
                                                    chaser_oe.orientation.z,
                                                    chaser_oe.orientation.w])


        # update time
        if self.t != 0:
            delta_t = (chaser_oe.header.stamp - self.t).to_sec()
            # advance state
            if delta_t > 0.1:
                self.advance_state(delta_t)
                self.t = chaser_oe.header.stamp

        # update target location
        self.update_target_oe()
        self.update_lock.release()


    # Performs measurement update
    def callback_aon(self, meas_msg):
        return
        if self.t == 0 or self.R_body is None:
            #discard measurement
            return

        self.update_lock.acquire()

        # advance state to current measurement
        delta_t = (meas_msg.header.stamp - self.t).to_sec()
        self.t = meas_msg.header.stamp

        self.advance_state(delta_t)

        # get angles
        z_k = np.zeros([2, 1])
        z_k[0] = meas_msg.value.azimut
        z_k[1] = meas_msg.value.elevation

        # calculate ideal measurement given advanced state
        h_xk, foo = self.get_H_of_x()

        # calculate innovation
        y = z_k - h_xk
        # get measurement sensitivity matrix



        print z_k, h_xk, abs(y) > np.pi

        H = self.get_H()

        # calculate residual covariance
        S = H.dot(self.P).dot(H.T) + self.R

        # get kalman gain
        # NOTE: may fail if S is close to be singular - pseudo inverse?
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.residual = y
        print "Inno=", np.linalg.norm(K.dot(y))
        self.P = (np.eye(6)-K.dot(H)).dot(self.P)

        # update target location
        self.update_target_oe()
        self.update_lock.release()

    def get_state(self):
        return [self.x, self.P]

    def get_target_oe(self):
        self.update_lock.acquire()
        copy_oe_t = copy(self.oe_t)
        self.update_lock.release()
        return copy_oe_t


