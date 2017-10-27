import numpy as np
import rospy
from tf import transformations
def R_x(q): return np.array([[1, 0, 0],[0, np.cos(q), -np.sin(q)],[0, np.sin(q), np.cos(q)]])
def R_y(q): return np.array([[np.cos(q), 0, np.sin(q)],[0, 1, 0],[-np.sin(q), 0, np.cos(q)]])
def R_z(q): return np.array([[np.cos(q), -np.sin(q), 0],[np.sin(q), np.cos(q), 0],[0, 0, 1]])

class BaseRelativeOrbitalFilter:
    # used references:
    # [1] the other one... TODO

    # [2] Improved maneuver-free approach to angles only navigation
    #       for space rendez-vous" by Sullivan/Koenig/D'Amico

    def __init__(self):
        #00138646026702
        self.x = np.array([[0.005, 0.0, -2.68790078041e-06, -1.91228213279e-06, -1.34998766832e-07, -0.000442743021758]]).T
        self.alpha_c = np.zeros([6, 1])
        self.T_c = 0
        self.P = np.ones([6,6])*0.001 #([0.01, 0.2, 0.001, 0.001, 0.001, 0.001])
        self.Q = np.diag([0.01, 0.1, 0.0001, 0.0001, 0.0001, 0.0001])
        self.R = np.array([[0.01, 0.01],[-0.01, 0.01]])
        print self.R
        self.t = rospy.Time(0, 0)

    #calculates target orbital elements based on
    # current filter state and current position of chaser
    def get_target_oe(self):
        a_s = self.alpha_c[0]
        M_s = self.alpha_c[1]
        e_s = self.alpha_c[2]
        w_s = self.alpha_c[3]
        i_s = self.alpha_c[4]
        O_s = self.alpha_c[5]
        c_ws = np.cos(w_s)
        s_ws = np.sin(w_s)
        c_is = np.cos(i_s)
        s_is = np.sin(i_s)
        u_s = M_s + w_s

        a_t = a_s * (1 + self.x[0])
        e_t_c_wt = e_s * c_ws + self.x[2]
        e_t_s_wt = e_s * s_ws + self.x[3]
        i_t = i_s + self.x[4]
        O_t = self.x[5] * s_is + O_s

        e_t = np.sqrt(e_t_c_wt ** 2 + e_t_s_wt ** 2)

        # arbitrarly chosen! check this
        if e_t > 10**(-7):
            w_t = np.arccos(e_t_c_wt/e_t)
        else:
            # circular orbit => w_t not defined
            w_t = 0

        u_t = u_s + (self.x[1]-c_is*(O_t - O_s))
        M_t = u_t - w_t

        alpha_t = np.array([a_t, M_t, e_t, w_t, i_t, O_t])
        return alpha_t

    def get_H_of_x(self):
        # Get absolute state of target based on current values
        alpha_t = self.get_target_oe()
        a_t = alpha_t[0]
        M_t = alpha_t[1]
        e_t = alpha_t[2]
        w_t = alpha_t[3]
        i_t = alpha_t[4]
        O_t = alpha_t[5]

        a_c = self.alpha_c[0]
        M_c = self.alpha_c[1]
        e_c = self.alpha_c[2]
        w_c = self.alpha_c[3]
        i_c = self.alpha_c[4]
        O_c = self.alpha_c[5]
        T_c = self.T_c

        #convert mean anomaly to eccentric anomaly
        E=0
        for i in range(0, 15):
            E = E + (M_t + e_t*np.sin(E)-E)/(1-e_t*np.cos(E))

        # get true anomaly
        T = 2*np.arctan2(np.sqrt(1+e_t)*np.sin(E/2.0), np.sqrt(1-e_t)*np.cos(E/2.0))


        # convert to chaser cartesian and get vector
        p_c = a_c * (1 - e_c ** 2)
        R_c_per = np.array([p_c * np.cos(T_c), p_c * np.sin(T_c), 0]) / (1.0 + e_c * np.cos(T_c))
        R_c = R_z(O_c).dot(R_x(i_c)).dot(R_z(w_c)).dot(R_c_per)
        # convert to TARGET cartesian and get vector
        p_t = a_t * (1 - e_t ** 2)
        R_t_per = np.array([p_t * np.cos(T), p_t * np.sin(T), 0]) / (1.0 + e_t * np.cos(T))
        R_t = R_z(O_t).dot(R_x(i_t)).dot(R_z(w_t)).dot(R_t_per)

        p_teme = (R_t - R_c) * 1000
        p_body = np.dot(self.R_body[0:3, 0:3].T, p_teme)
        #calculate angles
        alpha = np.arcsin(p_body[1] / np.linalg.norm(p_body))
        eps = np.arctan(p_body[0] / p_body[2])

        z = np.zeros([2,1])
        z[0] = alpha
        z[1] = eps
        return [z, np.linalg.norm(p_body)]

    def get_H(self):
        alpha_t = self.get_target_oe()
        a_t = alpha_t[0]
        M_t = alpha_t[1]
        e_t = alpha_t[2]
        w_t = alpha_t[3]
        i_t = alpha_t[4]
        O_t = alpha_t[5]


        a_c = self.alpha_c[0]
        M_c = self.alpha_c[1]
        e_c = self.alpha_c[2]
        w_c = self.alpha_c[3]
        i_c = self.alpha_c[4]
        O_c = self.alpha_c[5]
        T_c = self.T_c
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
        c_f = np.cos(self.T_c)
        s_f = np.sin(self.T_c)

        k = 1 + e_c*c_f
        phi[0,:] = [1.0, 0, -k*c_f, -k*s_f, 0, 0]
        phi[1,:] = [0, 1.0, (k+1.0)*s_f, -(k+1.0)*c_f + e_c/2.0, 0, 0]
        phi[2,:] = [0, 0, 0, 0, np.sin(self.T_c+w_c), -np.cos(self.T_c+w_c)]
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

    def get_STM(self, alpha_c, t):
        J_qns = self.get_J_qns(alpha_c)

        A_kep = self.get_A_kep_qns(alpha_c)
        A_j2 = self.get_A_j2_qns(alpha_c)

        mu = 3.986004418 * 10 ** (14 - 9)  # [km^3 * s^-2]
        a = alpha_c[0]
        e = alpha_c[2]
        i = alpha_c[4]

        # substitutions
        eta = np.sqrt(1.0 - e ** 2.0)
        R_E = 6378.137
        J_2 = 1.08262668 * 10 ** (-3.0)

        kappa = (3.0 / 4.0) * (J_2 * (R_E ** 2) * np.sqrt(mu)) / (np.power(a, 3.5) * np.power(eta, 4))
        w_dot = np.array([[0, 0, 0, -kappa * 5.0 * (np.cos(i) ** 2 - 1.0), 0, 0]]).T
        J_qns_t2 = self.get_J_qns(alpha_c + w_dot * t)
        phi_j2 = J_qns_t2.T.dot(np.eye(6) + (A_kep + A_j2) * t).dot(J_qns)
        return phi_j2

    def get_Dx(self, alpha_c, x):
        # Implementation of Equation 10 of [2]
        # variables
        mu = 3.986004418 * 10 ** (14 - 9)  # [km^3 * s^-2]
        R_E = 6378.137
        J_2 = 1.08262668 * 10 ** (-3)
        y = 0.75 * J_2 * R_E**2 * np.sqrt(mu)

        # chaser variables
        a_s = alpha_c[0]
        e_s = alpha_c[2]
        w_s = alpha_c[3]
        i_s = alpha_c[4]

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
        p1 = np.zeros([6, 1])
        p1[1] = np.sqrt(1.0-e_t**2) * (3.0*c_it**2 - 1) + Q_t - 2.0*c_it*c_is
        p1[2] = -e_t_s_wt * Q_t
        p1[3] = e_t_c_wt * Q_t
        p1[5] = -2.0 * c_it * s_is
        m_p1 = (y*np.power(a_t, -3.5))/((1-e_t**2)**2)

        # part2
        p2 = np.zeros([6, 1])
        p2[1] = (1.0 + np.sqrt(1.0-e_s**2))*(3.0*c_is**2-1.0)
        p2[2] = -e_s * s_ws * Q_s
        p2[3] =  e_s * c_ws * Q_s
        p2[5] = -2 * c_it * s_is
        m_p2 = (y*np.power(a_s, -3.5))/((1-e_s**2)**2)

        # part 3
        p3 = np.zeros([6, 1])
        p3[1] = np.power(a_t, -1.5) - np.power(a_s, -1.5)
        m_p3 = np.sqrt(mu)

        d_x = m_p1 * p1 - m_p2 * p2 + m_p3 * p3

        return d_x

    def get_mean_anomaly(self, e, t):
        sin_e = np.sqrt(1 - e ** 2) * np.sin(t)
        cos_e = (e + np.cos(t)) / (1 + e * np.cos(t))
        m = np.arctan2(sin_e, cos_e)
        if m < 0:
            m = m + 2 * np.pi
        return m

    def advance_state(self, delta_t):
        # advance state according to timestep from msg
        dx = self.get_Dx(self.alpha_c, self.x) * delta_t  # primitive integrator

        self.x = self.x + dx

        # advance covariance according to timestep
        F = self.get_STM(self.alpha_c, delta_t)
        self.P = F.dot(self.P).dot(F.T) + self.Q


    # Performs prediction step
    def callback_state(self, chaser_oe):
        if self.t > chaser_oe.header.stamp:
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!! ignore"
            return


        M_c = self.get_mean_anomaly(chaser_oe.position.eccentricity, np.deg2rad(chaser_oe.position.true_anomaly))
        a_c = chaser_oe.position.semimajoraxis
        w_c = np.deg2rad(chaser_oe.position.arg_perigee)
        e_c = chaser_oe.position.eccentricity
        i_c = np.deg2rad(chaser_oe.position.inclination)
        O_c = np.deg2rad(chaser_oe.position.raan)
        self.T_c = np.deg2rad(chaser_oe.position.true_anomaly)
        self.time_c = chaser_oe.header.stamp
        # store orientation
        self.R_body = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                    chaser_oe.orientation.y,
                                                    chaser_oe.orientation.z,
                                                    chaser_oe.orientation.w])

        # update own position
        self.alpha_c = np.array([[a_c, M_c, e_c, w_c, i_c, O_c]]).T

        # update time
        if self.t != 0:
            delta_t = (chaser_oe.header.stamp - self.t).to_sec()
            self.t = chaser_oe.header.stamp

            # advance state
            self.advance_state(delta_t)
        else:
            self.t = chaser_oe.header.stamp


    # Performs measurement update
    def callback_aon(self, meas_msg):

        if self.t == 0:
            #discard measurement
            return


        # advance state to current measurement
        delta_t = (meas_msg.header.stamp - self.t).to_sec()
        self.t = meas_msg.header.stamp

        #self.advance_state(delta_t)

        # get angles
        z_k = np.zeros([2, 1])
        z_k[0] = meas_msg.value.azimut
        z_k[1] = meas_msg.value.elevation

        # calculate ideal measurement given advanced state
        h_xk, foo = self.get_H_of_x()

        # calculate innovation
        y = z_k - h_xk
        # get measurement sensitivity matrix

        H = self.get_H()

        # calculate residual covariance
        S = H.dot(self.P).dot(H.T) + self.R

        # get kalman gain
        # NOTE: may fail if S is close to be singular - pseudo inverse?
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        # update state
        # TEST H MATRIX!

        print z_k.T, h_xk.T, y.T
        print np.linalg.norm(S)

        self.x = self.x + K.dot(y)

        print np.linalg.norm(self.P)
        self.P = (np.eye(6)-K.dot(H)).dot(self.P)
        print np.linalg.norm(self.P)
        print "------"





    def get_state(self):
        return [self.x, self.P]


    # Methods for STM calculation
    def get_A_kep_qns(self, alpha_c):
        mu = 3.986004418 * 10 ** (14 - 9)  # [km^3 * s^-2]
        n = np.sqrt(mu) / np.power(alpha_c[0], 1.5)
        A_kep_qns = np.zeros([6, 6])
        A_kep_qns[1, 0] = -1.5 * n
        return A_kep_qns

    def get_A_j2_qns(self, alpha_c):
        mu = 3.986004418 * 10 ** (14 - 9)  # [km^3 * s^-2]

        a = alpha_c[0]
        e = alpha_c[2]
        i = alpha_c[4]

        # substitutions
        eta = np.sqrt(1.0 - e ** 2.0)
        R_E = 6378.137
        J_2 = 1.08262668 * 10 ** (-3)

        kappa = (3.0 / 4.0) * (J_2 * (R_E ** 2) * np.sqrt(mu)) / (np.power(a, 3.5) * np.power(eta, 4))
        E = 1.0 + eta
        F = 4.0 + 3.0 * eta
        G = 1.0 / eta ** 2.0
        P = 3.0 * np.cos(i) ** 2.0 - 1.0
        Q = 5.0 * np.cos(i) ** 2.0 - 1.0
        R = np.cos(i)
        S = np.sin(2.0 * i)
        T = np.sin(i) ** 2.0
        U = np.sin(i)
        V = np.tan(i / 2.0)
        W = np.cos(i / 2.0) ** 2.0

        # build matrix
        A_j2_qns = np.zeros([6, 6])
        A_j2_qns[1, :] = [-7.0 / 2.0 * E * P, 0, e * F * G * P, 0, -F * S, 0]
        A_j2_qns[3, :] = [-7.0 / 2.0 * e * Q, 0, 4.0 * e ** 2.0 * G * Q, 0, -5.0 * e * S, 0]
        A_j2_qns[5, :] = [7.0 / 2.0 * S, 0, -4.0 * e * G * S, 0, 2.0 * T, 0]
        A_j2_qns = kappa * A_j2_qns
        return A_j2_qns

    def get_J_qns(self, alpha_c):
        w = alpha_c[3]
        J_qns = np.eye(6)
        J_qns[2, 2:4] = [np.cos(w), np.sin(w)]
        J_qns[3, 2:4] = [-np.sin(w), np.cos(w)]
        return J_qns
