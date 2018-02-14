import numpy as np

import space_tf as stf


class RelativeOrbitalSTM:

    def __init__(self):
        self.J2_term = (3.0 / 4.0) * (stf.Constants.J_2 * (stf.Constants.R_earth ** 2) * np.sqrt(stf.Constants.mu_earth))


    def get_matrix(self, alpha_c, dt):
        J_qns = self.get_J_qns(alpha_c)

        A_kep = self.get_A_kep_qns(alpha_c)
        A_j2 = self.get_A_j2_qns(alpha_c)

        a = alpha_c[0]
        e = alpha_c[2]
        i = alpha_c[4]

        # substitutions
        eta = np.sqrt(1.0 - e ** 2.0)

        kappa = self.J2_term / (np.power(a, 3.5) * np.power(eta, 4))
        w_dot = np.array([[0, 0, 0, -kappa * 5.0 * (np.cos(i) ** 2 - 1.0), 0, 0]]).T
        J_qns_t2 = self.get_J_qns(alpha_c + w_dot * dt)
        phi_j2 = J_qns_t2.T.dot(np.eye(6) + (A_kep + A_j2) * dt).dot(J_qns)
        return phi_j2


    def get_A_kep_qns(self, alpha_c):
        n = np.sqrt(stf.Constants.mu_earth) / np.power(alpha_c[0], 1.5)
        A_kep_qns = np.zeros([6, 6])
        A_kep_qns[1, 0] = -1.5 * n
        return A_kep_qns

    def get_A_j2_qns(self, alpha_c):
        a = alpha_c[0]
        e = alpha_c[2]
        i = alpha_c[4]

        # substitutions
        eta = np.sqrt(1.0 - e ** 2.0)

        kappa = self.J2_term / (np.power(a, 3.5) * np.power(eta, 4))
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

    def get_B_oe(self, alpha_c):
        a = alpha_c[0]
        e = alpha_c[2]
        i = alpha_c[4]
        w = alpha_c[3]
        f_m = alpha_c[1] # true anomaly at t
        phi_m = alpha_c[1]+alpha_c[3] # true argument of lattitude at t
        n = np.sqrt(stf.Constants.mu_earth) / np.power(alpha_c[0], 1.5)

        # substitutions
        eta = np.sqrt(1.0 - e ** 2.0)

        k = 1+e*np.cos(f_m)
        e_x = e * np.cos(w)
        e_y = e * np.sin(w)

        B_oe = np.zeros((6,3))
        B_oe[0,:] = [2/eta**2*e*np.sin(f_m), (2*k)/eta**2, 0]

        B_oe[1,0] = ((eta-1)*k*np.cos(f_m)-2*eta*e)/(e*k)
        B_oe[1,1] = (-(eta-1)*(k+1)*np.sin(f_m))/(e*k)

        B_oe[2,0] = np.sin(phi_m)
        B_oe[2,1] = ((k+1)*np.cos(phi_m)+e_x)/k
        B_oe[2,2] = (e_y*np.sin(phi_m))/(k*np.tan(i))

        B_oe[3, 0] = -np.cos(phi_m)
        B_oe[3, 1] = ((k + 1) * np.sin(phi_m) + e_y) / k
        B_oe[3, 2] = -(e_x * np.sin(phi_m)) / (k * np.tan(i))

        B_oe[4,:] = [0, 0, np.cos(phi_m)/k]
        B_oe[5, :] = [0, 0, np.sin(phi_m) / k]

        return -eta/(a*n)*B_oe


    def get_Phi_emp(self, tau_emp, delta_t):
        psi = np.exp(-delta_t/tau_emp)
        B_emp = np.diag([1, 1, 1]* psi)
        return B_emp


    def get_Phi_b(self, tau_b, delta_t):
        psi = np.exp(-delta_t / tau_b)
        B_emp = np.diag([1, 1] * psi)
        return B_emp






