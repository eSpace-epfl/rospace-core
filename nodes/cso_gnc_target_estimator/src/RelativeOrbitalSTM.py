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




