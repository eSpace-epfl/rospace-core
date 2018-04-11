import numpy as np


class BDotController(object):
    """ Simple implementation of BDOT controller as specified in
        Swisscube Document S3-B-C-ADCS-1-2_BDot_Controller_Code.pdf
        available at
        http://escgesrv1.epfl.ch/06%20-%20Attitude%20control/S3-B-C-ADCS-1-2-Bdot_Controller_Code.pdf
     """
    def __init__(self):
        self.lmbda = 0.5  # [-] (spelled wrong on purpose)
        self.A = 4.861*10**(-3) # [m^2]
        self.K_Bdot = 1.146 * 10**-4  # [Nms]
        self.n = 427 # [-]
        self.maxCurrent = 22*10**-3 # [A]
        self.threshold = 0 # tbd

        self.Bdot_old = np.array([0,0,0])
        self.B_old = np.array([0,0,0])


    def run_controller(self, B_field, w_vec):
        delta_t = 0

        Bdot = (1-self.lmbda)*self.Bdot_old + self.lmbda*(B_field-self.B_old)/delta_t

        w = np.linalg.norm(w_vec)
        K_s = 0
        if w >= self.threshold:
            K_s = -1
        else:
            K_s = 1

        K_B = 1/np.sum(B_field**2)
        I = K_B*Bdot*K_s*self.K_Bdot*(1/(self.A*self.n))
        I_max = max(I)

        # limit current
        K_i  = self.maxCurrent/I_max
        I = I*K_i

        # return currents for torquers
        return I

