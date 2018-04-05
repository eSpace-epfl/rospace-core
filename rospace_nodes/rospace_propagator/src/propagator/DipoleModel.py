import math
import numpy as np


class DipoleModel(object):

    def __init__(self):
        self._H_hyst_last = np.empty([0, 1])
        self._Bs = np.empty([0, 1])
        self._m_hyst = np.empty([0, 3])
        self._m_barMag = np.empty([0, 3])
        self._Hc = np.empty([0, 1])
        self._k = np.empty([0, 1])
        self._direction_hyst = np.empty([0, 3])
        self._volume = np.empty([0, 1])
        self._B_hyst_last = np.empty([0, 1])

        self.MU_0 = 4e-7 * math.pi
        self.TWO_DIV_PI = 2. / math.pi

    def addBarMagnet(self, direction, m):
        norm = np.linalg.norm(direction)
        if norm != 1.0:
            direction = direction / norm
        self._m_barMag = np.append(self._m_barMag, [m*direction], axis=0)

    def addHysteresis(self, direction, volume, Hc, Bs, Br=None, upperPoint=None):
        # needed Hc, Bs, Br, volume, dir
        if Br is not None:
            self._Hc = np.insert(self._Hc, self._Hc.size, Hc, axis=0)
            self._Bs = np.insert(self._Bs, self._Bs.size, Bs, axis=0)
            _k = 1 / Hc * math.tan(math.pi * 0.5 * Br / Bs)
            self._k = np.insert(self._k, self._k.size, _k, axis=0)

        elif upperPoint is not None:
            self._Hc = np.insert(self._Hc, self._Hc.size, Hc, axis=0)
            self._Bs = np.insert(self._Bs, self._Bs.size, Bs, axis=0)

            _k = 1 / (upperPoint[1] - Hc) * math.tan(upperPoint[0] * 0.5 * math.pi / Bs)
            self._k = np.append(self._k, [_k], axis=0)

        else:
            raise ValueError("Need more information to create model!")

        # normalize if necessary
        norm = np.linalg.norm(direction)
        if norm != 1.0:
            direction = direction / norm
        self._direction_hyst = np.append(self._direction_hyst, [direction], axis=0)
        self._volume = np.insert(self._volume, self._volume.size, volume, axis=0)

        self._H_hyst_last = np.append(self._H_hyst_last, [[0.]], axis=0)
        self._B_hyst_last = np.append(self._B_hyst_last, [[0.]], axis=0)

    def initializeHysteresisModel(self, B_field):
        self._compute_m_hyst(B_field)

    def getDipoleVectors(self, B_field):
        self._compute_m_hyst(B_field)
        return np.append(self._m_hyst, self._m_barMag, axis=0)

    def _compute_m_hyst(self, B_field):
        _H = B_field / self.MU_0
        H_hyst = np.dot(self._direction_hyst, _H)[:, None]
        sign = np.where(H_hyst > self._H_hyst_last, -1., 1.)  # defining left or right hysteresis
        change = np.where(H_hyst == self._H_hyst_last, 0., 1.)  # keep old value if no change

        B_hyst = self.TWO_DIV_PI * self._Bs * np.arctan(self._k * (H_hyst + (sign * self._Hc)))
        B_hyst = B_hyst * change + self._B_hyst_last * (1 - change)  # use old B_hyst if H didnt change

        self._B_hyst_last = B_hyst
        self._H_hyst_last = H_hyst
        self._m_hyst = (B_hyst * self._volume / self.MU_0) * self._direction_hyst
