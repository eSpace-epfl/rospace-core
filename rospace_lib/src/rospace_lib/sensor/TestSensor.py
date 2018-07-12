# @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.
import numpy as np


class ThreeAxisTestSensor(object):
    """Three Axis model of Test Rate gyroscope.

    For the implementation of every axis see :class:`.TestSensor`.
    """

    def __init__(self):
        self.x_rate_gyro = TestSensor()
        self.y_rate_gyro = TestSensor()
        self.z_rate_gyro = TestSensor()
        # for now, we assume body frame for everything

    def set_true_value(self, rotation_rate, timestamp):
        """Set true value of spacecraft's rotation rate.

        Args:
            rotation_rate (numpy.array): rotation rate for x, y & z axis in [rad/s]
            timestamp (rospy.Time): Header timestamp from ROS message
        """
        self.x_rate_gyro.set_true_value(rotation_rate[0], timestamp)
        self.y_rate_gyro.set_true_value(rotation_rate[1], timestamp)
        self.z_rate_gyro.set_true_value(rotation_rate[2], timestamp)

    def get_value(self):
        """Gets current value of IMU sensor reading.

        Returns:
            numpy.array: Value of the IMU sensor reading
        """
        reading = np.zeros(3)
        reading[0] = self.x_rate_gyro.get_value()
        reading[1] = self.y_rate_gyro.get_value()
        reading[2] = self.z_rate_gyro.get_value()
        return reading


class TestSensor(object):
    """Model of a 1 axis Test Rate gyroscope.

    """
    def __init__(self):
        self.min_rotation = np.deg2rad(-75.0)  # [rad/s]
        self.max_rotation = np.deg2rad(75.0)   # [rad/s]

        self.white_noise_density = np.deg2rad(0.03)  # [rad/s * 1/sqrt(Hz)] resp. \sigma_{g} in [1]
        self.random_walk = np.deg2rad(0.0001)  # [rad/s^2 * 1/sqrt(Hz)] resp. \sigma_{bg} in [1]
        self.update_rate = 1  # [hz]

        # "ideal world value"
        self._current_true_rotation_rate = 0.0

        self._bd = 0

    def set_true_value(self, rotation_rate, timestamp):
        """Stores the current actual rotation_rate of the spacecraft.

        Args:
            rotation_rate (float): rotation rate for x, y & z axis in [rad/s]
            timestamp (rospy.Time): Header timestamp from ROS message
        """
        self._current_true_rotation_rate = rotation_rate

    def get_value(self):
        """Compute and return the value of the current sensor reading.

        Please refer to [1] for an explanation of this model.

        Returns:
            float: True value of rotation rate disturbed by white noise
                and random walk
        """
        # first, update random walk noise.
        sigma_bgd = self.random_walk * np.sqrt(1.0 / self.update_rate)
        self._bd = self._bd + np.random.normal(0.0, sigma_bgd**2)

        # sample higher freq. gaussian white noise
        sigma_gd = self.white_noise_density * 1.0 / np.sqrt(1.0 / self.update_rate)
        nd = np.random.normal(0.0, sigma_gd**2)

        # truncate by min/max values
        truncated_true_rotation = min(self.max_rotation, self._current_true_rotation_rate)
        truncated_true_rotation = max(self.min_rotation, truncated_true_rotation)

        # return true value disturbed by white noise (nd) and random walk (bd)
        return truncated_true_rotation + nd + self._bd
