# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

""" Library for time-handling in a space simulation

Addition to ros-provided rostime library.

Combines current ros-time with a previously set epoch date, such that
current-now is epoch date + ros time.

Also changes rate and

Example:
    Epoch_0 is 23.12.2017 15:47:00 UTC (= date that corresponds to rostime 00:00 (secs:nsecs).
    Current rostime is 10:00 (simulation is running since 10s)
    Now = Epoch_0+Rostime = 23.12.2017 15:47:10
"""

import rospy
from datetime import datetime, timedelta
import time
import inspect


class Epoch:
    """Wrapper for Epochtime in ROS"""

    @staticmethod
    def _changeFrequency(new_frequency):
        """Change ROS parameter for publishing frequency.

        Only the clock module SimTimePublisher is allowed to use this method.

        Raises:
            RuntimeError: If method not called by SimTimePublisher

        """
        frame = inspect.currentframe().f_back
        class_name = inspect.getframeinfo(frame)[0].split("/")[-1][:-3]  # should be "SimTimePublisher"
        if class_name == "SimTimePublisher":
            rospy.set_param('/publish_freq', new_frequency)
        else:
            raise RuntimeError(
                "Only SimTimePublisher.py is allowed to change the Frequency parameter used in Epoch class!")

    @staticmethod
    def _changeStep(new_step):
        """Change ROS parameter for step size.

        Only the clock module SimTimePublisher is allowed to use this method.

        Raises:
            RuntimeError: If method not called by SimTimePublisher

        """
        # Only clock module should change this! Do not change!
        frame = inspect.currentframe().f_back
        class_name = inspect.getframeinfo(frame)[0].split("/")[-1][:-3]  # should be "SimTimePublisher"
        if class_name == "SimTimePublisher":
            rospy.set_param('/time_step_size', str(new_step))
        else:
            raise RuntimeError(
                "Only SimTimePublisher.py is allowed to change the Step Size parameter used in Epoch class!")

    def __init__(self):
        """Get Epoch_0 from rosparam server and saves it."""
        i = 0
        while not rospy.has_param("/epoch") and i < 10 and not rospy.is_shutdown():
            rospy.logwarn("Epoch not avalailable yet. Waiting...")
            i = i + 1
            time.sleep(1)

        self.epoch_string = rospy.get_param('/epoch')
        self.epoch_datetime = datetime.strptime(self.epoch_string, "%Y-%m-%d %H:%M:%S")

        # update class instances
        rospy.loginfo("STORING FREQ AND STEP ###########################")
        Epoch.publish_frequency = rospy.get_param('/publish_freq')
        Epoch.time_step_size = rospy.get_param('/time_step_size')

        rospy.loginfo(str(Epoch.time_step_size))

    def now(self):
        """Get current simulation time in UTC."""
        time_since_epoch = rospy.Time.now()
        time_delta = timedelta(0, time_since_epoch.secs, time_since_epoch.nsecs / 1e3)
        return self.epoch_datetime + time_delta

    def now_jd(self):
        """Get current simulation time as a julian date."""
        now_utc = self.now()
        J2000 = 2451545.0
        J2000_date = datetime(2000, 1, 1, 12, 00, 00)  # UTC time of J2000
        delta = now_utc - J2000_date
        return J2000 + delta.total_seconds() / (60.0 * 60 * 24)

    def Rate(self):
        """Get Rate of simulator driving node.

        The return rate can be used with rospy's sleep method. If the Simulator frequency
        is set to 0.0 (running as fast as possible). The method returns sets the returned
        rate to 1000.

        Returns:
            rospy.Rate: rate corresponding to the frequency of driving node.

        """
        freq = float(rospy.get_param('/publish_freq', None))

        if freq is None:
            raise RuntimeError("No Frequency has been set by the driving node..")
        elif freq == 0.0:
            rospy.logwarn("Simulator running as fast as possible. Returning rate of 1000")
        else:
            return rospy.Rate(freq)

    def time_step(self):
        """Get time step of simulator, set by driving node.

        Returns:
            float: time step of simulation in [ns]

        """
        ts = float(rospy.get_param('/time_step_size', None))

        if ts is None:
            raise RuntimeError("No Time step has been set by the driving node..")
        else:
            return ts
