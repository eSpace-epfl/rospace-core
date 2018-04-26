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

Example:
    Epoch_0 is 23.12.2017 15:47:00 UTC (= date that corresponds to rostime 00:00 (secs:nsecs).
    Current rostime is 10:00 (simulation is running since 10s)
    Now = Epoch_0+Rostime = 23.12.2017 15:47:10
"""

import rospy
from datetime import datetime, timedelta
import time


class Epoch:
    """Wrapper for Epochtime in ROS"""
    # instances needed to keep nodes synchronous
    publish_frequency = None
    time_step_size = None

    def __init__(self):
        """Gets Epoch_0 from rosparam server and saves it"""
        i = 0
        while not rospy.has_param("/epoch") and i < 10 and not rospy.is_shutdown():
            rospy.logwarn("Epoch not avalailable yet. Waiting...")
            i = i + 1
            time.sleep(1)

        self.epoch_string = rospy.get_param('/epoch')
        self.epoch_datetime = datetime.strptime(self.epoch_string, "%Y-%m-%d %H:%M:%S")

        # update class instances
        Epoch.publish_frequency = rospy.get_param('/publish_freq')
        Epoch.time_step_size = rospy.get_param('/time_step_size')

    def now(self):
        """Returns current simulation time in UTC"""
        time_since_epoch = rospy.Time.now()
        time_delta = timedelta(0, time_since_epoch.secs, time_since_epoch.nsecs / 1e3)
        return self.epoch_datetime + time_delta

    def now_jd(self):
        """Returns current simulation time as a julian date"""
        now_utc = self.now()
        J2000 = 2451545.0
        J2000_date = datetime(2000, 1, 1, 12, 00, 00)  # UTC time of J2000
        delta = now_utc - J2000_date
        return J2000 + delta.total_seconds() / (60.0 * 60 * 24)

    def changeFrequency(self, new_frequency):
        """Changes ROS parameter for publishing frequency"""
        # Only propagator node should change this! Do not change!
        Epoch.publish_frequency = new_frequency
        rospy.set_param('/publish_freq', new_frequency)

    def changeStep(self, new_step):
        """Changes ROS parameter for step size"""
        # Only propagator node should change this! Do not change!
        Epoch.time_step_size = new_step
        rospy.set_param('/time_step_size', str(new_step))