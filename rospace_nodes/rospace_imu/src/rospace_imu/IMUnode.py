#!/usr/bin/env python
# Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
# Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.
import rospy
import numpy as np
from rospace_msgs.msg import PoseVelocityStamped
from sensor_msgs.msg import Imu

import importlib


def callback_IMU_pose(data):
    """Feeds "PoseVelocityStamped" message to rate gyroscope model.

    Args:
        data (msg.PoseVelocityStamped): pose stamped message containing the
            spin of the spacecraft
    """
    rotation_rate = np.zeros(3)
    rotation_rate[0] = data.spin.x
    rotation_rate[1] = data.spin.y
    rotation_rate[2] = data.spin.z
    last_callback_time.secs = data.header.stamp.secs
    last_callback_time.nsecs = data.header.stamp.nsecs

    rate_gyro.set_true_value(rotation_rate, 0)


def publish_IMU_message():
    """Publish the IMU sensor readings at a rate of 10[Hz].
    """
    pub = rospy.Publisher("imu", Imu, queue_size=10)

    rate = rospy.Rate(10)  # 10hz

    while not rospy.is_shutdown():
        rate_gyro_reading = rate_gyro.get_value()

        msg = Imu()
        msg.header.stamp = last_callback_time
        msg.angular_velocity.x = rate_gyro_reading[0]
        msg.angular_velocity.y = rate_gyro_reading[1]
        msg.angular_velocity.z = rate_gyro_reading[2]

        pub.publish(msg)

        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("imu_node", anonymous=True)

        # retrieves module corresponding to sensor module and class in rosparam
        # load imu configuration into workspace
        imu_cfg = rospy.get_param(rospy.get_namespace(), 0)
        try:
            assert (imu_cfg is not 0)
        except AssertionError:
            raise ValueError("Imu configuration could not be found in spacecraft cfg file!")

        # find correct sensor whose name ends with "imu"
        imu_cfg = [value for key, value in imu_cfg.iteritems() if key.endswith("imu")][0]

        sensor_module = importlib.import_module("rospace_lib.sensor." + imu_cfg["module"])
        sensor_class = getattr(sensor_module, imu_cfg["class"])
        rate_gyro = sensor_class()

        import_sensor_lib = importlib.import_module("matplotlib.text")

        last_callback_time = rospy.Time(0, 0)
        subs = rospy.Subscriber("pose", PoseVelocityStamped, callback_IMU_pose)
        publish_IMU_message()
    except rospy.ROSInterruptException:
        pass
