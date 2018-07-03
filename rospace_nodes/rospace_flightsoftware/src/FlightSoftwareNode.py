#!/usr/bin/env python
# @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import numpy as np
import rospy
import message_filters

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu

from rospace_lib.swisscube import BDotController


def callback_controller(B_field_data, imu_data):
    """Trigger callback when magnetic field and imu messages received during one time-step.

    Method computes the new controller output based on the information in the messages and publishes
    them.

    Args:
        B_field_data (geometry_msgs.msg.Vector3Stamped): B-field message
        imu_data (sensor_msgs.msg.Imu): Imu message

    """
    B_field = np.zeros(3)
    B_field[0] = B_field_data.vector.x
    B_field[1] = B_field_data.vector.y
    B_field[2] = B_field_data.vector.z

    spin = np.zeros(3)
    spin[0] = imu_data.angular_velocity.x
    spin[1] = imu_data.angular_velocity.y
    spin[2] = imu_data.angular_velocity.z

    timestamp = B_field_data.header.stamp.to_sec()

    mt_current = ctrl.run_controller(B_field, spin, timestamp)

    msg = Vector3Stamped()
    msg.header.stamp = B_field_data.header.stamp
    msg.vector.x = mt_current[0]
    msg.vector.y = mt_current[1]
    msg.vector.z = mt_current[2]
    pub_magnetorquer.publish(msg)


if __name__ == '__main__':
    try:
        rospy.init_node("BDotController")
        ctrl = BDotController()

        subs_magnetfield = message_filters.Subscriber("B_field", Vector3Stamped)
        subs_imu = message_filters.Subscriber("imu", Imu)
        Tsync = message_filters.TimeSynchronizer([subs_magnetfield, subs_imu], 10)
        Tsync.registerCallback(callback_controller)

        pub_magnetorquer = rospy.Publisher("torque_current", Vector3Stamped, queue_size=10)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
