#!/usr/bin/env python
# @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.
import rospy
import numpy as np
from geometry_msgs.msg import Vector3Stamped, WrenchStamped

from rospace_lib.actuators import ThreeAxisMagnetorquer


def callback_B_field(data):
    """Callback triggered by new Message on Magnetic field topic.

    Method updates the B-field for the magnetorquer, computes
    the new torque based on the field and publishes the torque.

    Args:
        data (geometry_msgs.msg.Vector3Stamped): Magnetic field message

    """
    B_field_vector = np.zeros(3)
    B_field_vector[0] = data.vector.x
    B_field_vector[1] = data.vector.y
    B_field_vector[2] = data.vector.z

    torquer.set_B_field(B_field_vector)

    torque = torquer.get_torque()

    msg = WrenchStamped()
    msg.header.stamp = data.header.stamp
    msg.wrench.force.x = 0.0
    msg.wrench.force.y = 0.0
    msg.wrench.force.z = 0.0
    msg.wrench.torque.x = torque[0]
    msg.wrench.torque.y = torque[1]
    msg.wrench.torque.z = torque[2]

    pubTorque.publish(msg)


def callback_I_magneto(data):
    """Triggered by new message provided by controller providing the current current.

    This method updates the current of the magnetorquer.

    Args:
        data (geometry_msgs.msg.Vector3Stamped): new current set by controller

    """
    I_vector = np.zeros(3)
    I_vector[0] = data.vector.x
    I_vector[1] = data.vector.y
    I_vector[2] = data.vector.z

    torquer.set_I(I_vector)


if __name__ == '__main__':
    try:
        torquer = ThreeAxisMagnetorquer(windings=427, area=0.0731**2, radius=0.0354)
        rospy.init_node('magnetorquer_node', anonymous=True)

        pubTorque = rospy.Publisher("torque", WrenchStamped, queue_size=10)
        subsB = rospy.Subscriber("B_field", Vector3Stamped, callback_B_field)
        subsI = rospy.Subscriber("I_magneto", Vector3Stamped, callback_I_magneto)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
