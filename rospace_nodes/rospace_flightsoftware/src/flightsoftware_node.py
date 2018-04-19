#!/usr/bin/env python
#  @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.
import rospy
from geometry_msgs.msg import Vector3Stamped, WrenchStamped

import numpy as np
from rospace_lib.swisscube import *

def callback_b_field(data):

    B_field = np.zeros(3)

    B_field[0] = data.vector.x
    B_field[1] = data.vector.y
    B_field[2] = data.vector.z

    timestamp = data.header.stamp.to_sec()

    mt_current = ctrl.run_controller(B_field, -1, timestamp)
    print mt_current
    msg = Vector3Stamped()
    msg.header.stamp = data.header.stamp
    msg.vector.x = mt_current[0]
    msg.vector.y = mt_current[1]
    msg.vector.z = mt_current[2]
    pub_magnetotorquer.publish(msg)



if __name__ == '__main__':
    try:
        # do whatever
        rospy.init_node("test")
        ctrl = BDotController()
        subs_magnetfield = rospy.Subscriber("B_field",Vector3Stamped, callback=callback_b_field)
        pub_magnetotorquer = rospy.Publisher("torque_current", Vector3Stamped, queue_size=10)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass