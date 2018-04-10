#!/usr/bin/env python
#  @copyright Copyright (c) 2018, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.
import rospy
from sensor_msgs.msg import Imu
from ADXRS614 import *

def publishIMUMessage():
    pub = rospy.Publisher('imu', Imu, queue_size=10)
    rospy.init_node('imu_node', anonymous=True)
    rate = rospy.Rate(1) # 10hz

    rate_gyro = ThreeAxisADXRS614()

    while not rospy.is_shutdown():
        rate_gyro_reading = np.rad2deg(rate_gyro.get_value())

        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.angular_velocity.x = rate_gyro_reading[0]
        msg.angular_velocity.y = rate_gyro_reading[1]
        msg.angular_velocity.z = rate_gyro_reading[2]

        pub.publish(msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        publishIMUMessage()
    except rospy.ROSInterruptException:
        pass