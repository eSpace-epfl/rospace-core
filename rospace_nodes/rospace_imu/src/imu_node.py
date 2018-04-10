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
from rospace_msgs.msg import PoseVelocityStamped
from ADXRS614 import *

def callback_pose(data):
    rotation_rate = np.zeros(3)
    rotation_rate[0] = data.spin.x
    rotation_rate[1] = data.spin.y
    rotation_rate[2] = data.spin.z

    rate_gyro.set_true_value(rotation_rate)

def publishIMUMessage():
    pub = rospy.Publisher('imu', Imu, queue_size=10)

    rate = rospy.Rate(1) # 10hz



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
        rate_gyro = ThreeAxisADXRS614()
        rospy.init_node('imu_node', anonymous=True)
        subs = rospy.Subscriber("pose", PoseVelocityStamped, callback_pose)
        publishIMUMessage()
    except rospy.ROSInterruptException:
        pass