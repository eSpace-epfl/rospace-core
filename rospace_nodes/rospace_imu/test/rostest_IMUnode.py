#!/usr/bin/env python
import unittest
import sys
import os
import rospy
import numpy as np

from rospace_msgs.msg import PoseVelocityStamped
from sensor_msgs.msg import Imu

PKG = 'rospace_imu'
MSG_DELAY = 0.2

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...

from rospace_lib.sensor.ADXRS614 import *


class IMUnodeTest(unittest.TestCase):

    def setUp(self):
        rospy.init_node("test_IMUnode")

        spacecraft = rospy.get_param("~spacecraft")

        self.imu_sub = rospy.Subscriber("/" + spacecraft + "/imu", Imu, self.imu_callback)

        self.pose_sub = rospy.Subscriber("/" + spacecraft + "/pose", PoseVelocityStamped, self.pose_callback)

        self.test_settings = spacecraft + "/propagator_settings"

        # wait until Propagator initializes
        while not hasattr(self, 'pose_time'):
            rospy.sleep(0.001)

    def imu_callback(self, msg):
        self.imu_time = msg.header.stamp
        self.imu_data = np.zeros(3)
        self.imu_data[0] = msg.angular_velocity.x
        self.imu_data[1] = msg.angular_velocity.y
        self.imu_data[2] = msg.angular_velocity.z

    def pose_callback(self, msg):
        self.pose_time = msg.header.stamp

    def test_one_equals_one(self):
        rospy.loginfo("-D- test_one_equals_one")
        self.assertEquals(1, 1, "1!=1")

    def test_integration_of_node(self):
        pass

if __name__ == '__main__':
    import rostest
    rostest.rosrun(PKG, 'integration_test_IMUnode', IMUnodeTest)

    print("Done with it baby")
