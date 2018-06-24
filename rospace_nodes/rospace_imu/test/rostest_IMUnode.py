#!/usr/bin/env python
import unittest
import sys
import os
import rospy
import message_filters
import numpy as np

from rospace_msgs.msg import PoseVelocityStamped
from sensor_msgs.msg import Imu

PKG = 'rospace_imu'

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...


class IMUnodeTest(unittest.TestCase):

    def setUp(self):

        rospy.init_node("test_IMUnode")

        # set up publishers and subscribers
        self.spacecraft = rospy.get_param("~spacecraft")
        self.imu_sub = message_filters.Subscriber("/" + self.spacecraft + "/imu", Imu)
        self.pose_sub = message_filters.Subscriber("/" + self.spacecraft + "/pose", PoseVelocityStamped)
        Tsync = message_filters.TimeSynchronizer([self.imu_sub, self.pose_sub], 10)
        Tsync.registerCallback(self.msgs_callback)
        self.msg_nr = 0

        # wait until Propagator initializes
        while not hasattr(self, 'pose_time'):
            rospy.sleep(0.001)

    def msgs_callback(self, imu_msg, pose_msg):
        self.imu_time = imu_msg.header.stamp
        self.imu_data = np.zeros(3)
        self.imu_data[0] = imu_msg.angular_velocity.x
        self.imu_data[1] = imu_msg.angular_velocity.y
        self.imu_data[2] = imu_msg.angular_velocity.z

        self.spin_data = np.zeros(3)
        self.spin_data[0] = pose_msg.spin.x
        self.spin_data[1] = pose_msg.spin.y
        self.spin_data[2] = pose_msg.spin.z
        self.pose_time = pose_msg.header.stamp
        self.msg_nr += 1

    def test_integration_of_node(self):
        """Test that node has been correctly integrated by checking if input value is approximately output value
        for multiple messages."""
        old_msg_nr = self.msg_nr

        container_input = np.zeros(3*20)
        container_input = container_input.reshape(20, 3)
        container_output = np.zeros(3*20)
        container_output = container_output.reshape(20, 3)

        for i in range(0, 30):
            while old_msg_nr == self.msg_nr:
                # wait for new message
                rospy.sleep(0.0001)

            if self.imu_time.secs == self.pose_time.secs and self.imu_time.nsecs == self.pose_time.nsecs:
                container_input[i] = self.spin_data
                container_output[i] = self.imu_data
                self.assertAlmostEquals(container_input[i][0], container_output[i][0], 0)
                self.assertAlmostEquals(container_input[i][1], container_output[i][1], 0)
                self.assertAlmostEquals(container_input[i][2], container_output[i][2], 0)

            old_msg_nr = self.msg_nr

        # the expected mean error between in and output is around 0
        self.assertAlmostEquals(np.mean(container_input - container_output), 0, 1)


if __name__ == '__main__':
    import rostest
    rostest.rosrun(PKG, 'integration_test_IMUnode', IMUnodeTest)
