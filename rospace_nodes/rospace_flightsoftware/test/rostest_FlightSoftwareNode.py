#!/usr/bin/env python
# Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

import unittest
import sys
import os
import rospy
import numpy as np

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu

PKG = 'rospace_flightsoftware'
CHECK_MSGS = 10  # number of messages to be checked in every test

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...


class FlightSoftwareNodeTest(unittest.TestCase):
    """Node integration test for Flight software node.

    These test check the correct implementation of the interfaces for the Flight Software node. It is checked
    if messages required for the correct functioning of the node are correctly received or send.

    Attributes:
        b_sub (rospy.Subscriber): subscriber to the "B_field" topic.
        i_sub (rospy.Subscriber): subscriber to the "imu" topic.
        c_sub (rospy.Subscriber): subscriber to the "I_magneto" topic.

    """

    def setUp(self):
        rospy.init_node("test_FlightSoftware")

        spacecraft = rospy.get_param("~spacecraft")

        self.b_sub = rospy.Subscriber("/" + spacecraft + "/B_field", Vector3Stamped, self.b_field_callback)
        self.i_sub = rospy.Subscriber("/" + spacecraft + "/imu", Imu, self.imu_callback)
        self.c_sub = rospy.Subscriber("/" + spacecraft + "/I_magneto", Vector3Stamped, self.controller_callback)

        # Wait until PropagatorNode finishes initialization
        while not hasattr(self, "b_field_time"):
            rospy.sleep(0.001)

        rospy.sleep(0.2)  # sleep so all messages really updated!

    def b_field_callback(self, b_field_msg):
        """B-field subscriber callback storing last published message.

        Args:
            b_field_msg (geometry_msgs.msg.Vector3Stamped): published b-field message

        """
        self.b_field_time = b_field_msg.header.stamp

        self.b_field = np.zeros(3)
        self.b_field[0] = b_field_msg.vector.x
        self.b_field[1] = b_field_msg.vector.y
        self.b_field[2] = b_field_msg.vector.z

    def imu_callback(self, imu_msg):
        """IMU subscriber callback storing last published message.

        Args:
            imu_msg (sensor_msgs.msg.Imu): published IMU message
        """
        self.imu_time = imu_msg.header.stamp

        self.imu_spin = np.zeros(3)
        self.imu_spin[0] = imu_msg.angular_velocity.x
        self.imu_spin[1] = imu_msg.angular_velocity.y
        self.imu_spin[2] = imu_msg.angular_velocity.z

    def controller_callback(self, ctr_msg):
        """Controller subscriber callback storing last published message.

        Args:
            ctr_msg (geometry_msgs.msg.Vector3Stamped): published controller message

        """
        self.ctr_time = ctr_msg.header.stamp

        self.ctr_current = np.zeros(3)
        self.ctr_current[0] = ctr_msg.vector.x
        self.ctr_current[1] = ctr_msg.vector.y
        self.ctr_current[2] = ctr_msg.vector.z

    def wait_for_new_msg(self, topic_time_attribute, old_msg_time):
        """Wait until new message has been published to topic.

        Args:
            topic_time_attribute (string): name of topic_time attribute (e.g.: ctr_time for controller callback)
            old_msg_time (float): time stamp of old message in [s]

        Returns:
            float: time stamp of last received message in [s]

        Raises:
            RuntimeError: if topic_time attribute not initialized yet or no new messages received after 5 seconds.

        """
        if hasattr(self, topic_time_attribute):
            topic_time = getattr(self, topic_time_attribute)
        else:
            raise RuntimeError("No message received to topic: 'self." + topic_time_attribute + "' not initialized!")

        void_time = 0
        while not old_msg_time < topic_time.to_sec():
            # wait for new message at new time-step
            rospy.sleep(0.0001)
            void_time += 0.0001
            if void_time >= 5:
                # 5secs no messages is long enough. Fail the test
                raise RuntimeError("No new messages received by on topic. Attribute " +
                                   topic_time_attribute + " is not updated!")
            topic_time = getattr(self, topic_time_attribute)

        return topic_time.to_sec()

    # Tests ##########################################

    def test_b_field_publishing(self):
        """Check that B-field messages are published to the correct topic."""
        old_msg_time = 0
        for _ in range(0, CHECK_MSGS):
            old_msg_time = self.wait_for_new_msg("b_field_time", old_msg_time)

    def test_imu_publishing(self):
        """Check that IMU messages are published to the correct topic."""
        old_msg_time = 0
        for _ in range(0, CHECK_MSGS):
            old_msg_time = self.wait_for_new_msg("imu_time", old_msg_time)

    def test_flightsoftware_publishing(self):
        """Check that the flight software is publishing to the correct topic."""
        old_msg_time = 0
        for _ in range(0, CHECK_MSGS):
            old_msg_time = self.wait_for_new_msg("ctr_time", old_msg_time)


if __name__ == "__main__":
    import rostest
    rostest.rosrun(PKG, 'integration_test_FlightSoftwareNode', FlightSoftwareNodeTest)
