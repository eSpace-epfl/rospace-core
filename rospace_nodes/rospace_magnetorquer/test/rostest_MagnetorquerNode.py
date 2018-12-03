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

from geometry_msgs.msg import WrenchStamped, Vector3Stamped
from rospace_msgs.msg import SatelliteTorque
from sensor_msgs.msg import Imu

PKG = 'rospace_magnetorquer'
CHECK_MSGS = 10  # number of messages to be checked in every test
RUN_FAKE_IMU = False  # bool for turning publisher of fake messages on and off

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...


class MagnetorquerNodeTest(unittest.TestCase):
    """Node Integration test for the Magnetorquer node.

    These test check the correct implementation of the interfaces for the Magnetorquer node. Mainly
    if messages are received or send under different circumstances are checked.

    The first tests with a "0" in their name run without a fake IMU publisher. All other tests activate the publisher
    by setting the global variable RUN_FAKE_IMU.

    The nodes communicating with the magnetorquer node publish under following circumstances:
    :module:`propagator.PropagatorNode`: "B_field" and "dist_torque" at every simulation time-step
    :module:`rospace_flightsoftware.FlightSoftwareNode`: "I_magneto" when messages to "imu" and "B_field" topics
             published at same time-step (synchronized)
    :module:`rospace_magnetorquer.MagnetorquerNode`: "actuator_torque" when message published to "B_field" topic

    Attributes:
        timestep (float): simulation time-step size
        act_msg (list): stores topic at [0] and message type at [1] of "Actuator" topic
        bfield_msg (list): stores topic at [0] and message type at [1] of "B_field" topic
        ctr_msg (list): stores topic at [0] and message type at [1] of "Control" topic
        dtorq_msg (list): stores topic at [0] and message type at [1] of "Disturbance Torque" topic
        b_sub (rospy.Subscriber): subscriber to the "B_field" topic.
        spin_pub (rospy.Publisher): fake IMU publisher

    """

    def setUp(self):
        """Sets up every unit test.

        Every Subscriber has its own callback in which the last published messages as well as its time stamp
        is being stored.

        """
        rospy.init_node("test_Magnetorquer")

        spacecraft = rospy.get_param("~spacecraft")

        self.act_msg = ["/" + spacecraft + "/actuator_torque", WrenchStamped]
        self.bfield_msg = ["/" + spacecraft + "/B_field", Vector3Stamped]
        self.ctr_msg = ["/" + spacecraft + "/I_magneto", Vector3Stamped]
        self.dtorq_msg = ["/" + spacecraft + "/dist_torque", SatelliteTorque]

        self.b_sub = rospy.Subscriber("/" + spacecraft + "/B_field", Vector3Stamped, self.b_field_callback)

        self.spin_pub = rospy.Publisher("/" + spacecraft + "/imu", Imu, queue_size=10)

        _ = rospy.wait_for_message(self.bfield_msg[0], self.bfield_msg[1], timeout=20)

    def b_field_callback(self, b_field_msg):
        """B-field subscriber callback storing last published message.

        In case the global variable "RUN_FAKE_IMU" is set to True, this method
        also publishes a messages to the imu topic with random spin as the IMU
        only publishes if messages for the B-field are received.

        Args:
            b_field_msg (geometry_msgs.msg.Vector3Stamped): published b-field message

        """
        self.b_field_time = b_field_msg.header.stamp

        self.b_field = np.zeros(3)
        self.b_field[0] = b_field_msg.vector.x
        self.b_field[1] = b_field_msg.vector.y
        self.b_field[2] = b_field_msg.vector.z

        if RUN_FAKE_IMU:
            imu_msg = Imu()
            imu_msg.header.stamp = b_field_msg.header.stamp
            imu_msg.angular_velocity.x = np.random.normal(0, 1)
            imu_msg.angular_velocity.y = np.random.normal(0, 1)
            imu_msg.angular_velocity.z = np.random.normal(0, 1)

            self.spin_pub.publish(imu_msg)

    # Unit-tests ###################################################################
    def test_0_no_spin_msg(self):
        """Test that no current is being published if no spin from IMU provided.

        This test has to be run before publishing any message to the imu topic!

        """
        no_msg = None
        for _ in range(0, CHECK_MSGS):
            try:
                no_msg = rospy.wait_for_message(self.ctr_msg[0], self.ctr_msg[1], timeout=0.1)
                if no_msg is not None:
                    raise RuntimeError(
                        "Control attributes are published, even when no information about spin provided!")
            except Exception:
                pass

    def test_0_no_current_no_torque(self):
        """Test that torque equals zero if no current published.

        This test has to be run before publishing any message to the imu topic!

        """

        for _ in range(0, CHECK_MSGS):
            act_msg = rospy.wait_for_message(self.act_msg[0], self.act_msg[1], timeout=5)

            self.assertEqual(act_msg.wrench.torque.x, 0)
            self.assertEqual(act_msg.wrench.torque.y, 0)
            self.assertEqual(act_msg.wrench.torque.z, 0)

    def test_magnetorquer_publishing(self):
        """Test that messages from B_field topic are received by checking that actuator messages are being published."""

        for _ in range(0, CHECK_MSGS):
            _ = rospy.wait_for_message(self.bfield_msg[0], self.bfield_msg[1], timeout=5)

    def test_subscription_to_current(self):
        """Test that current messages are received on correct topic."""
        global RUN_FAKE_IMU
        RUN_FAKE_IMU = True
        rospy.sleep(0.2)

        for _ in range(0, CHECK_MSGS):
            _ = rospy.wait_for_message(self.ctr_msg[0], self.ctr_msg[1], timeout=5)

        RUN_FAKE_IMU = False

    @unittest.skip("This test will be moved in near future to a separate Propagator Integration test.")
    def test_torque_received_by_propagator(self):
        """Test that torques published by actuator are received by propagator and accounted for.

        This test compares the input torques published by the Magnetorquer node and the by the propagator
        published external torques. The external torques are published after the integration of the attitude
        dynamics, hence one time-step after the input torques are published.

        This test will be moved to a separate Propagator Integration test as it is more related to the propagator
        than to the magnetorquer node.
        """
        global RUN_FAKE_IMU
        RUN_FAKE_IMU = True
        rospy.sleep(0.2)

        old_msg_time = self.act_time.to_sec()
        send_torque = self.act_torque

        for _ in range(0, CHECK_MSGS):
            # wait for new message with external torques acting on spacecraft from propagator
            [recv_torque, _] = self.wait_for_new_msg("ext_time", old_msg_time, topic_attribute="ext_torque")

            rospy.loginfo("COMP: " + str(self.ext_time.to_sec()) + "---- " + str(self.act_time.to_sec()))
            self.assertEqual(send_torque[0], recv_torque[0])
            self.assertEqual(send_torque[1], recv_torque[1])
            self.assertEqual(send_torque[2], recv_torque[2])

            # wait for new message from actuator
            old_msg_time = self.ext_time.to_sec()
            _ = self.wait_for_new_msg("act_time", old_msg_time)

            old_msg_time = self.act_time.to_sec()
            send_torque = self.act_torque

        RUN_FAKE_IMU = False


if __name__ == "__main__":
    import rostest
    rostest.rosrun(PKG, 'integration_test_MagnetorquerNode', MagnetorquerNodeTest)
