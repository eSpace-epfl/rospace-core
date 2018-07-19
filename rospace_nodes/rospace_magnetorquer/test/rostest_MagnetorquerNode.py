#!/usr/bin/env python
# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

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
        a_sub (rospy.Subscriber): subscriber to the "actuator_torque" topic.
        b_sub (rospy.Subscriber): subscriber to the "B_field" topic.
        c_sub (rospy.Subscriber): subscriber to the "I_magneto" topic.
        d_sub (rospy.Subscriber): subscriber to the "dist_torque" topic.
        spin_pub (rospy.Publisher): fake IMU publisher

    """

    def setUp(self):
        """Sets up every unit test.

        Every Subscriber has its own callback in which the last published messages as well as its time stamp
        is being stored.

        """
        rospy.init_node("test_Magnetorquer")

        spacecraft = rospy.get_param("~spacecraft")
        self.timestep = rospy.get_param("~timestep")

        self.a_sub = rospy.Subscriber("/" + spacecraft + "/actuator_torque", WrenchStamped, self.actuator_callback)
        self.b_sub = rospy.Subscriber("/" + spacecraft + "/B_field", Vector3Stamped, self.b_field_callback)
        self.c_sub = rospy.Subscriber("/" + spacecraft + "/I_magneto", Vector3Stamped, self.controller_callback)
        self.d_sub = rospy.Subscriber("/" + spacecraft + "/dist_torque", SatelliteTorque, self.dist_torque_callback)

        self.spin_pub = rospy.Publisher("/" + spacecraft + "/imu", Imu, queue_size=10)

        # Wait until PropagatorNode finishes initialization
        while not hasattr(self, "b_field_time"):
            rospy.sleep(0.001)

        rospy.sleep(0.2)  # sleep so all messages really updated!

    def b_field_callback(self, b_field_msg):
        """B-field subscriber callback storing last published message.

        In case the global variable "RUN_FAKE_IMU" is set to True, this method
        also publishes a messages to the imu topic with random spin.

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

    def actuator_callback(self, act_msg):
        """Actuator subscriber callback storing last published message.

        Args:
            act_msg (geometry_msgs.msg.WrenchStamped): published actuator message

        """
        self.act_time = act_msg.header.stamp

        self.act_torque = np.zeros(3)
        self.act_torque[0] = act_msg.wrench.torque.x
        self.act_torque[1] = act_msg.wrench.torque.y
        self.act_torque[2] = act_msg.wrench.torque.z

    def dist_torque_callback(self, dist_msg):
        """Induced disturbance torques subscriber callback storing last published message.

        Args:
            dist_msg (rospace_msgs.msg.SatelliteTorque): published message holding induced disturbance torques

        """
        self.ext_time = dist_msg.header.stamp

        self.ext_torque = np.zeros(3)
        self.ext_torque[0] = dist_msg.external.torque.x
        self.ext_torque[1] = dist_msg.external.torque.y
        self.ext_torque[2] = dist_msg.external.torque.z

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

    # Unit-tests ###################################################################
    def test_0_no_spin_msg(self):
        """Test that no current is being published if no spin from IMU provided.

        This test has to be run before publishing any message to the imu topic!

        """
        for _ in range(0, CHECK_MSGS):
            if hasattr(self, "ctr_time"):  # some message was received..
                raise RuntimeError("Control attributes are published, even when no information about spin provided!")
            rospy.sleep(0.05)

    def test_0_no_current_no_torque(self):
        """Test that torque equals zero if no current published.

        This test has to be run before publishing any message to the imu topic!

        """
        old_msg_time = 0

        for _ in range(0, CHECK_MSGS):
            old_msg_time = self.wait_for_new_msg("act_time", old_msg_time)

            self.assertEqual(self.act_torque[0], 0)
            self.assertEqual(self.act_torque[1], 0)
            self.assertEqual(self.act_torque[2], 0)

    def test_magnetorquer_publishing(self):
        """Test that messages from B_field topic are received by checking that actuator messages are being published."""
        old_msg_time = 0

        for _ in range(0, CHECK_MSGS):
            _ = self.wait_for_new_msg("b_field_time", old_msg_time)
            old_msg_time = self.act_time.to_sec()

    def test_subscription_to_current(self):
        """Test that current messages are received on correct topic."""
        global RUN_FAKE_IMU
        RUN_FAKE_IMU = True
        rospy.sleep(0.2)

        old_msg_time = 0

        for _ in range(0, CHECK_MSGS):
            old_msg_time = self.wait_for_new_msg("ctr_time", old_msg_time)

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
            _ = self.wait_for_new_msg("ext_time", old_msg_time)

            # TODO: check if this works.. theoretically much quicker than frequency of nodes, but still....
            # yup it does

            self.assertEqual(send_torque[0], self.ext_torque[0])
            self.assertEqual(send_torque[1], self.ext_torque[1])
            self.assertEqual(send_torque[2], self.ext_torque[2])

            # wait for new message from actuator
            _ = self.wait_for_new_msg("act_time", old_msg_time)

            old_msg_time = self.act_time.to_sec()
            send_torque = self.act_torque

        RUN_FAKE_IMU = False


if __name__ == "__main__":
    import rostest
    rostest.rosrun(PKG, 'integration_test_MagnetorquerNode', MagnetorquerNodeTest)
