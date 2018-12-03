#!/usr/bin/env python
import unittest
import sys
import os
import rospy
from mock import patch, Mock
from rospace_lib.clock import *

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...


class SimTimePublisherTest(unittest.TestCase):

    def setUp(self):
        """Set up for tests. Revert request for time manually."""
        SimTimePublisher._sim_time_setup_requested = False

    @patch("rospace_lib.clock.SimTimeService.__init__")
    def test_multiple_calls_to_sim_time_set_up(self, SimTimeService_mock):
        """Test that only one node can set up the simulation time.

        Error should be raised if multiple times set up is called. As no GUI
        is used the Service which links to the GUI is mocked. Here only the
        constructor of the service class is used and therefore that one is patched
        so no error is thrown.
        """
        rospy.init_node('test_node', anonymous=True)
        SimTimeService_mock.return_value = None

        simtime_pub = SimTimePublisher()

        simtime_pub.set_up_simulation_time()
        with self.assertRaises(RuntimeError):
            simtime_pub.set_up_simulation_time()

    @patch("rospace_lib.clock.SimTimeService.__init__")
    def test_missing_sleep_at_end(self, STS_init_mock):
        """Test that error raised if no sleep at end of simulation timestep called.

        As no GUI is used the Service which links to the GUI is mocked. As during a time update
        the step size and realtime factor are compared these are mocked and set to the same value
        as on initialization so that the frequency and stepsize is not being changed in the epoch class.
        """
        rospy.init_node('test_node', anonymous=True)

        # mock __init__ of SimTimeService
        STS_init_mock.return_value = None

        simtime_pub = SimTimePublisher()
        simtime_pub.set_up_simulation_time()

        # mock ClockService instance variables
        mck = Mock(realtime_factor=simtime_pub._SimTime.realtime_factor,
                   step_size=simtime_pub._SimTime.step_size)
        simtime_pub.ClockService = mck

        simtime_pub.update_simulation_time()

        with self.assertRaises(RuntimeError):
            simtime_pub.update_simulation_time()


if __name__ == '__main__':
    import rostest
    rostest.rosrun('epoch_clock', 'test_SimTimeClock', SimTimePublisherTest)
