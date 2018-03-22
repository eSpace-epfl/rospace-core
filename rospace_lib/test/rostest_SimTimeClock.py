#!/usr/bin/env python
import unittest
import sys
import os
import rospy

from rospace_lib.clock import *
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...


class SimTimeClockTest(unittest.TestCase):

    def setUp(self):
        SimTimeClock._sim_time_setup_requested = False

    def test_multiple_calls_to_sim_time_set_up(self):
        rospy.init_node('test_node', anonymous=True)
        SimClock = SimTimeClock()

        SimClock.set_up_simulation_time()
        with self.assertRaises(RuntimeError):
            SimClock.set_up_simulation_time()

    def test_missing_sleep_at_end(self):
        rospy.init_node('test_node', anonymous=True)
        SimClock = SimTimeClock()

        SimClock.set_up_simulation_time()

        SimClock.update_simulation_time()

        with self.assertRaises(RuntimeError):
            SimClock.update_simulation_time()


if __name__ == '__main__':
    import rostest
    rostest.rosrun('epoch_clock', 'test_SimTimeClock', SimTimeClockTest)
