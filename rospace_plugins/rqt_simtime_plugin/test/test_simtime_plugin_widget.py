#!/usr/bin/env python
import unittest
import sys
import os
import rospy

from PyQt5.QtWidgets import QApplication

from rqt_simtime_plugin import *
from rospace_lib.clock import *


sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../src")  # hack...


class SimtimePluginTest(unittest.TestCase):
    _SimClock = None

    results = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
               2.0, 4.0, 6.0, 8.0, 10.0,
               20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
               150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0,
               600.0, 700.0]

    @classmethod
    def setUpClass(cls):
        rospy.init_node('test_node_simtime_plugin', anonymous=True)
        SimtimePluginTest._SimClock = SimTimeClock()
        SimtimePluginTest._SimClock.set_up_simulation_time()

    def test_correct_value_initialization(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        dt_value = widget.dtSize_val_box.value()
        dt_real_value = SimtimePluginTest._SimClock._SimTime.step_size*1e-9
        freq_value = widget.PubFreq_val_box.value()

        self.assertAlmostEquals(dt_real_value, dt_value, places=4)
        self.assertAlmostEquals(SimtimePluginTest._SimClock._SimTime.frequency, freq_value,places=1)

    def test_correct_change_in_values(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(12.1)
        widget.dtSize_val_box.setValue(123.321)
        widget.change_dt()

        SimtimePluginTest._SimClock.update_simulation_time()
        SimtimePluginTest._SimClock._comp_time = None

        # check if internal parameters updated:
        self.assertAlmostEquals(widget.dtSize_val, widget.dtSize_val_box.value(), places=4)
        self.assertAlmostEquals(widget.PubFreq_val, widget.PubFreq_val_box.value(), places=1)

        dt_value = widget.dtSize_val_box.value()
        dt_real_value = SimtimePluginTest._SimClock._SimTime.step_size*1e-9

        freq_value = widget.PubFreq_val_box.value()
        freq_real_value = SimtimePluginTest._SimClock._SimTime.frequency

        self.assertAlmostEquals(dt_value, dt_real_value, places=4)
        self.assertAlmostEquals(freq_value, freq_real_value, places=1)

    def test_increase_dt(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(1.0)
        widget.dtSize_val_box.setValue(0.1)
        widget.change_dt()

        for i in SimtimePluginTest.results:
            widget.on_increase_dt_clicked()
            self.assertEquals(widget.dtSize_val_box.value(), i)

    def test_increase_and_round_dt(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(1.0)

        widget.dtSize_val_box.setValue(20.22)
        widget.change_dt()

        widget.on_increase_dt_clicked()
        self.assertEquals(widget.dtSize_val_box.value(), 30.0)

        widget.dtSize_val_box.setValue(110.78)
        widget.change_dt()

        widget.on_increase_dt_clicked()
        self.assertEquals(widget.dtSize_val_box.value(), 160.0)

    def test_maximal_dt(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(1.0)

        widget.dtSize_val_box.setValue(11000.0)
        widget.change_dt()
        self.assertEquals(widget.dtSize_val_box.value(), 10000.0)

        widget.on_increase_dt_clicked()
        self.assertEquals(widget.dtSize_val_box.value(), 10000.0)

    def test_decrease_dt(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(1.0)
        widget.dtSize_val_box.setValue(800.0)
        widget.change_dt()

        for i in reversed(SimtimePluginTest.results):
            widget.on_decrease_dt_clicked()
            self.assertEquals(widget.dtSize_val_box.value(), i)

        # go up to 0.0001
        widget.on_decrease_dt_clicked()
        widget.on_decrease_dt_clicked()
        self.assertEquals(widget.dtSize_val_box.value(), 0.0001)

    def test_minimal_dt(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(1.0)

        widget.dtSize_val_box.setValue(0.000001)
        widget.change_dt()
        self.assertEquals(widget.dtSize_val_box.value(), 0.0001)

        widget.on_decrease_dt_clicked()
        self.assertEquals(widget.dtSize_val_box.value(), 0.0001)

    def test_increase_freq(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(0.1)
        widget.change_dt()

        results_freq = SimtimePluginTest.results[:-2]
        for i in results_freq:
            widget.on_increase_pFreq_clicked()
            self.assertEquals(widget.PubFreq_val_box.value(), i)

    def test_maximal_freq(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.dtSize_val_box.setValue(1.0)

        widget.PubFreq_val_box.setValue(600.0)
        widget.change_dt()

        self.assertEquals(widget.PubFreq_val_box.value(), 500.0)

        widget.on_increase_pFreq_clicked()
        self.assertEquals(widget.PubFreq_val_box.value(), 500.0)

    def test_decrease_freq(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.PubFreq_val_box.setValue(500.0)
        widget.change_dt()

        results_freq = SimtimePluginTest.results[:-3]
        for i in reversed(results_freq):
            widget.on_decrease_pFreq_clicked()
            self.assertEquals(widget.PubFreq_val_box.value(), i)

    def test_minimal_freq(self):
        app = QApplication(sys.argv)
        widget = ServiceCallerWidget()

        widget.dtSize_val_box.setValue(1.0)

        widget.PubFreq_val_box.setValue(-0.001)
        widget.change_dt()

        self.assertEquals(widget.PubFreq_val_box.value(), 0.0)

        widget.on_decrease_pFreq_clicked()
        self.assertEquals(widget.PubFreq_val_box.value(), 0.0)


if __name__ == '__main__':
    import rostest
    rostest.rosrun('rqt_simtime_plugin', 'test_SimtimePlugin', SimtimePluginTest)
