#!/usr/bin/env python

from __future__ import division

import os

from python_qt_binding import loadUi
from python_qt_binding.QtCore import Slot, qWarning
from python_qt_binding.QtGui import QIcon
from python_qt_binding.QtWidgets import QWidget

import rospkg
import rospy
import rosservice

from rqt_py_common.extended_combo_box import ExtendedComboBox


class ServiceCallerWidget(QWidget):

    def __init__(self):
        super(ServiceCallerWidget, self).__init__()
        self.setObjectName('ServiceCallerWidget')

        # subsribe to service for gui
        rospy.wait_for_service('/start_stop_clock')
        self._services = rosservice.get_service_class_by_name(
                                                        "/start_stop_clock")
        self._service_info = {}
        self._service_info['service_name'] = "/start_stop_clock"
        self._service_info['service_class'] = self._services
        self._service_info['service_proxy'] = rospy.ServiceProxy(
                                                "/start_stop_clock",
                                                self._service_info['service_class'],
                                                persistent=True)
        rp = rospkg.RosPack()
        ui_file = os.path.join(rp.get_path('rqt_simtime_plugin'),
                               'resource',
                               'rqt_simtime.ui')

        loadUi(ui_file, self, {'ExtendedComboBox': ExtendedComboBox})
        self.call_service_button.setIcon(QIcon.fromTheme(
                                                    'media-playback-start'))
        # self.RTF_val_box.valueChanged.connect(self.changeRTF)

        # populate boxes for the first time
        self.dtSize_val = None
        self.PubFreq_val = None
        self.init_box_values()

        # self.dtSize_val = self.dtSize_val_box.value()
        self.dtSize_val_box.editingFinished.connect(self.change_dt)

        # self.PubFreq_val = self.PubFreq_val_box.value()
        self.PubFreq_val_box.editingFinished.connect(self.change_dt)

        # self._service_info['expressions'] = {}
        # self._service_info['counter'] = 0

    def init_box_values(self):
        request = self._service_info['service_class']._request_class()
        request.trigger = False
        request.dt_size = -1.0
        request.p_freq = -1.0
        response = self.call_service_node(request)

        self.PubFreq_val = response.cur_pfreq
        self.dtSize_val = response.cur_dt*1e-9
        self.PubFreq_val_box.setValue(self.PubFreq_val)
        self.dtSize_val_box.setValue(self.dtSize_val)
        self.RTF_val_box.setValue(self.PubFreq_val*self.dtSize_val)

    @Slot()
    def on_increase_dt_clicked(self):
        if self.dtSize_val < 1:
            val = self.dtSize_val_box.value() + 0.1
        elif self.dtSize_val < 10:  # between 1 and 9.9
            if self.dtSize_val % 2 == 0:
                val = self.dtSize_val_box.value() + 2.0
            else:
                val = self.dtSize_val_box.value() + 1.0
        elif self.dtSize_val < 100:  # between 10 and 99.9
            val = round(self.dtSize_val_box.value() + 10.0, -1)
        elif self.dtSize_val < 500:
            val = round(self.dtSize_val_box.value() + 50.0, -1)
        else:
            val = round(self.dtSize_val_box.value() + 100.0, +1)

        if val > 10000.0:  # limit maximal timestep
            val = 10000.0

        self.dtSize_val_box.setValue(val)
        self.change_dt()  # don't stop simulation

    @Slot()
    def on_decrease_dt_clicked(self):
        if self.dtSize_val <= 1.0:
            val = self.dtSize_val_box.value() - 0.1

        elif self.dtSize_val <= 10.0:  # between 1 and 9.9
            if self.dtSize_val % 2 == 0:
                val = self.dtSize_val_box.value() - 2.0
            else:
                val = self.dtSize_val_box.value() - 1.0
            if val < 1.0:
                val = 1.0

        elif self.dtSize_val <= 100.0:  # between 10 and 99.9
            val = self.dtSize_val_box.value() - 10.0

        elif self.dtSize_val <= 500.0:
            val = self.dtSize_val_box.value() - 50.0

        else:
            val = self.dtSize_val_box.value() - 100.0

        if val < 0.0001:
            val = 0.0001

        self.dtSize_val_box.setValue(val)
        self.change_dt()  # don't stop simulation

    @Slot()
    def on_increase_pFreq_clicked(self):
        if self.PubFreq_val < 1:
            val = self.PubFreq_val_box.value() + 0.1
        elif self.PubFreq_val < 10:  # between 1 and 9.9
            if self.PubFreq_val % 2 == 0:
                val = self.PubFreq_val_box.value() + 2.0
            else:
                val = self.PubFreq_val_box.value() + 1.0
        elif self.PubFreq_val < 100:  # between 10 and 99.9
            val = round(self.PubFreq_val_box.value() + 5.0, -1)
        else:
            val = round(self.PubFreq_val_box.value() + 50.0, -1)

        if val > 500.0:
            val = 500.0

        self.PubFreq_val_box.setValue(val)
        self.change_dt()  # don't stop simulation

    @Slot()
    def on_decrease_pFreq_clicked(self):
        if self.PubFreq_val <= 1.0:
            val = self.PubFreq_val_box.value() - 0.1

        elif self.PubFreq_val <= 10.0:  # between 1 and 9.9
            if self.PubFreq_val % 2 == 0:
                val = self.PubFreq_val_box.value() - 2.0
            else:
                val = self.PubFreq_val_box.value() - 1.0
            if val < 1.0:
                val = 1.0

        elif self.PubFreq_val <= 100.0:  # between 10 and 99.9
            val = self.PubFreq_val_box.value() - 10.0

        else:
            val = self.PubFreq_val_box.value() - 50.0

        if val < 0.1:
            val = 0.1

        self.PubFreq_val_box.setValue(val)
        self.change_dt()  # don't stop simulation

    @Slot()
    def on_call_service_button_clicked(self):
        request = self._service_info['service_class']._request_class()
        request.trigger = True
        request.dt_size = -1.0
        request.p_freq = -1.0
        self.call_service_node(request)

    @Slot()
    def change_dt(self):
        if self.dtSize_val_box.value() > 10000.0:
            self.dtSize_val_box.setValue(10000.0)
        if self.dtSize_val_box.value() <= 0.0001:
            self.dtSize_val_box.setValue(0.0001)

        if self.PubFreq_val_box.value() > 500.0:
            self.PubFreq_val_box.setValue(500.0)
        if self.PubFreq_val_box.value() < 0.1:
            self.PubFreq_val_box.setValue(0.1)

        if (self.dtSize_val != self.dtSize_val_box.value() or
                self.PubFreq_val != self.PubFreq_val_box.value()):
            self.dtSize_val = self.dtSize_val_box.value()
            self.PubFreq_val = self.PubFreq_val_box.value()
            request = self._service_info['service_class']._request_class()
            request.trigger = False
            request.dt_size = self.dtSize_val*1e9
            request.p_freq = self.PubFreq_val
            response = self.call_service_node(request)

            self.PubFreq_val = round(response.cur_pfreq, 1)
            self.dtSize_val = round(response.cur_dt*1e-9, 4)
            self.PubFreq_val_box.setValue(self.PubFreq_val)
            self.dtSize_val_box.setValue(self.dtSize_val)
            self.RTF_val_box.setValue(self.PubFreq_val*self.dtSize_val)

    def call_service_node(self, request):
        try:
            response = self._service_info['service_proxy'](request)
        except rospy.ServiceException as e:
            qWarning('ServiceCaller.on_call_service_button_clicked():',
                     ' request:\n%r' % (request))
            qWarning('ServiceCaller.on_call_service_button_clicked(): error',
                     ' calling service "%s":\n%s' %
                     (self._service_info['service_name'], e))

        return response
