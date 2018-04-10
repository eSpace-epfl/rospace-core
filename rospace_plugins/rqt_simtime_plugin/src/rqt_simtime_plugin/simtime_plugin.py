#!/usr/bin/env python

# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

from rqt_gui_py.plugin import Plugin
from .simtime_plugin_widget import ServiceCallerWidget


class ServiceCaller(Plugin):

    def __init__(self, context):
        super(ServiceCaller, self).__init__(context)
        self.setObjectName('ServiceCaller')

        self._widget = ServiceCallerWidget()
        if context.serial_number() > 1:
            self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        context.add_widget(self._widget)
