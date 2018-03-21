# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

from datetime import datetime


class EpochClock(object):
    """
    Class handeling simulation time in ROS.

    Stores starting/ending epoch, updates time factors and creates clock
    messages with the updated simulation time

    Args:
        oe_epoch: epoch of initial orbital elements
        frequency: publish frequency of ROS nodes [1/s]
        step_size: simulation step size [s]
    """
    def __init__(self,
                 oe_epoch=None,
                 frequency=None,
                 step_size=None):

        self.epoch = ""
        self.oe_epoch = ""
        self.epoch_end = ""

        if oe_epoch is not None:
            self.oe_epoch = oe_epoch
            self.datetime_oe_epoch = datetime.strptime(self.oe_epoch,
                                                       "%Y%m%dT%H:%M:%S")
        else:
            self.datetime_oe_epoch = datetime.utcnow()

        # real time update rate -- attempted updates per second
        if frequency is not None:
            self.frequency = frequency
        else:
            self.frequency = 10

        self.rate = float(1) / float(self.frequency)

        # maximal step size, run at real time if not defined
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = self.rate

        self.step_size = self.step_size*1e9  # step size in [ns]
        self.step_size = int(self.step_size)

        self.currentTime = int(0)  # time in [ns]
        self.realtime_factor = self.frequency * self.step_size*1e-9
        self.stopTime = 0.0

    def updateClock(self, msg_cl):
        """
        Update current simulation time and create the published message

        Args:
            msg_cl: ROS message for clock. Time is put in here

        Returns:
            rosgraph_msgs.msg.Clock : clock message
        """

        self.currentTime += self.step_size
        full_seconds = int(self.currentTime*1e-9)
        msg_cl.clock.secs = full_seconds
        msg_cl.clock.nsecs = int(self.currentTime - full_seconds*1e9)
        return msg_cl

    def updateTimeFactors(self, new_rtf, new_freq, new_dt):
        """
        Update Realtime Factor, Frequency, step-size and publishing rate
        based on provided frequency

        Args:
            new_rtf: new realtime factor
            new_freq: new frequency
            new_dt: new time step size
        """

        self.realtime_factor = new_rtf
        self.frequency = new_freq
        self.rate = float(1) / float(self.frequency)
        self.step_size = new_dt
