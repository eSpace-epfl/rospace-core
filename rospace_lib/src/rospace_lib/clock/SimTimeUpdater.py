# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

from datetime import datetime, timedelta


class SimTimeUpdater(object):
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
                 step_size=None,
                 time_shift=0.0):

        try:
            assert(time_shift >= 0.0)
        except AssertionError:
            raise AssertionError("Time shift cannot be negative!")

        if oe_epoch is not None:
            self.datetime_oe_epoch = datetime.strptime(oe_epoch, "%Y%m%dT%H:%M:%S")
            self.datetime_oe_epoch_shifted = \
                self.datetime_oe_epoch - timedelta(seconds=time_shift)
        else:
            self.datetime_oe_epoch = datetime.utcnow()
            self.datetime_oe_epoch_shifted = \
                self.datetime_oe_epoch - timedelta(seconds=time_shift)

        self.time_shift = time_shift * 1e9
        self.time_shift_passed = time_shift == 0.0

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
        if (self.currentTime > self.time_shift or
           self.currentTime < self.time_shift):
            # before starting time or after
            self.currentTime += self.step_size
        elif (self.currentTime < self.time_shift and
              self.currentTime + self.step_size > self.time_shift):
            # timestep needs to be adjusted to start at correct epoch
            shift = int(self.currentTime + self.step_size - self.time_shift)
            self.currentTime = self.time_shift
            mesg = "\033[93m[WARN] [EpochClock] Shortend next timestep by " \
                   + str(shift * 1e-9) \
                   + "s to reach desired intial epoch. \033[0m"
            print mesg
            # set that time shift has passed to true
            self.time_shift_passed = True
        else:
            # time hit exactly correct starting epoch
            self.currentTime += self.step_size
            self.time_shift_passed = True

        full_seconds = int(self.currentTime*1e-9)
        msg_cl.clock.secs = full_seconds
        msg_cl.clock.nsecs = int(self.currentTime - full_seconds*1e9)

        time_delta = timedelta(0, msg_cl.clock.secs, msg_cl.clock.nsecs / 1e3)

        return [msg_cl, self.datetime_oe_epoch_shifted + time_delta]

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
        if new_freq > 0:
            self.rate = float(1) / float(self.frequency)
        else:  # frequency = None : run as quickly as possible
            self.rate = 0
        self.step_size = new_dt
