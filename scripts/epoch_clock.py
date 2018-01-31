#!/usr/bin/env python
#
# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""
Warning: This code becomes obsolete very soon - don't change/refactor/update anymore.

(mp, 23.12.17)
"""
import rospy
import random
import threading
from Tkinter import Tk, Label, Button
from datetime import datetime
from rosgraph_msgs.msg import Clock
from time import sleep


class EpochClockGUI(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()
        self.paused = False

    def callback_quit(self):
        self.root.quit()

    def update_current_time(self, epoch, elapsed, curr_time):
        if hasattr(self, 'time_label') and \
                hasattr(self, "epoch_label") and \
                hasattr(self, "elapsed_time_label"):
            self.epoch_label.config(text="Epoch0: " + str(epoch))
            self.elapsed_time_label.config(text="Elapsed: " + str(elapsed))
            self.time_label.config(text="Epoch: " + str(curr_time))

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Resume")
            rospy.logwarn("Clock paused")
        else:
            self.pause_button.config(text="Pause")
            rospy.logwarn("Clock resumed")

    def run(self):
        self.root = Tk()
        self.root.title("Epoch Clock")
        self.epoch_label = Label(self.root, text="")
        self.epoch_label.pack()
        self.elapsed_time_label = Label(self.root, text="")
        self.elapsed_time_label.pack()
        self.time_label = Label(self.root, text="")
        self.time_label.pack()
        self.pause_button = Button(self.root, text="Pause", command=self.toggle_pause)
        self.pause_button.pack()
        self.root.wm_attributes("-topmost", 1)
        self.root.mainloop()


if __name__ == '__main__':
    rospy.init_node('epoch_clock', anonymous=True)
    rospy.loginfo("Starting Epoch Clock with settings:")

    epoch = ""
    datetime_epoch = []
    realtime_factor = []
    frequency = []
    paused = False

    if rospy.has_param("~init_epoch"):
        epoch = str(rospy.get_param("~init_epoch"))
        datetime_epoch = datetime.strptime(epoch, "%Y%m%dT%H:%M:%S")
    else:
        datetime_epoch = datetime.utcnow()

    if rospy.has_param("~realtime_factor"):
        realtime_factor = int(rospy.get_param("~realtime_factor"))
    else:
        realtime_factor = 1

    if rospy.has_param("~frequency"):
        frequency = int(rospy.get_param("~frequency"))
    else:
        frequency = 50.0

    rate = float(1) / float(frequency)

    rospy.loginfo("Epoch = " + datetime_epoch.strftime("%Y-%m-%d %H:%M:%S"))
    rospy.loginfo("Realtime Factor = " + str(realtime_factor))

    datetime_startup = datetime.utcnow()  # this is considered time 0

    rospy.set_param('/epoch', datetime_epoch.strftime("%Y-%m-%d %H:%M:%S"))
    rospy.set_param('use_sim_time', True)

    # Init publisher and rate limiter
    pub = rospy.Publisher('clock', Clock, queue_size=10)

    gui = EpochClockGUI()
    pause_start = None

    # publish clock message according to realtime factor
    while not rospy.is_shutdown():
        if not gui.paused:

            # check if we are returning from a pause
            if pause_start is not None:
                # correct startup time by pause time
                datetime_startup = datetime_startup + (datetime.utcnow() - pause_start)
                pause_start = None

            elapsed_time = (datetime.utcnow() - datetime_startup)
            sim_elapsed_time = (elapsed_time * realtime_factor)

            msg = Clock()
            msg.clock.secs = int(sim_elapsed_time.seconds + sim_elapsed_time.days * (24 * 3600))
            msg.clock.nsecs = sim_elapsed_time.microseconds * 1e3
            pub.publish(msg)

            gui.update_current_time(datetime_epoch,
                                    sim_elapsed_time,
                                    datetime_epoch + sim_elapsed_time)
        if gui.paused:
            # if we just got into a pause, store time
            if pause_start is None:
                pause_start = datetime.utcnow()

        sleep(rate)

    gui.callback_quit()
