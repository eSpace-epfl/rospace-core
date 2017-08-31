#!/usr/bin/env python
import rospy
import random

from datetime import datetime
from rosgraph_msgs.msg import Clock
from time import sleep



if __name__ == '__main__':
    rospy.init_node('epoch_clock', anonymous=True)
    rospy.loginfo("Starting Epoch Clock with settings:")

    epoch =""
    datetime_epoch = []
    realtime_factor = []
    frequency = []

    if rospy.has_param("~epoch"):
        epoch = str(rospy.get_param("~epoch"))
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
        frequency = 20.0

    rate = float(1)/float(frequency)

    rospy.loginfo("Epoch = " + datetime_epoch.strftime("%Y-%m-%d %H:%M:%S"))
    rospy.loginfo("Realtime Factor = " + str(realtime_factor))

    datetime_startup = datetime.utcnow() # this is considered time 0

    rospy.set_param('epoch', datetime_epoch.strftime("%Y-%m-%d %H:%M:%S"))
    rospy.set_param('use_sim_time', True)

    # Init publisher and rate limiter
    pub = rospy.Publisher('clock', Clock, queue_size=10)

    # publish clock message according to realtime factor
    while not rospy.is_shutdown():
        elapsed_time = (datetime.utcnow() - datetime_startup)
        sim_elapsed_time = (elapsed_time * realtime_factor)

        msg = Clock()
        msg.clock.secs = int(sim_elapsed_time.seconds + sim_elapsed_time.days*(24*3600))
        msg.clock.nsecs = sim_elapsed_time.microseconds * 1e3
        pub.publish(msg)
        sleep(rate)
