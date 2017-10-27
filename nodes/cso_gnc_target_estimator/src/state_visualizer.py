#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import rospy
from space_msgs.msg import *

def plot(msg):
    global axarr
    global init
    global line1,line2

    stamp = msg.header.stamp
    time = stamp.secs + stamp.nsecs * 1e-9

    if not init:
        f, axarr = plt.subplots(2, sharex=True)
        init = True

        line1, = axarr[0].plot(msg.relorbit.relorbit.dA, time)
        line2, = axarr[1].plot(msg.relorbit.relorbit.dL, time)
    else:
        line1.set_ydata(np.append(line1.get_ydata(),msg.relorbit.relorbit.dA))
        line1.set_xdata(np.append(line1.get_xdata(), time))

        line2.set_ydata(np.append(line2.get_ydata(), msg.relorbit.relorbit.dL))
        line2.set_xdata(np.append(line2.get_xdata(), time))
        print time
    plt.set_xlim([0,time])
    plt.draw()
    plt.pause(0.00000000001)

if __name__ == '__main__':
    init = False
    rospy.init_node("plotter")
    rospy.Subscriber("state", RelOrbElemWithCovarianceStamped, plot)
    plt.ion()
    plt.show()
    rospy.spin()