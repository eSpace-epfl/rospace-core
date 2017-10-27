#!/usr/bin/env python

import rospy
import numpy as np
from BaseRelativeOrbitalFilter import BaseRelativeOrbitalFilter
from space_msgs.msg import *

filter = BaseRelativeOrbitalFilter()


if __name__ == '__main__':
    rospy.init_node("cso_gnc_target_estimator", anonymous=True)

    pub = rospy.Publisher("state", RelOrbElemWithCovarianceStamped, queue_size=10)

    # get first state before subscribers start
    [x, P] = filter.get_state()

    # sensor subscribers
    rospy.Subscriber("aon", AzimutElevationStamped, filter.callback_aon)

    # own state subscriber
    rospy.Subscriber("oe", SatelitePose, filter.callback_state)

    # filter publisher
    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        #publish as relOrbit with covariance
        msg = RelOrbElemWithCovarianceStamped()
        msg.header.stamp =rospy.get_rostime()
        msg.relorbit.relorbit.dA = x[0]
        msg.relorbit.relorbit.dL = x[1]
        msg.relorbit.relorbit.dEx = x[2]
        msg.relorbit.relorbit.dEy = x[3]
        msg.relorbit.relorbit.dIx = x[4]
        msg.relorbit.relorbit.dIy = x[5]

        # flatten matrix in row-major order ("style C")
        msg.relorbit.covariance = P.flatten("C")
        pub.publish(msg)
        [x, P] = filter.get_state()

        r.sleep()

