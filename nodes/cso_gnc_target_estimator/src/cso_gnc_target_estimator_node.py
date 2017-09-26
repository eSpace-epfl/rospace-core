#!/usr/bin/env python

import rospy
from BaseRelativeOrbitalFilter import BaseRelativeOrbitalFilter
from space_msgs.msg import *

filter = BaseRelativeOrbitalFilter()


if __name__ == '__main__':
    rospy.init_node("cso_gnc_target_estimator", anonymous=True)

    # sensor subscribers
    rospy.Subscriber("aon", AzimutElevationStamped, filter.callback_aon)

    # own state subscriber
    rospy.Subscriber("oe", SatelitePose, filter.callback_state)

    # filter publisher
    r = rospy.Rate(1)

    while not rospy.is_shutdown():
        state = filter.get_state()
