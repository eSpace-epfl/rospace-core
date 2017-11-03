#!/usr/bin/env python

import rospy
import numpy as np
from BaseRelativeOrbitalFilter import BaseRelativeOrbitalFilter
from space_msgs.msg import *
from geometry_msgs.msg import PointStamped

filter = BaseRelativeOrbitalFilter()
import space_tf as stf
import tf


if __name__ == '__main__':
    rospy.init_node("cso_gnc_target_estimator", anonymous=True)

    pub = rospy.Publisher("state", RelOrbElemWithCovarianceStamped, queue_size=10)
    pub_res = rospy.Publisher("residual", PointStamped, queue_size=10)
    pub_point = rospy.Publisher("state_pos", PointStamped, queue_size=10)

    # get first state before subscribers start
    [x, P] = filter.get_state()
    res = filter.residual

    # sensor subscribers
    rospy.Subscriber("aon", AzimutElevationStamped, filter.callback_aon)

    # own state subscriber
    rospy.Subscriber("oe", SatelitePose, filter.callback_state)

    # filter publisher
    r = rospy.Rate(10)
    br = tf.TransformBroadcaster()

    while not rospy.is_shutdown():

        # publish residual
        msg_res = PointStamped()
        msg_res.header.stamp = rospy.Time.now()
        msg_res.point.x = res[0]
        msg_res.point.y = res[1]
        pub_res.publish(msg_res)

        #publish as relOrbit with covariance
        msg = RelOrbElemWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.relorbit.relorbit.dA = x[0]
        msg.relorbit.relorbit.dL = x[1]
        msg.relorbit.relorbit.dEx = x[2]
        msg.relorbit.relorbit.dEy = x[3]
        msg.relorbit.relorbit.dIx = x[4]
        msg.relorbit.relorbit.dIy = x[5]
        # flatten matrix in row-major order ("style C")
        msg.relorbit.covariance = P.flatten("C")
        pub.publish(msg)



        if filter.has_oe:
            # publish absolute target position as seen by filter
            target_oe = filter.get_target_oe()
            # convert to cartesian
            cart_c = stf.Cartesian()
            cart_c.from_keporb(target_oe)



            br.sendTransform(cart_c.R*1000,
                            np.array([0, 0, 0, 1]),
                            filter.t,
                             "filter_state",
                             "teme")

        [x, P] = filter.get_state()
        res = filter.residual

        r.sleep()

