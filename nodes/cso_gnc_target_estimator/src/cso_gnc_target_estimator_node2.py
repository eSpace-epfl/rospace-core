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
    pub_tar_oe = rospy.Publisher("state_oe", SatelitePose, queue_size=10)
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
        print "RES :", res[0], res[1]
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

            target_osc_oe = stf.OscKepOrbElem()
            target_osc_oe.from_mean_elems(target_oe)
            # convert to cartesian
            cart_c_osc = stf.Cartesian()
            cart_c_osc.from_keporb(target_osc_oe)

            cart_c = stf.Cartesian()
            cart_c.from_keporb(target_oe)

            print "OSC=",cart_c_osc.R*1000

            br.sendTransform(cart_c_osc.R*1000,
                            np.array([0, 0, 0, 1]),
                            filter.t,
                             "filter_state_osc",
                             "teme")

            br.sendTransform(cart_c.R * 1000,
                             np.array([0, 0, 0, 1]),
                             filter.t,
                             "filter_state_mean",
                             "teme")

            msg_tar = SatelitePose()

            msg_tar.header.stamp = filter.t
            msg_tar.header.frame_id = "teme"

            msg_tar.position.semimajoraxis = target_oe.a
            msg_tar.position.eccentricity = target_oe.e
            msg_tar.position.inclination = np.rad2deg(target_oe.i)
            msg_tar.position.arg_perigee = np.rad2deg(target_oe.w)
            msg_tar.position.raan = np.rad2deg(target_oe.O)
            msg_tar.position.true_anomaly = np.rad2deg(target_oe.v)
            pub_tar_oe.publish(msg_tar)

        [x, P] = filter.get_state()
        res = filter.residual

        r.sleep()

