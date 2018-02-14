#!/usr/bin/env python

import rospy
import numpy as np
from UKFRelativeOrbitalFilter import UKFRelativeOrbitalFilter
from space_msgs.msg import *
from geometry_msgs.msg import PointStamped

filter = None
import space_tf as stf
import tf

import message_filters

if __name__ == '__main__':
    rospy.init_node("cso_gnc_target_estimator", anonymous=True)


    P_init = np.array(rospy.get_param("~P")).reshape((6, 6), order='C')



    R_init = np.diag(np.array(rospy.get_param("~R")).astype(np.float))

    Q_init = np.diag(np.diag(P_init))/1000

    #Q_init = np.diag(np.array(rospy.get_param("~Q")).astype(np.float))

    print R_init
    x_dict = rospy.get_param("~x")
    roe_init = stf.QNSRelOrbElements()
    roe_init.dA = float(x_dict["dA"])
    roe_init.dL = float(x_dict["dL"])
    roe_init.dIx = float(x_dict["dIx"])
    roe_init.dIy = float(x_dict["dIy"])
    roe_init.dEx = float(x_dict["dEx"])
    roe_init.dEy = float(x_dict["dEy"])

    x_init = roe_init.as_vector().reshape(6)

    filter = UKFRelativeOrbitalFilter(x=x_init,
                                      P=P_init,
                                      R=R_init,
                                      Q=Q_init)

    pub = rospy.Publisher("state", RelOrbElemWithCovarianceStamped, queue_size=10)
    pub_tar_oe = rospy.Publisher("state_oe", SatelitePose, queue_size=10)
    # get first state before subscribers start
    [t_ukf, x, P] = filter.get_state()

    # sensor subscribers
    #rospy.Subscriber("aon", AzimutElevationStamped, filter.callback_aon)

    # own state subscriber
    #rospy.Subscriber("oe", SatelitePose, filter.callback_state)

    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)
    aon_sub = message_filters.Subscriber('aon', AzimutElevationStamped)
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub, aon_sub], 10)
    ts.registerCallback(filter.callback_aon)
    # filter publisher
    r = rospy.Rate(10)

    while not rospy.is_shutdown():


        [t_ukf, x, P] = filter.get_state()

        msg = RelOrbElemWithCovarianceStamped()
        msg.header.stamp = rospy.Time.from_seconds(t_ukf)
        msg.relorbit.relorbit.dA = x[0]
        msg.relorbit.relorbit.dL = x[1]
        msg.relorbit.relorbit.dEx = x[2]
        msg.relorbit.relorbit.dEy = x[3]
        msg.relorbit.relorbit.dIx = x[4]
        msg.relorbit.relorbit.dIy = x[5]
        # flatten matrix in row-major order ("style C")
        msg.relorbit.covariance = P.flatten("C")
        pub.publish(msg)

        r.sleep()

