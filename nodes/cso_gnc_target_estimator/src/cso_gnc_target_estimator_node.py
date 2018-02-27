#!/usr/bin/env python
"""
ROS node for relative navigation filtering

    Author: Michael Pantic

"""

import rospy
import numpy as np
from UKFRelativeOrbitalFilter import UKFRelativeOrbitalFilter
from space_msgs.msg import *
from geometry_msgs.msg import PointStamped

filter = None
import space_tf as stf
import scipy
import tf
import message_filters

if __name__ == '__main__':
    rospy.init_node("cso_gnc_target_estimator", anonymous=True)

    # read configuration values
    enable_bias = bool(rospy.get_param("~enable_bias"))
    enable_emp = bool(rospy.get_param("~enable_emp"))

    P_roe = np.array(rospy.get_param("~P")).astype(np.float).reshape((6, 6), order='C')

    if enable_bias:
        P_bias = np.diag(np.array(rospy.get_param("~P_bias")).astype(np.float))
    else:
        P_bias = np.array((0,0))

    if enable_emp:
        P_emp = np.diag(np.array(rospy.get_param("~P_emp")).astype(np.float))
    else:
        P_emp = np.array((0,0))

    # build covariance matrix P based on enabled options
    if enable_emp and enable_bias:
        P_init = scipy.linalg.block_diag(P_roe, P_bias, P_emp)
    elif enable_emp and not enable_bias:
        P_init = scipy.linalg.block_diag(P_roe, P_emp)
    elif not enable_emp and enable_bias:
        P_init = scipy.linalg.block_diag(P_roe, P_bias)
    else:
        P_init = P_roe

    # set up R and Q
    R_init = np.diag(np.array(rospy.get_param("~R")).astype(np.float))
    Q_init = np.diag(np.array(rospy.get_param("~Q")).astype(np.float))

    # defines the used mean-to-osculating tranformation
    mode = rospy.get_param("~mode")

    # Load initial x_roe state
    x_dict = rospy.get_param("~x")
    roe_init = stf.QNSRelOrbElements()
    roe_init.dA = float(x_dict["dA"])
    roe_init.dL = float(x_dict["dL"])
    roe_init.dIx = float(x_dict["dIx"])
    roe_init.dIy = float(x_dict["dIy"])
    roe_init.dEx = float(x_dict["dEx"])
    roe_init.dEy = float(x_dict["dEy"])

    x_init = np.zeros(P_init.shape[0])
    x_init[0:6] = roe_init.as_vector().reshape(6)

    filter = UKFRelativeOrbitalFilter(x=x_init,
                                      P=P_init,
                                      R=R_init,
                                      Q=Q_init,
                                      enable_bias= enable_bias,
                                      enable_emp=enable_emp,
                                      mode=mode,
                                      output_debug=True)

    [t_ukf, x, P] = filter.get_state()

    # set up combined target/chaser/aon subscription such that they are received synchronized
    # Note: Target state subscription is ONLY used for evaluation and debug!!
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)
    aon_sub = message_filters.Subscriber('aon', AzimutElevationRangeStamped)

    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub, aon_sub], 10)
    ts.registerCallback(filter.callback_aon)

    # set nominal publish rate
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
        #pub.publish(msg)

        r.sleep()

