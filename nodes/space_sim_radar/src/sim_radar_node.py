#!/usr/bin/env python
import rospy
import message_filters
from space_msgs.msg import OrbitalElementsStamped, RangeStamped
import space_tf
from numpy import deg2rad
import numpy as np

pub = rospy.Publisher('radar', RangeStamped, queue_size=10)

def callback(target_oe, chaser_oe):

    # calculate baseline
    tf_target_oe = space_tf.OrbitalElements()
    tf_chaser_oe = space_tf.OrbitalElements()

    tf_target_oe.i = deg2rad(target_oe.orbit.inclination)
    tf_target_oe.w = deg2rad(target_oe.orbit.arg_perigee)
    tf_target_oe.omega = deg2rad(target_oe.orbit.raan)
    tf_target_oe.t = deg2rad(target_oe.orbit.true_anomaly)
    tf_target_oe.a = target_oe.orbit.semimajoraxis
    tf_target_oe.e = target_oe.orbit.eccentricity

    tf_chaser_oe.i = deg2rad(chaser_oe.orbit.inclination)
    tf_chaser_oe.w = deg2rad(chaser_oe.orbit.arg_perigee)
    tf_chaser_oe.omega = deg2rad(chaser_oe.orbit.raan)
    tf_chaser_oe.t = deg2rad(chaser_oe.orbit.true_anomaly)
    tf_chaser_oe.a = chaser_oe.orbit.semimajoraxis
    tf_chaser_oe.e = chaser_oe.orbit.eccentricity

    # convert to TEME
    tf_target_teme = space_tf.CartesianTEME()
    tf_chaser_teme = space_tf.CartesianTEME()
    space_tf.Converter.convert(tf_target_oe, tf_target_teme)
    space_tf.Converter.convert(tf_chaser_oe, tf_chaser_teme)

    msg = RangeStamped()
    msg.header.stamp = rospy.Time.now()
    msg.value.range = np.linalg.norm(tf_target_teme.R -tf_chaser_teme.R)
    pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('RADAR SIM', anonymous=True)
    rospy.loginfo("Radar sim started")
    target_oe_sub = message_filters.Subscriber('target_oe', OrbitalElementsStamped)
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', OrbitalElementsStamped)

    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(callback)
    rospy.spin()
