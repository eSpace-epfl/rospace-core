#!/usr/bin/env python
import rospy
import message_filters
from space_msgs.msg import OrbitalElementsStamped, RangeStamped
import space_tf
import numpy as np

pub = rospy.Publisher('radar', RangeStamped, queue_size=10)

def callback(target_oe, chaser_oe):

    # calculate baseline
    tf_target_oe = space_tf.Converter.fromOEMessage(target_oe)
    tf_chaser_oe = space_tf.Converter.fromOEMessage(chaser_oe)

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

    sensor_cfg = rospy.get_param("~sensor", 0)
    print sensor_cfg["fov_x"]

    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(callback)
    rospy.spin()
