#!/usr/bin/env python
import rospy
import sys
import time
import message_filters
from geometry_msgs.msg import PointStamped
from space_msgs.msg import SatelitePose, RangeStamped
import space_tf
from tf import transformations
import numpy as np
from space_sensor_model import SimpleRangeFOVSensor

pub = rospy.Publisher('radar', RangeStamped, queue_size=10)
sensor_obj = SimpleRangeFOVSensor()

def callback(target_oe, chaser_oe):

    # calculate baseline
    tf_target_oe = space_tf.Converter.fromOEMessage(target_oe.position)
    tf_chaser_oe = space_tf.Converter.fromOEMessage(chaser_oe.position)

    # convert to TEME
    tf_target_teme = space_tf.CartesianTEME()
    tf_chaser_teme = space_tf.CartesianTEME()
    space_tf.Converter.convert(tf_target_oe, tf_target_teme)
    space_tf.Converter.convert(tf_chaser_oe, tf_chaser_teme)

    # vector from chaser to target in chaser body frame in [m]

    ## get current rotation of body
    R_body = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                chaser_oe.orientation.y,
                                                chaser_oe.orientation.z,
                                                chaser_oe.orientation.w])

    p_teme = (tf_target_teme.R -tf_chaser_teme.R)*1000
    p_body = np.dot(R_body[0:3,0:3].T, p_teme)

    # check if visible and augment sensor value
    visible, value = sensor_obj.get_measurement(p_body)

    if visible:
        msg = RangeStamped()
        msg.header.stamp = rospy.Time.now()
        msg.value.range = np.linalg.norm(tf_target_teme.R -tf_chaser_teme.R)*1000
        pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('RADAR_SIM', anonymous=True)
    rospy.loginfo("Radar sim started")
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    sensor_cfg = rospy.get_param("~sensor", 0)
    print sensor_cfg

    sensor_obj.fov_x = float(sensor_cfg["fov_x"])
    sensor_obj.fov_y = float(sensor_cfg["fov_y"])
    sensor_obj.max_range = float(sensor_cfg["max_range"])
    sensor_obj.mu = float(sensor_cfg["mu"])
    sensor_obj.sigma = float(sensor_cfg["sigma"])

    # set transforms!
    sensor_obj.set_frame_by_string(sensor_cfg["pose"], sensor_cfg["position"])

    # set message syncher and start
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(callback)
    rospy.spin()
