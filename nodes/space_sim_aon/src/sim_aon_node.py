#!/usr/bin/env python

# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import numpy as np
import rospy

import message_filters
from tf import transformations
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import space_tf
from space_msgs.msg import SatelitePose, AzimutElevationStamped
from space_sensor_model import PaperAnglesSensor


class AONSensorNode:
    def __init__(self, sensor, rate=1.0):
        self.last_publish_time = rospy.Time.now()
        self.rate = rate
        self.sensor = sensor
        self.pub = rospy.Publisher('aon', AzimutElevationStamped, queue_size=10)

        self.pub_m = rospy.Publisher("aon_observation", Marker, queue_size=10)

    def callback(self, target_oe, chaser_oe):
        # calculate baseline
        O_T = space_tf.KepOrbElem()
        O_C = space_tf.KepOrbElem()

        O_T.from_message(target_oe.position)
        O_C.from_message(chaser_oe.position)

        # convert to cartesian
        p_T = space_tf.Cartesian()
        p_C = space_tf.Cartesian()

        p_T.from_keporb(O_T)
        p_C.from_keporb(O_C)
        # vector from chaser to target in chaser body frame in [m]

        # get current rotation of chaser
        R_J2K_C = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                    chaser_oe.orientation.y,
                                                    chaser_oe.orientation.z,
                                                    chaser_oe.orientation.w])

        # Calculate relative vector from chaser to target in J2K Frame
        J2K_p_C_T = (p_T.R -p_C.R)*1000

        # rotate vector into chaser (C) frame
        p_C_T = np.dot(R_J2K_C[0:3,0:3].T, J2K_p_C_T)

        # publish observation
        msg = Marker()
        msg.header.frame_id = "chaser"
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.points.append(Point(0, 0, 0))
        msg.points.append(Point(p_C_T[0], p_C_T[1], p_C_T[2]))
        msg.scale.x = 100
        msg.scale.y = 200
        msg.color.a = 1.0
        msg.color.r = 1.0
        self.pub_m.publish(msg)

        # check if visible and augment sensor value
        visible, value = sensor_obj.get_measurement(p_C_T)

        if visible and (target_oe.header.stamp - self.last_publish_time).to_sec() > 1.0/self.rate:
            msg = AzimutElevationStamped()
            msg.header.stamp = target_oe.header.stamp
            msg.value.azimut = value[0]
            msg.value.elevation = value[1]
            self.pub.publish(msg)
            self.last_publish_time = target_oe.header.stamp


if __name__ == '__main__':
    rospy.init_node('AON_SIM', anonymous=True)
    rospy.loginfo("AoN sim started")
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    sensor_cfg = rospy.get_param("~sensor", 0)

    sensor_obj = PaperAnglesSensor()
    sensor_obj.fov_x = float(sensor_cfg["fov_x"])
    sensor_obj.fov_y = float(sensor_cfg["fov_y"])
    sensor_obj.max_range = float(sensor_cfg["max_range"])
    sensor_obj.mu = float(sensor_cfg["mu"])
    sensor_obj.sigma = float(sensor_cfg["sigma"])

    pub_rate = float(rospy.get_param("~publish_rate", 1))

    # set transforms!
    sensor_obj.set_frame_by_string(sensor_cfg["pose"], sensor_cfg["position"])

    # set up node
    node = AONSensorNode(sensor_obj, rate=pub_rate)

    # set message syncher and start
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(node.callback)
    rospy.spin()
