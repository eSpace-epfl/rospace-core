#!/usr/bin/env python
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
        tf_target_oe = space_tf.KepOrbElem()
        tf_chaser_oe = space_tf.KepOrbElem()

        tf_target_oe.from_message(target_oe.position)
        tf_chaser_oe.from_message(chaser_oe.position)

        # convert to TEME
        tf_target_teme = space_tf.CartesianTEME()
        tf_chaser_teme = space_tf.CartesianTEME()

        tf_target_teme.from_keporb(tf_target_oe)
        tf_chaser_teme.from_keporb(tf_chaser_oe)
        # vector from chaser to target in chaser body frame in [m]

        ## get current rotation of body
        R_body = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                    chaser_oe.orientation.y,
                                                    chaser_oe.orientation.z,
                                                    chaser_oe.orientation.w])

        p_teme = (tf_target_teme.R -tf_chaser_teme.R)*1000
        p_body = np.dot(R_body[0:3,0:3].T, p_teme)


        # publish observation
        msg = Marker()
        msg.header.frame_id = "cso"
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.points.append(Point(0, 0, 0))
        msg.points.append(Point(p_body[0], p_body[1], p_body[2]))
        msg.scale.x = 100
        msg.scale.y = 200
        msg.color.a = 1.0
        msg.color.r = 1.0
        self.pub_m.publish(msg)


        # check if visible and augment sensor value
        visible, value = sensor_obj.get_measurement(p_body)

        if visible and (target_oe.header.stamp - self.last_publish_time).to_sec() > 1.0/self.rate:

            msg = AzimutElevationStamped()
            msg.header.stamp = target_oe.header.stamp
            msg.value.azimut = value[0] + np.asscalar(np.random.normal(sensor_obj.mu[0], sensor_obj.sigma[0], 1))
            msg.value.elevation = value[1] + np.asscalar(np.random.normal(sensor_obj.mu[1], sensor_obj.sigma[1], 1))

            if(len(sensor_obj.mu) == 3):
                msg.range = np.linalg.norm(p_body) + np.asscalar(np.random.normal(sensor_obj.mu[2], sensor_obj.sigma[2], 1))
            #msg.R = tf_target_teme.R
            #msg.V = tf_target_teme.V
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
   # sensor_obj.mu = np.array(sensor_cfg["mu"])
   # sensor_obj.sigma = np.array(sensor_cfg["sigma"])
    #print sensor_obj.mu
    sensor_obj.mu = [float(x) for x in str(sensor_cfg["mu"]).split(" ")]
    sensor_obj.sigma = [float(x) for x in str(sensor_cfg["sigma"]).split(" ")]
    print sensor_obj.mu
    print sensor_obj.sigma[0]

    pub_rate = float(rospy.get_param("~publish_rate", 1))

    # set transforms!
    sensor_obj.set_frame_by_string(sensor_cfg["pose"], sensor_cfg["position"])

    # set up node
    node = AONSensorNode(sensor_obj, rate=pub_rate)

    # set message syncher and start
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(node.callback)
    rospy.spin()
