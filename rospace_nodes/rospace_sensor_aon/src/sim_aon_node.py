#!/usr/bin/env python
import numpy as np
import rospy

import message_filters
from tf import transformations
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import space_tf
from space_msgs.msg import SatelitePose, AzimutElevationRangeStamped
from space_sensor_model import PaperAnglesSensor


class AONSensorNode:
    def __init__(self, sensor, rate=1.0):
        self.last_publish_time = rospy.Time.now()
        self.rate = rate
        self.sensor = sensor
        self.pub = rospy.Publisher('aon', AzimutElevationRangeStamped, queue_size=10)

        self.pub_m = rospy.Publisher("aon_observation", Marker, queue_size=10)

    def callback(self, target_oe, chaser_oe):
        # calculate baseline
        osc_O_t = space_tf.KepOrbElem()
        osc_O_c = space_tf.KepOrbElem()

        osc_O_t.from_message(target_oe.position)
        osc_O_c.from_message(chaser_oe.position)

        # convert to TEME
        S_t = space_tf.CartesianTEME()
        S_c = space_tf.CartesianTEME()

        S_t.from_keporb(osc_O_t)
        S_c.from_keporb(osc_O_c)
        # vector from chaser to target in chaser body frame in [m]

        ## get current rotation of body
        R_J2K_CB = transformations.quaternion_matrix([chaser_oe.orientation.x,
                                                    chaser_oe.orientation.y,
                                                    chaser_oe.orientation.z,
                                                    chaser_oe.orientation.w])

        r_J2K = (S_t.R -S_c.R)*1000
        r_CB = np.dot(R_J2K_CB[0:3,0:3].T, r_J2K)

        # publish observation as marker for visualization
        msg = Marker()
        msg.header.frame_id = "cso"
        msg.type = Marker.ARROW
        msg.action = Marker.ADD
        msg.points.append(Point(0, 0, 0))
        msg.points.append(Point(r_CB[0], r_CB[1], r_CB[2]))
        msg.scale.x = 100
        msg.scale.y = 200
        msg.color.a = 1.0
        msg.color.r = 1.0
        self.pub_m.publish(msg)

        # check if visible and augment sensor value
        visible, value = sensor_obj.get_measurement(r_CB)
        msg = AzimutElevationRangeStamped()
        msg.header.stamp = target_oe.header.stamp

        # throttle publishing to maximum rate
        if not (target_oe.header.stamp - self.last_publish_time).to_sec() > 1.0/self.rate:
            return

        # if target is visible, publish message
        if visible:

            # these measurements already have noise added
            msg.value.azimut = value[0]
            msg.value.elevation = value[1]
            msg.valid = True

            # if range measurement is activated
            if len(sensor_obj.mu) == 3:
                msg.value.range = np.linalg.norm(r_CB) + np.asscalar(np.random.normal(sensor_obj.mu[2], sensor_obj.sigma[2], 1))

        else:
            msg.valid = False

        self.pub.publish(msg)
        self.last_publish_time = target_oe.header.stamp


if __name__ == '__main__':
    rospy.init_node('AON_SIM', anonymous=True)
    rospy.loginfo("AoN sim started")

    # subscribe to target/chaser location
    target_oe_sub = message_filters.Subscriber('target_oe', SatelitePose)
    chaser_oe_sub = message_filters.Subscriber('chaser_oe', SatelitePose)

    sensor_cfg = rospy.get_param("~sensor", 0)

    # load parameters
    sensor_obj = PaperAnglesSensor()
    sensor_obj.fov_x = float(sensor_cfg["fov_x"])
    sensor_obj.fov_y = float(sensor_cfg["fov_y"])
    sensor_obj.max_range = float(sensor_cfg["max_range"])

    # set sensor_obj.mu
    sensor_obj.mu = [float(x) for x in str(sensor_cfg["mu"]).split(" ")]
    sensor_obj.sigma = [float(x) for x in str(sensor_cfg["sigma"]).split(" ")]

    pub_rate = float(rospy.get_param("~publish_rate", 1))

    # set transforms!
    sensor_obj.set_frame_by_string(sensor_cfg["pose"], sensor_cfg["position"])

    # set up node
    node = AONSensorNode(sensor_obj, rate=pub_rate)

    # set message syncher and start
    ts = message_filters.TimeSynchronizer([target_oe_sub, chaser_oe_sub], 10)
    ts.registerCallback(node.callback)
    rospy.spin()
