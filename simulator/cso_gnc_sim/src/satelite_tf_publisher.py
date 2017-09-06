#!/usr/bin/env python
import rospy
import tf
import numpy as np
from geometry_msgs.msg import PoseStamped

sensor_cfg = 0
no_position = 0
body_frame = 0


def handle_pose(msg):
    global sensor_cfg
    global no_position
    global body_frame

    if no_position:
        msg.pose.position.x = 0
        msg.pose.position.y = 0
        msg.pose.position.z = 0

    br = tf.TransformBroadcaster()
    quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    br.sendTransform((msg.pose.position.x*1000, msg.pose.position.y*1000, msg.pose.position.z*1000),
                     quat,
                     rospy.Time.now(),
                     body_frame,
                     msg.header.frame_id)


    for k in sensor_cfg:
        br.sendTransform(np.array([float(x) for x in sensor_cfg[k]["position"].split(" ")]),
                         np.array([float(x) for x in sensor_cfg[k]["pose"].split(" ")]),
                     rospy.Time.now(),
                     k,
                     body_frame)

if __name__ == '__main__':

    rospy.init_node('satelite_tf_publisher')
    sensor_cfg =  rospy.get_param("~sensors", "")
    body_frame = rospy.get_param("~body_frame", "cso")
    no_position = bool(rospy.get_param("~no_position", True))


    rospy.Subscriber('pose',
                     PoseStamped,
                     handle_pose)
    rospy.spin()
