#!/usr/bin/env python
import rospy
import tf
import numpy as np
import message_filters
from geometry_msgs.msg import PoseStamped

sensor_cfg = 0
position_mode = 0
body_frame = 0
frame_cfg = 0

parent_position = np.array([0, 0, 0])
parent_orientation = np.array([0, 0, 0, 1])

def handle_sync(pose_msg, parent_msg):
    global parent_orientation
    global parent_position
    parent_orientation = np.array([parent_msg.pose.orientation.x, parent_msg.pose.orientation.y, parent_msg.pose.orientation.z, parent_msg.pose.orientation.w])
    parent_position = np.array([parent_msg.pose.position.x*1000, parent_msg.pose.position.y*1000, parent_msg.pose.position.z*1000])

    handle_pose(pose_msg)

def handle_pose(msg):
    global sensor_cfg
    global no_position
    global body_frame
    global frame_cfg

    quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    pos = np.array([msg.pose.position.x*1000, msg.pose.position.y*1000, msg.pose.position.z*1000])

    if position_mode == "zero_pos":
        pos = np.array([0,0,0])
    elif position_mode =="relative":
        pos = pos - parent_position

    br = tf.TransformBroadcaster()

    br.sendTransform(pos,
                     quat,
                     rospy.Time.now(),
                     body_frame,
                     msg.header.frame_id)

    for k in frame_cfg:
        br.sendTransform(np.array([float(x) for x in frame_cfg[k]["position"].split(" ")]),
                         np.array([float(x) for x in frame_cfg[k]["pose"].split(" ")]),
                         rospy.Time.now(),
                         k,
                         body_frame)


    for k in sensor_cfg:
        br.sendTransform(np.array([float(x) for x in sensor_cfg[k]["position"].split(" ")]),
                         np.array([float(x) for x in sensor_cfg[k]["pose"].split(" ")]),
                     rospy.Time.now(),
                     k,
                     body_frame)


if __name__ == '__main__':

    rospy.init_node('satelite_tf_publisher')
    sensor_cfg =  rospy.get_param("~sensors", "")
    frame_cfg = rospy.get_param("~frames", "")
    body_frame = rospy.get_param("~body_frame", "cso")
    position_mode = rospy.get_param("~mode", "absolute")

    # position mode zero_pos = no position is broadcasted for body frame (for debugging)
    # position mode absolute = absolute position from orbital elements is published
    # position mode relative = relative position w.r.t to "relative_parent" message is published

    if position_mode == "absolute" or position_mode == "zero_pos":
        rospy.Subscriber('pose',
                         PoseStamped,
                         handle_pose)

    elif position_mode == "relative":
        pose_sub = message_filters.Subscriber('pose', PoseStamped)
        relative_parent_sub = message_filters.Subscriber('relative_parent', PoseStamped)
        ts = message_filters.TimeSynchronizer([pose_sub, relative_parent_sub], 10)
        ts.registerCallback(handle_sync)

    rospy.spin()
