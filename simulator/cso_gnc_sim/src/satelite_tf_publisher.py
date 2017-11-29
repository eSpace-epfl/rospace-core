#!/usr/bin/env python
import rospy
import tf
import numpy as np
import message_filters
import space_tf
from space_msgs.msg import SatelitePose
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
                     msg.header.stamp,
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


def handle_target_oe(msg):
    global body_frame
    # convert to R/V
    target_oe = space_tf.KepOrbElem()
    target_oe.from_message(msg.position)

    # convert to TEME
    tf_target_teme = space_tf.CartesianTEME()
    tf_target_teme.from_keporb(target_oe)

    # calculate reference frame
    i = tf_target_teme.R / np.linalg.norm(tf_target_teme.R)

    j = tf_target_teme.V / np.linalg.norm(tf_target_teme.V)
    k = np.cross(i, j)
    R_ref = np.identity(4)
    R_ref[0, 0:3] = i
    R_ref[1, 0:3] = j
    R_ref[2, 0:3] = k
    R_ref[0:3, 0:3] = R_ref[0:3, 0:3].T

    q_ref = tf.transformations.quaternion_from_matrix(R_ref)

    # publish
    br = tf.TransformBroadcaster()

    br.sendTransform(tf_target_teme.R*1000,
                    q_ref,
                   msg.header.stamp,
                  body_frame+"_ref",
                 "teme")



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

        rospy.Subscriber('target_oe',
                         SatelitePose,
                         handle_target_oe)


    elif position_mode == "relative":
        pose_sub = message_filters.Subscriber('pose', PoseStamped)
        relative_parent_sub = message_filters.Subscriber('relative_parent', PoseStamped)
        ts = message_filters.TimeSynchronizer([pose_sub, relative_parent_sub], 10)
        ts.registerCallback(handle_sync)

    rospy.spin()
