#!/usr/bin/env python

# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import numpy as np
import rospy
import time
import threading
import message_filters
import rospace_lib

from math import radians

from OrekitPropagator import OrekitPropagator
from FileDataHandler import FileDataHandler
from rospace_msgs.msg import PoseVelocityStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3Stamped
from rospace_msgs.msg import ThrustIsp
from rospace_msgs.msg import SatelitePose
from rospace_msgs.msg import SatelliteTorque
from std_srvs.srv import Empty


class ExitServer(threading.Thread):
    '''Server which shuts down node correctly when called.

    Rospy currently has a bug which doesn't shutdown the node correctly.
    This causes a problem when a profiler is used, as the results are not output
    if shut down is not performed in the right way.
    '''
    def __init__(self):
        threading.Thread.__init__(self)
        self.exiting = False
        self.start()

    def exit_node(self, req):
        self.exiting = True
        return []

    def run(self):
        rospy.Service('/exit_node', Empty, self.exit_node)
        rospy.spin()


def get_init_state_from_param():
    """
    Method to get orbital elements from parameters.

    Depending on which parameters defined in launch file different
    parameters are extracted.

    Returns:
        Object: Initial state of chaser
        Object: Initial state of target
    """
    if rospy.has_param("~oe_ch_init/a"):
        # mean elements for init
        a = float(rospy.get_param("~oe_ch_init/a"))
        e = float(rospy.get_param("~oe_ch_init/e"))
        i = float(rospy.get_param("~oe_ch_init/i"))
        O = float(rospy.get_param("~oe_ch_init/O"))
        w = float(rospy.get_param("~oe_ch_init/w"))

        init_state_ch = rospace_lib.KepOrbElem()
        init_state_ch.a = a
        init_state_ch.e = e
        init_state_ch.i = radians(i)  # inclination
        init_state_ch.O = radians(O)
        init_state_ch.w = radians(w)

        if rospy.has_param("~oe_ch_init/v"):
            init_state_ch.v = radians(float(rospy.get_param("~oe_ch_init/v")))
        elif rospy.has_param("~oe_ch_init/m"):
            init_state_ch.m = radians(float(rospy.get_param("~oe_ch_init/m")))
        else:
            raise ValueError("No Anomaly for initialization of chaser")

        if rospy.get_param("~oe_ta_rel"):  # relative target state
            qns_init_ta = rospace_lib.QNSRelOrbElements()
            # a = 0.001
            qns_init_ta.dA = float(rospy.get_param("~oe_ta_init/ada"))  # / (a*1000.0)
            qns_init_ta.dL = float(rospy.get_param("~oe_ta_init/adL"))  # / (a*1000.0)
            qns_init_ta.dEx = float(rospy.get_param("~oe_ta_init/adEx"))  # / (a*1000.0)
            qns_init_ta.dEy = float(rospy.get_param("~oe_ta_init/adEy"))  # / (a*1000.0)
            qns_init_ta.dIx = float(rospy.get_param("~oe_ta_init/adIx"))  # / (a*1000.0)
            qns_init_ta.dIy = float(rospy.get_param("~oe_ta_init/adIy"))  # / (a*1000.0)

            init_state_ta = rospace_lib.KepOrbElem()
            init_state_ta.from_qns_relative(qns_init_ta, init_state_ch)

        else:  # absolute target state
            a_t = float(rospy.get_param("~oe_ta_init/a"))
            e_t = float(rospy.get_param("~oe_ta_init/e"))
            i_t = float(rospy.get_param("~oe_ta_init/i"))
            O_t = float(rospy.get_param("~oe_ta_init/O"))
            w_t = float(rospy.get_param("~oe_ta_init/w"))

            init_state_ta = rospace_lib.KepOrbElem()
            init_state_ta.a = a_t
            init_state_ta.e = e_t
            init_state_ta.i = radians(i_t)
            init_state_ta.O = radians(O_t)
            init_state_ta.w = radians(w_t)

            if rospy.has_param("~oe_ta_init/v"):
                init_state_ta.v = radians(float(rospy.get_param("~oe_ta_init/v")))
            elif rospy.has_param("~oe_ta_init/m"):
                init_state_ta.m = radians(float(rospy.get_param("~oe_ta_init/m")))
            else:
                raise ValueError("No Anomaly for initialization of target")

    elif rospy.has_param("~oe_ch_init/x"):
        x = float(rospy.get_param("~oe_ch_init/x"))
        y = float(rospy.get_param("~oe_ch_init/y"))
        z = float(rospy.get_param("~oe_ch_init/z"))
        xDot = float(rospy.get_param("~oe_ch_init/xDot"))
        yDot = float(rospy.get_param("~oe_ch_init/yDot"))
        zDot = float(rospy.get_param("~oe_ch_init/zDot"))

        init_state_ch = rospace_lib.CartesianITRF()
        init_state_ch.R = np.array([x, y, z])
        init_state_ch.V = np.array([xDot, yDot, zDot])

        x = float(rospy.get_param("~oe_ta_init/x"))
        y = float(rospy.get_param("~oe_ta_init/y"))
        z = float(rospy.get_param("~oe_ta_init/z"))
        xDot = float(rospy.get_param("~oe_ta_init/xDot"))
        yDot = float(rospy.get_param("~oe_ta_init/yDot"))
        zDot = float(rospy.get_param("~oe_ta_init/zDot"))

        init_state_ta = rospace_lib.CartesianITRF()
        init_state_ta.R = np.array([x, y, z])
        init_state_ta.V = np.array([xDot, yDot, zDot])

    return [init_state_ch, init_state_ta]


def cart_to_msgs(cart, att, time):
    """
    Packs cartesian orbit elements to message.

    Args:
        cart (:obj:`rospace_lib.Cartesian`): orbit state vector
        att (Orekit.Attitude): satellite attitude in quaternions
        time (:obj:`rospy.Time`): time stamp

    Returns:
       msg.SatelitePose: message containing orbital elements and orientation
       msg.PoseStamed: message for cartesian TEME pose
    """

    # convert to keplerian elements
    oe = rospace_lib.KepOrbElem()
    oe.from_cartesian(cart)

    msg = SatelitePose()

    msg.header.stamp = time
    msg.header.frame_id = "J2K"

    msg.position.semimajoraxis = oe.a
    msg.position.eccentricity = oe.e
    msg.position.inclination = np.rad2deg(oe.i)
    msg.position.arg_perigee = np.rad2deg(oe.w)
    msg.position.raan = np.rad2deg(oe.O)
    msg.position.true_anomaly = np.rad2deg(oe.v)

    orient = att.getRotation()
    spin = att.getSpin()
    acc = att.getRotationAcceleration()

    msg.orientation.x = orient.q1
    msg.orientation.y = orient.q2
    msg.orientation.z = orient.q3
    msg.orientation.w = orient.q0

    # set message for cartesian TEME pose
    msg_pose = PoseVelocityStamped()
    msg_pose.header.stamp = time
    msg_pose.header.frame_id = "J2K"
    msg_pose.pose.position.x = cart.R[0]
    msg_pose.pose.position.y = cart.R[1]
    msg_pose.pose.position.z = cart.R[2]
    msg_pose.pose.orientation.x = orient.q1
    msg_pose.pose.orientation.y = orient.q2
    msg_pose.pose.orientation.z = orient.q3
    msg_pose.pose.orientation.w = orient.q0
    msg_pose.velocity.x = cart.V[0]
    msg_pose.velocity.y = cart.V[1]
    msg_pose.velocity.z = cart.V[2]
    msg_pose.spin.x = spin.x
    msg_pose.spin.y = spin.y
    msg_pose.spin.z = spin.z
    msg_pose.rot_acceleration.x = acc.x
    msg_pose.rot_acceleration.x = acc.y
    msg_pose.rot_acceleration.x = acc.z

    return [msg, msg_pose]


def force_torque_to_msgs(force, torque, time):
    """Packs force and torque vectors in satellite frame to message.

    Args:
        force: force vector acting on satellite
        torque: DisturbanceTorqueStorage object holding current torques acting on satellite
        time: time stamp

    Returns:
       msg.WrenchStamped: message for total force and torque acting on satellite
       msg.SatelliteTorque: message for individual disturbance torques
    """

    FT_msg = WrenchStamped()

    FT_msg.header.stamp = time
    FT_msg.header.frame_id = "sat_frame"

    FT_msg.wrench.force.x = force[0]
    FT_msg.wrench.force.y = force[1]
    FT_msg.wrench.force.z = force[2]
    FT_msg.wrench.torque.x = torque.dtorque[5][0]
    FT_msg.wrench.torque.y = torque.dtorque[5][1]
    FT_msg.wrench.torque.z = torque.dtorque[5][2]

    msg = SatelliteTorque()

    msg.header.stamp = time
    msg.header.frame_id = "sat_frame"

    msg.gravity.active_disturbance = torque.add[0]
    msg.gravity.torque.x = torque.dtorque[0][0]
    msg.gravity.torque.y = torque.dtorque[0][1]
    msg.gravity.torque.z = torque.dtorque[0][2]

    msg.magnetic.active_disturbance = torque.add[1]
    msg.magnetic.torque.x = torque.dtorque[1][0]
    msg.magnetic.torque.y = torque.dtorque[1][1]
    msg.magnetic.torque.z = torque.dtorque[1][2]

    msg.solar_pressure.active_disturbance = torque.add[2]
    msg.solar_pressure.torque.x = torque.dtorque[2][0]
    msg.solar_pressure.torque.y = torque.dtorque[2][1]
    msg.solar_pressure.torque.z = torque.dtorque[2][2]

    msg.drag.active_disturbance = torque.add[3]
    msg.drag.torque.x = torque.dtorque[3][0]
    msg.drag.torque.y = torque.dtorque[3][1]
    msg.drag.torque.z = torque.dtorque[3][2]

    msg.external.active_disturbance = torque.add[4]
    msg.external.torque.x = torque.dtorque[4][0]
    msg.external.torque.y = torque.dtorque[4][1]
    msg.external.torque.z = torque.dtorque[4][2]

    return [FT_msg, msg]


def Bfield_to_msgs(bfield, time):
    """Packs the local magnetic field to message.

    Args:
        bfield: local magnetic field vector in satellite frame
        time: time stamp

    Returns:
       msg.Vector3Stamped: message for the local magnetic field vector
    """

    msg = Vector3Stamped()

    msg.header.stamp = time
    msg.header.frame_id = "sat_frame"

    msg.vector.x = bfield.x
    msg.vector.y = bfield.y
    msg.vector.z = bfield.z

    return msg


if __name__ == '__main__':
    rospy.init_node('propagation_node', anonymous=True)

    ExitServer = ExitServer()
    SimTime = rospace_lib.clock.SimTimePublisher()
    SimTime.set_up_simulation_time()

    # Init publisher and rate limiter
    pub_ch = rospy.Publisher('oe_chaser', SatelitePose, queue_size=10)
    pub_pose_ch = rospy.Publisher('pose_chaser', PoseVelocityStamped, queue_size=10)
    pub_dtorque_ch = rospy.Publisher('dtorque_chaser', SatelliteTorque, queue_size=10)
    pub_FT_ch = rospy.Publisher('forcetorque_chaser', WrenchStamped, queue_size=10)
    pub_Bfield_ch = rospy.Publisher('B_field_chaser', Vector3Stamped, queue_size=10)

    pub_ta = rospy.Publisher('oe_target', SatelitePose, queue_size=10)
    pub_pose_ta = rospy.Publisher('pose_target', PoseVelocityStamped, queue_size=10)
    pub_dtorque_ta = rospy.Publisher('dtorque_target', SatelliteTorque, queue_size=10)
    pub_FT_ta = rospy.Publisher('forcetorque_target', WrenchStamped, queue_size=10)
    pub_Bfield_ta = rospy.Publisher('B_field_target', Vector3Stamped, queue_size=10)

    [init_state_ch, init_state_ta] = get_init_state_from_param()

    OrekitPropagator.init_jvm()

    # Initialize Data handlers, loading data in orekit .zip file
    FileDataHandler.load_magnetic_field_models(SimTime.datetime_oe_epoch)

    prop_chaser = OrekitPropagator()
    # get settings from yaml file
    ch_prop_file = "/" + rospy.get_param("~ns_chaser") + "/propagator_settings"
    propSettings = rospy.get_param(ch_prop_file, 0)
    prop_chaser.initialize(propSettings,
                           init_state_ch,
                           SimTime.datetime_oe_epoch)

    # Subscribe to propulsion node and attitude control if one of those is active
    if prop_chaser._hasThrust:
        external_force_ch = message_filters.Subscriber('thrust_force_chaser', WrenchStamped)
        thrust_ispM_ch = message_filters.Subscriber('IspMean_chaser', ThrustIsp)
        Tsync = message_filters.TimeSynchronizer([external_force_ch, thrust_ispM_ch], 10)
        Tsync.registerCallback(prop_chaser.thrust_callback)
    if prop_chaser._hasAttitudeProp:
        external_torque = rospy.Subscriber('actuator_torque_chaser', WrenchStamped, prop_chaser.magnetotorque_callback)

    prop_target = OrekitPropagator()
    # get settings from yaml file
    ta_prop_file = "/" + rospy.get_param("~ns_target") + "/propagator_settings"
    propSettings = rospy.get_param(ta_prop_file, 0)
    prop_target.initialize(propSettings,
                           init_state_ta,
                           SimTime.datetime_oe_epoch)

    # TODO: Michael's way how to subscribe to forces of torquer. Change so that subscribers are the same
    # sub_target_force = rospy.Subscriber('force', WrenchStamped, prop_target.thrust_torque_callback)

    # Subscribe to propulsion node and attitude control if one of those is active
    if prop_target._hasThrust:
        external_force_ta = message_filters.Subscriber('thrust_force_target', WrenchStamped)
        thrust_ispM_ta = message_filters.Subscriber('IspMean_target', ThrustIsp)
        Tsync = message_filters.TimeSynchronizer([external_force_ta, thrust_ispM_ta], 10)
        Tsync.registerCallback(prop_target.thrust_callback)
    if prop_target._hasAttitudeProp:
        external_torque = rospy.Subscriber('actuator_torque_target', WrenchStamped, prop_target.magnetotorque_callback)

    FileDataHandler.create_data_validity_checklist()

    rospy.loginfo("Propagators initialized!")

    while not rospy.is_shutdown() and not ExitServer.exiting:
        comp_time = time.clock()

        epoch_now = SimTime.update_simulation_time()
        if SimTime.time_shift_passed:
            # check if data still loaded
            FileDataHandler.check_data_availability(epoch_now)
            # propagate to epoch_now
            [cart_ch, att_ch, force_ch, d_torque_ch, B_field_ch] = \
                prop_chaser.propagate(epoch_now)
            [cart_ta, att_ta, force_ta, d_torque_ta, B_field_ta] = \
                prop_target.propagate(epoch_now)

            rospy_now = rospy.Time.now()

            [msg_ch, msg_pose_ch] = cart_to_msgs(cart_ch, att_ch, rospy_now)
            [msg_FT_ch, msg_dtorque_ch] = force_torque_to_msgs(force_ch,
                                                               d_torque_ch,
                                                               rospy_now)
            msg_B_field_ch = Bfield_to_msgs(B_field_ch, rospy_now)
            pub_ch.publish(msg_ch)
            pub_pose_ch.publish(msg_pose_ch)
            pub_dtorque_ch.publish(msg_dtorque_ch)
            pub_FT_ch.publish(msg_FT_ch)
            pub_Bfield_ch.publish(msg_B_field_ch)

            [msg_ta, msg_pose_ta] = cart_to_msgs(cart_ta, att_ta, rospy_now)

            [msg_FT_ta, msg_dtorque_ta] = force_torque_to_msgs(force_ta,
                                                               d_torque_ta,
                                                               rospy_now)

            msg_B_field_ta = Bfield_to_msgs(B_field_ta, rospy_now)

            pub_ta.publish(msg_ta)
            pub_pose_ta.publish(msg_pose_ta)
            pub_dtorque_ta.publish(msg_dtorque_ta)
            pub_FT_ta.publish(msg_FT_ta)
            pub_Bfield_ta.publish(msg_B_field_ta)

        SimTime.sleep_to_keep_frequency()
