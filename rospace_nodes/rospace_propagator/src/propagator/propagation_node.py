#!/usr/bin/env python

# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

#   Orbit propagation node that publishes coordinates
#   in keplerian (oe) and cartesian (state) coordinates.

import numpy as np
import rospy
import time
import threading
import message_filters
import epoch_clock
import rospace_lib

from time import sleep
from math import radians

from OrekitPropagator import OrekitPropagator
from EpochClock import EpochClock
from rosgraph_msgs.msg import Clock
from rospace_msgs.srv import ClockService
from rospace_msgs.srv import SyncNodeService
from rospace_msgs.msg import PoseVelocityStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3Stamped
from rospace_msgs.msg import ThrustIsp
from rospace_msgs.msg import SatelitePose
from rospace_msgs.msg import SatelliteTorque

class ClockServer(threading.Thread):
    """
    Class setting up and communicating with Clock service.

    Can start/pause simulation and change simulation parameters
    like publishing frequency and step size.
    """

    def __init__(self,
                 realtime_factor,
                 frequency,
                 step_size,
                 start_running=False):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.start()
        self.SimRunning = start_running
        self.realtime_factor = realtime_factor
        self.frequency = frequency
        self.step_size = step_size

        self.syncSubscribers = 0
        self.readyCount = 0

    def handle_start_stop_clock(self, req):
        """
        Method called when service requested.

        Start/Pauses simulation or changes siulation parameter depending
        on input given through GUI

        Args:
            req: ClockService srv message

        Return:
            bool: SimulationRunning
            float: Step size
            float: Publish frequency .
        """

        if req.trigger:  # start/stop simulation
            if self.SimRunning:
                self.SimRunning = False
            else:
                self.SimRunning = True
        elif req.dt_size > 0 and req.p_freq > 0:
            self.frequency = req.p_freq
            self.step_size = req.dt_size
            self.realtime_factor = req.p_freq * req.dt_size

        return [self.SimRunning, self.step_size, self.frequency]

    def handle_sync_nodes(self, req):
        """
        Very basic Service to sync nodes.

        Every node has to subscribe and after each time step
        call service.

        Args:
            req: SyncNodeService srv message

        Returns:
            bool : True if adding node to subscribtion list,
                   False if node reports ready
        """

        if req.subscribe is True and req.ready is False:
            self.syncSubscribers += 1
            return True
        elif req.subscribe is False and req.ready is True:
            self.updateReadyCount(reset=False)
            return False

    def updateReadyCount(self, reset):
        """
        Method to count nodes which reported to be ready after one time step.

        Args:
            reset: if true resets ready count if false increases count by 1

        """

        self.lock.acquire()
        if reset:
            self.readyCount = 0
        else:
            self.readyCount += 1
        self.lock.release()

    def run(self):
        """
        Rospy service for synchronizing nodes and simulation time.
        Method is running/spinning on seperate thread.
        """
        rospy.Service('/sync_nodes',
                      SyncNodeService,
                      self.handle_sync_nodes)
        rospy.loginfo("Node-Synchronisation Service ready.")

        rospy.Service('/start_stop_clock',
                      ClockService,
                      self.handle_start_stop_clock)
        rospy.loginfo("Clock Service ready. Can start simulation now.")

        rospy.spin()  # wait for node to shutdown.
        self.root.quit()


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
            qns_init_ta.dA = float(rospy.get_param("~oe_ta_init/ada")) #/ (a*1000.0)
            qns_init_ta.dL = float(rospy.get_param("~oe_ta_init/adL")) #/ (a*1000.0)
            qns_init_ta.dEx = float(rospy.get_param("~oe_ta_init/adEx")) #/ (a*1000.0)
            qns_init_ta.dEy = float(rospy.get_param("~oe_ta_init/adEy")) #/ (a*1000.0)
            qns_init_ta.dIx = float(rospy.get_param("~oe_ta_init/adIx")) #/ (a*1000.0)
            qns_init_ta.dIy = float(rospy.get_param("~oe_ta_init/adIy")) #/ (a*1000.0)

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
        cart: orbit state vector
        att: satellite attitude in quaternions
        time: time stamp

    Returns:
       msg.SatelitePose: message containing orbital elements and orientation
       msg.PoseStamed: message for cartesian TEME pose
    """

    # convert to keplerian elements
    oe = rospace_lib.KepOrbElem()
    oe.from_cartesian(cart)

    msg = SatelitePose()

    msg.header.stamp = time
    msg.header.frame_id = "teme"

    msg.position.semimajoraxis = oe.a
    msg.position.eccentricity = oe.e
    msg.position.inclination = np.rad2deg(oe.i)
    msg.position.arg_perigee = np.rad2deg(oe.w)
    msg.position.raan = np.rad2deg(oe.O)
    msg.position.true_anomaly = np.rad2deg(oe.v)

    msg.orientation.x = att[0]
    msg.orientation.y = att[1]
    msg.orientation.z = att[2]
    msg.orientation.w = att[3]

    # set message for cartesian TEME pose
    msg_pose = PoseVelocityStamped()
    msg_pose.header.stamp = time
    msg_pose.header.frame_id = "teme"
    msg_pose.pose.position.x = cart.R[0]
    msg_pose.pose.position.y = cart.R[1]
    msg_pose.pose.position.z = cart.R[2]
    msg_pose.pose.orientation.x = att[0]
    msg_pose.pose.orientation.y = att[1]
    msg_pose.pose.orientation.z = att[2]
    msg_pose.pose.orientation.w = att[3]
    msg_pose.velocity.x = cart.V[0]
    msg_pose.velocity.y = cart.V[1]
    msg_pose.velocity.z = cart.V[2]

    return [msg, msg_pose]


def force_torque_to_msgs(force, torque, time):

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

    msg = Vector3Stamped()

    msg.header.stamp = time
    msg.header.frame_id = "teme_frame"

    msg.vector.x = bfield[0]
    msg.vector.y = bfield[1]
    msg.vector.z = bfield[2]

    return msg


if __name__ == '__main__':
    rospy.init_node('propagation_node', anonymous=True)

    # get defined simulation parameters
    sim_parameter = dict()
    if rospy.has_param("~TIME_SHIFT"):
        # get simulation time shift before t=0:
        sim_parameter['TIME_SHIFT'] = float(rospy.get_param("~TIME_SHIFT"))
    if rospy.has_param("~frequency"):
        sim_parameter['frequency'] = int(rospy.get_param("~frequency"))
    if rospy.has_param("~oe_epoch"):
        sim_parameter['oe_epoch'] = str(rospy.get_param("~oe_epoch"))
    if rospy.has_param("~step_size"):
        sim_parameter['step_size'] = float(rospy.get_param("~step_size"))

    SimTime = EpochClock(**sim_parameter)

    rospy.loginfo("Epoch of init. Orbit Elements = " +
                  SimTime.datetime_oe_epoch.strftime("%Y-%m-%d %H:%M:%S"))
    rospy.loginfo("Realtime Factor = " + str(SimTime.realtime_factor))

    # set publish frequency & time step size as ros parameter
    # do this before setting /epoch parameter to ensure epoch_clock library
    # can find them
    rospy.set_param('/publish_freq', SimTime.frequency)
    rospy.set_param('/time_step_size', str(SimTime.step_size))  # set as string, so no overflow
    rospy.set_param('/epoch',
                    SimTime.datetime_oe_epoch.strftime("%Y-%m-%d %H:%M:%S"))

    pub_clock = rospy.Publisher('/clock', Clock, queue_size=10)
    if rospy.has_param("/start_running"):
        # Start GUI clock service on new thread
        ClockServer = ClockServer(SimTime.realtime_factor,
                                  SimTime.frequency,
                                  SimTime.step_size,
                                  rospy.get_param("/start_running"))
    else:
        ClockServer = ClockServer(SimTime.realtime_factor,
                                  SimTime.frequency,
                                  SimTime.step_size)

    # Subscribe to propulsion node and attitude control
    thrust_force = message_filters.Subscriber('force', WrenchStamped)
    thrust_ispM = message_filters.Subscriber('IspMean', ThrustIsp)
    # att_sub = message_filters.Subscriber('chaser/attitude_ctrl', FixedRotationStamped)

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
    prop_chaser = OrekitPropagator()
    # get settings from yaml file
    ch_prop_file = "/" + rospy.get_param("~ns_chaser") + "/propagator_settings"
    propSettings = rospy.get_param(ch_prop_file, 0)
    prop_chaser.initialize(propSettings,
                           init_state_ch,
                           SimTime.datetime_oe_epoch)

    # add callback to thrust function
    Tsync = message_filters.TimeSynchronizer([thrust_force, thrust_ispM], 10)
    Tsync.registerCallback(prop_chaser.thrust_torque_callback)
    # att_sub.registerCallback(prop_chaser.attitude_fixed_rot_callback)

    prop_target = OrekitPropagator()
    # get settings from yaml file
    ta_prop_file = "/" + rospy.get_param("~ns_target") + "/propagator_settings"
    propSettings = rospy.get_param(ta_prop_file, 0)
    prop_target.initialize(propSettings,
                           init_state_ta,
                           SimTime.datetime_oe_epoch)

    OrekitPropagator.load_magnetic_field_models(SimTime.datetime_oe_epoch)
    OrekitPropagator.create_data_validity_checklist()
    rospy.loginfo("Propagators initialized!")

    # Update first step so that other nodes don't give errors
    # msg_cl = Clock()
    # msg_cl.clock.secs = int(0)
    # msg_cl.clock.nsecs = int(0)
    # msg_cl = Clock()
    # pub_clock.publish(msg_cl)
    # init epoch clock for propagator
    run_sim = sim_parameter['TIME_SHIFT'] == 0.0  # start propagating if no timeshift
    epoch = epoch_clock.Epoch()
    epoch_now = epoch.now()  # initialize, because simulation starts stopped

    while not rospy.is_shutdown():
        comp_time = time.clock()

        # change of realtime factor requested:
        if ClockServer.realtime_factor != SimTime.realtime_factor:
            SimTime.updateTimeFactors(ClockServer.realtime_factor,
                                      ClockServer.frequency,
                                      ClockServer.step_size)
            epoch.changeFrequency(ClockServer.frequency)
            epoch.changeStep(ClockServer.step_size)

        if ClockServer.SimRunning:
            # Wait for other nodes which subscribed to service
            while ClockServer.syncSubscribers > 0:
                if(ClockServer.readyCount >= ClockServer.syncSubscribers):
                    ClockServer.updateReadyCount(reset=True)
                    break

            # update clock
            msg_cl = Clock()
            [msg_cl, epoch_now, run_sim] = SimTime.updateClock(msg_cl)
            pub_clock.publish(msg_cl)

        if run_sim:
            # propagate to epoch_now
            [cart_ch, att_ch, force_ch, d_torque_ch, B_field_ch] = \
                prop_chaser.propagate(epoch_now)
            [cart_ta, att_ta, force_ta, d_torque_ta, B_field_ta] = \
                prop_target.propagate(epoch_now)

            rospy_now = rospy.Time.now()

            [msg_ch, msg_pose_ch] = cart_to_msgs(cart_ch, att_ch, rospy_now)
            [msg_ta, msg_pose_ta] = cart_to_msgs(cart_ta, att_ta, rospy_now)

            [msg_FT_ch, msg_dtorque_ch] = force_torque_to_msgs(force_ch,
                                                               d_torque_ch,
                                                               rospy_now)
            [msg_FT_ta, msg_dtorque_ta] = force_torque_to_msgs(force_ta,
                                                               d_torque_ta,
                                                               rospy_now)

            msg_B_field_ch = Bfield_to_msgs(B_field_ch, rospy_now)
            msg_B_field_ta = Bfield_to_msgs(B_field_ta, rospy_now)

            pub_ch.publish(msg_ch)
            pub_pose_ch.publish(msg_pose_ch)
            pub_dtorque_ch.publish(msg_dtorque_ch)
            pub_FT_ch.publish(msg_FT_ch)
            pub_Bfield_ch.publish(msg_B_field_ch)

            pub_ta.publish(msg_ta)
            pub_pose_ta.publish(msg_pose_ta)
            pub_dtorque_ta.publish(msg_dtorque_ta)
            pub_FT_ta.publish(msg_FT_ta)
            pub_Bfield_ch.publish(msg_B_field_ta)

        # calculate reminding sleeping time
        sleep_time = SimTime.rate - (time.clock() - comp_time)
        # Improve sleep time as Gazebo for more accurate time update
        if sleep_time > 0:
            sleep(sleep_time)
        elif ClockServer.syncSubscribers == 0:
            rospy.logwarn("Propagator too slow for publishing rate.")
