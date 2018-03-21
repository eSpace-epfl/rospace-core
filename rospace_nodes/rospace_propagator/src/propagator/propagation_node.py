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
import rospace_lib

from time import sleep
from math import radians

from OrekitPropagator import OrekitPropagator
from EpochClock import EpochClock
from rosgraph_msgs.msg import Clock
from rospace_msgs.srv import ClockService
from rospace_msgs.srv import SyncNodeService
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from rospace_msgs.msg import ThrustIsp
from rospace_msgs.msg import SatelitePose


class ClockServer(threading.Thread):
    """
    Class setting up and communicating with Clock service.

    Can start/pause simulation and change simulation parameters
    like publishing frequency and step size.
    """

    def __init__(self, realtime_factor, frequency, step_size):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.start()
        self.SimRunning = False
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
    msg.header.frame_id = "J2K"

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
    msg_pose = PoseStamped()
    msg_pose.header.stamp = time
    msg_pose.header.frame_id = "J2K"
    msg_pose.pose.position.x = cart.R[0]
    msg_pose.pose.position.y = cart.R[1]
    msg_pose.pose.position.z = cart.R[2]
    msg_pose.pose.orientation.x = att[0]
    msg_pose.pose.orientation.y = att[1]
    msg_pose.pose.orientation.z = att[2]
    msg_pose.pose.orientation.w = att[3]

    return [msg, msg_pose]


if __name__ == '__main__':
    rospy.init_node('propagation_node', anonymous=True)

    # get defined simulation parameters
    sim_parameter = dict()
    if rospy.has_param("~frequency"):
        sim_parameter['frequency'] = int(rospy.get_param("~frequency"))
    if rospy.has_param("~oe_epoch"):
        sim_parameter['oe_epoch'] = str(rospy.get_param("~oe_epoch"))

    SimTime = EpochClock(**sim_parameter)

    rospy.loginfo("Epoch of init. Orbit Elements = " +
                  SimTime.datetime_oe_epoch.strftime("%Y-%m-%d %H:%M:%S"))
    rospy.loginfo("Realtime Factor = " + str(SimTime.realtime_factor))

    # set publish frequency & time step size as ros parameter
    # do this before setting /epoch parameter to ensure epoch_clock library
    # can find them
    rospy.set_param('/publish_freq', SimTime.frequency)
    rospy.set_param('/time_step_size', SimTime.step_size)
    rospy.set_param('/epoch',
                    SimTime.datetime_oe_epoch.strftime("%Y-%m-%d %H:%M:%S"))

    pub_clock = rospy.Publisher('/clock', Clock, queue_size=10)

    # Start GUI clock service on new thread
    ClockServer = ClockServer(SimTime.realtime_factor,
                              SimTime.frequency,
                              SimTime.step_size)

    # Subscribe to propulsion node and attitude control
    thrust_force = message_filters.Subscriber('force', WrenchStamped)
    thrust_ispM = message_filters.Subscriber('IspMean', ThrustIsp)

    # Init publisher and rate limiter
    pub_ch = rospy.Publisher('oe_chaser', SatelitePose, queue_size=10)
    pub_pose_ch = rospy.Publisher('pose_chaser', PoseStamped, queue_size=10)

    pub_ta = rospy.Publisher('oe_target', SatelitePose, queue_size=10)
    pub_pose_ta = rospy.Publisher('pose_target', PoseStamped, queue_size=10)

    [init_state_ch, init_state_ta] = get_init_state_from_param()

    OrekitPropagator.init_jvm()
    prop_chaser = OrekitPropagator()
    # get settings from yaml file
    propSettings = rospy.get_param("/chaser/propagator_settings", 0)
    prop_chaser.initialize(propSettings,
                           init_state_ch,
                           SimTime.datetime_oe_epoch)

    # add callback to thrust function
    Tsync = message_filters.TimeSynchronizer([thrust_force, thrust_ispM], 10)
    Tsync.registerCallback(prop_chaser.add_thrust_callback)

    prop_target = OrekitPropagator()
    # get settings from yaml file
    propSettings = rospy.get_param("/target/propagator_settings", 0)
    prop_target.initialize(propSettings,
                           init_state_ta,
                           SimTime.datetime_oe_epoch)

    rospy.loginfo("Propagators initialized!")

    # Update first step so that other nodes don't give errors
    msg_cl = Clock()
    SimTime.updateClock(msg_cl)
    pub_clock.publish(msg_cl)
    # init epoch clock for propagator
    epoch = rospace_lib.clock.Epoch()

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
            SimTime.updateClock(msg_cl)
            pub_clock.publish(msg_cl)

        epoch_now = epoch.now()
        rospy_now = rospy.Time.now()

        # propagate to epoch_now
        [cart_ch, att_ch] = prop_chaser.propagate(epoch_now)
        [cart_t, att_t] = prop_target.propagate(epoch_now)

        [msg_ch, msg_pose_ch] = cart_to_msgs(cart_ch, att_ch, rospy_now)
        [msg_t, msg_pose_t] = cart_to_msgs(cart_t, att_t, rospy_now)

        pub_ch.publish(msg_ch)
        pub_pose_ch.publish(msg_pose_ch)

        pub_ta.publish(msg_t)
        pub_pose_ta.publish(msg_pose_t)

        # calculate reminding sleeping time
        sleep_time = SimTime.rate - (time.clock() - comp_time)
        # Improve sleep time as Gazebo for more accurate time update
        if sleep_time > 0:
            sleep(sleep_time)
        elif ClockServer.syncSubscribers == 0:
            rospy.logwarn("Propagator too slow for publishing rate.")
