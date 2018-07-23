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

import PropagatorParser

from OrekitPropagator import OrekitPropagator
from FileDataHandler import FileDataHandler
from rospace_msgs.msg import PoseVelocityStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3Stamped
from rospace_msgs.msg import ThrustIsp
from rospace_msgs.msg import SatelitePose
from rospace_msgs.msg import SatelliteTorque
from std_srvs.srv import Empty


class Spacecraft(object):
    """Class holding a object for every spacecraft that is being propagated.

    This object is being used for propagation and the publishing of messages
    containing the propagation output.

    The object holds its own build propagator object as well as all publishes and
    subscribers with correctly defined topics, so that no mix-up between spacecrafts
    can occur

    Args:
        namespace (string): name of spacecraft (namespace in which it is defined)

    Attributes:
        namespace (string): name of spacecraft (namespace in which it is defined)

    """

    @property
    def propagator_settings(self):
        """Return parsed propagator settings.

        Return:
            dict: Propagator settings used by :class:`propagator.PropagatorBuilder.PropagatorBuilder`

        """
        return self._parsed_settings["prop_settings"]

    @property
    def init_coords(self):
        """Return parsed initial coordinates.

        Return:
            dict: Initial coordinates used by :class:`propagator.PropagatorBuilder.PropagatorBuilder`

        """
        return self._parsed_settings["init_coords"]

    @propagator_settings.setter
    def propagator_settings(self, sett):
        self._parsed_settings["prop_settings"] = sett

    @init_coords.setter
    def init_coords(self, coords):
        self._parsed_settings["init_coords"] = coords

    def __init__(self, namespace):

        self.namespace = namespace
        self._propagator = None
        self._last_state = None

        self._parsed_settings = {}
        self._parsed_settings["init_coords"] = {}
        self._parsed_settings["prop_settings"] = {}

    def build_communication(self):
        """Create Publishers and Subscribers, which are linked to the correct topics in the spacecrafts namespace.

        This method has to be called after the propagator object was build using :func:`build_propagator`.

        Raises:
            RuntimeError: If method is called even though no propagator object has been provided

        """
        self._pub_oe = rospy.Publisher("/" + self.namespace + "/oe", SatelitePose, queue_size=10)
        self._pub_pose = rospy.Publisher("/" + self.namespace + "/pose", PoseVelocityStamped, queue_size=10)
        self._pub_dtorq = rospy.Publisher("/" + self.namespace + "/dist_torque", SatelliteTorque, queue_size=10)
        self._pub_ft = rospy.Publisher("/" + self.namespace + "/force_torque", WrenchStamped, queue_size=10)
        self._pub_bfield = rospy.Publisher("/" + self.namespace + "/B_field", Vector3Stamped, queue_size=10)

        try:
            assert (self._propagator is not None)
        except AssertionError:
            raise RuntimeError("[" + self.namespace + "] " +
                               "Error: Propagator object has to be build before calling this method!")

        if self._propagator._hasThrust:
            external_force_ch = message_filters.Subscriber("/" + self.namespace + "/thrust_force", WrenchStamped)
            thrust_ispM_ch = message_filters.Subscriber("/" + self.namespace + '/isp_mean', ThrustIsp)
            self._thrust_sub = message_filters.TimeSynchronizer([external_force_ch, thrust_ispM_ch], 10)
            self._thrust_sub.registerCallback(self._propagator.thrust_callback)
        if self._propagator._hasAttitudeProp:
            self._ext_torque_sub = rospy.Subscriber("/" + self.namespace + "/actuator_torque", WrenchStamped,
                                                    self._propagator.magnetorque_callback)

    def build_propagator(self, init_epoch):
        """Build the Orekit-Propagator object for the spacecraft.

        Args:
            init_epoch (datetime.datetime): initial epoch in which initial coordinates are defined

        """
        self._propagator = OrekitPropagator()
        self._propagator.initialize(self.namespace,
                                    self._parsed_settings["prop_settings"],
                                    self._parsed_settings["init_coords"],
                                    init_epoch)

    def propagate(self, epoch_now):
        """Propagate spacecraft to epoch_now.

        Args:
            epoch_now(datetime.datetime): time to which spacecraft will be propagated
        """
        self._last_state = self._propagator.propagate(epoch_now)

    def publish(self):
        """Publish all spacecraft related messages.

        Following messages are published:
            - Keplerian Elements
            - Pose (Position, Velocity, Attitude, Spin)
            - Disturbance Torques acting on spacecraft
            - Local magnetic field in spacecraft body frame

        """
        rospy_now = rospy.Time.now()
        # last_state is stored as: [cart, att, force, d_torque, B_field]
        [msg_oe, msg_pose] = cart_to_msgs(self._last_state[0], self._last_state[1], rospy_now)
        msg_B_field = Bfield_to_msgs(self._last_state[4], rospy_now)
        [msg_ft, msg_d_torque] = force_torque_to_msgs(self._last_state[2],
                                                      self._last_state[3],
                                                      rospy_now)

        self._pub_oe.publish(msg_oe)
        self._pub_pose.publish(msg_pose)
        self._pub_dtorq.publish(msg_d_torque)
        self._pub_ft.publish(msg_ft)
        self._pub_bfield.publish(msg_B_field)


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


def cart_to_msgs(cart, att, time):
    """Packs Cartesian orbit elements to message.

    Args:
        cart (:obj:`rospace_lib.Cartesian`): orbit state vector
        att (orekit.Attitude): satellite attitude in quaternions
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
        force (numpy.array): force vector acting on satellite in body frame
        torque (:obj:`OrekitPropagator.DisturbanceTorqueStorage`: current torques acting on satellite
        time (:obj:`rospy.Time`): time stamp

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
        bfield (Vector3D): local magnetic field vector in satellite frame
        time (:obj:`rospy.Time`): time stamp

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

    OrekitPropagator.init_jvm()

    # Initialize Data handlers, loading data in orekit.zip file
    FileDataHandler.load_magnetic_field_models(SimTime.datetime_oe_epoch)

    spacecrafts = []  # List of to be propagated spacecrafts
    sc_settings = rospy.get_param("scenario/init_coords")
    for ns_spacecraft, init_coords in sc_settings.items():
        # Parse settings for every spacecraft independently
        spc = PropagatorParser.parse_configuration_files(Spacecraft(ns_spacecraft),
                                                         ns_spacecraft,
                                                         init_coords)

        # Build propagator object from settings
        spc.build_propagator(SimTime.datetime_oe_epoch)

        # Set up publishers and subscribers
        spc.build_communication()

        spacecrafts.append(spc)

    FileDataHandler.create_data_validity_checklist()

    rospy.loginfo("Propagators initialized!")

    while not rospy.is_shutdown() and not ExitServer.exiting:
        comp_time = time.clock()

        epoch_now = SimTime.update_simulation_time()
        if SimTime.time_shift_passed:
            # check if data still loaded
            FileDataHandler.check_data_availability(epoch_now)

            for spc in spacecrafts:
                try:
                    # propagate to epoch_now
                    spc.propagate(epoch_now)
                except Exception as e:
                    print "ERROR in propagation of: [", spc.namespace, "]"
                    print e.message, e.args
                    print "Shutting down Propagator!"
                    ExitServer.exiting = True

            for spc in spacecrafts:
                # Publish messages
                spc.publish()

        # Maintain correct frequency
        SimTime.sleep_to_keep_frequency()
