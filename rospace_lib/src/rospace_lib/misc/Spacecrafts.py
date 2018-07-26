# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import rospy
import numpy as np
import message_filters
from datetime import datetime
from copy import deepcopy

from propagator.OrekitPropagator import OrekitPropagator
from rospace_lib.misc.FileDataHandler import FileDataHandler, to_datetime_date, to_orekit_date
from rospace_lib import Cartesian, CartesianTEME, CartesianLVLH, OscKepOrbElem, KepOrbElem

from rospace_msgs.msg import PoseVelocityStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3Stamped
from rospace_msgs.msg import ThrustIsp
from rospace_msgs.msg import SatelitePose
from rospace_msgs.msg import SatelliteTorque

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst


class Spacecraft(object):
    """Class holding a object for every spacecraft that is being propagated."""

    @property
    def mass(self):
        """Return mass stored in propagator.

        Return:
            float: Current mass of spacecraft

        """
        return self._propagator.getInitialState().getMass()

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
        # self._last_state = [None, None, None, None, None]
        self.cartesian_pose = None
        self.current_attitude = None
        self.acting_force_ = None

        self._parsed_settings = {}
        self._parsed_settings["init_coords"] = {}
        self._parsed_settings["prop_settings"] = {}

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


class Simulator_Spacecraft(Spacecraft):
    """Spacecraft object for the simulator.

    This object can be used for propagation and the publishing of messages
    containing the propagation output.

    The object holds its own build propagator object as well as all publishes and
    subscribers with correctly defined topics, so that no mix-up between spacecrafts
    can occur.

    Args:
        namespace (string): name of spacecraft (namespace in which it is defined)

    Attributes:
        namespace (string): name of spacecraft (namespace in which it is defined)

    """

    def __init__(self, namespace):
        super(Simulator_Spacecraft, self).__init__(namespace)

        self._pub_oe = None
        self._pub_pose = None
        self._pub_dtorq = None
        self._pub_ft = None
        self._pub_bfield = None

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


class Planning_Spacecraft(Spacecraft):

    @property
    def abs_state(self):
        return self._last_state[0]
        # state_frame = self._propagator._propagator_num.getInitialState().getFrame()
        # pv = self._propagator._propagator_num.getInitialState().getPVCoordinates()
        # if state_frame.toString() == "TEME":
        #     cart = CartesianTEME()
        # else:
        #     cart = Cartesian()

        # cart.R = np.array([pv.position.x, pv.position.y, pv.position.z]) / 1e3
        # cart.V = np.array([pv.velocity.x, pv.velocity.y, pv.velocity.z]) / 1e3
        # return cart

    @property
    def mass(self):
        return self._propagator._propagator_num.getInitialState().getMass()

    @property
    def date(self):
        abs_date = self._propagator._propagator_num.getInitialState().getDate()
        return to_datetime_date(abs_date)

    def __init__(self, namespace, chaser=False):
        super(Planning_Spacecraft, self).__init__(namespace)

        self.prop_type = None

        if chaser:
            self.rel_state = CartesianLVLH()

    def build_propagator(self, epoch_now, prop_type, parent_state=None):
        """Intermediate class for building propagator. Turns of pertub. for 2-body"
        """
        self.prop_type = prop_type

        if self.prop_type == "2-body":
            mesg = "\033[93m[" + self.namespace + \
                "]: Two Body Propagator is being build. Turning of all perturbations and attitude" + \
                " propagation.\033[0m"
            print mesg

            orb_prop_dict = self._parsed_settings["prop_settings"]["orbit_propagation"]
            att_prop_dict = self._parsed_settings["prop_settings"]["attitudeProvider"]

            orb_prop_dict["Gravity"]["type"] = ""
            orb_prop_dict["SolarModel"]["type"] = ""
            orb_prop_dict["DragModel"]["type"] = ""
            orb_prop_dict["Thrust"]["type"] = ""
            orb_prop_dict["ThirdBody"]["Sun"] = False
            orb_prop_dict["ThirdBody"]["Moon"] = False
            orb_prop_dict["SolidTides"]["add"] = False
            orb_prop_dict["OceanTides"]["add"] = False
            orb_prop_dict["addRelativity"] = False

            att_prop_dict["type"] = ""

            self._parsed_settings["prop_settings"]["orbit_propagation"] = orb_prop_dict

        # build propagator
        super(Planning_Spacecraft, self).build_propagator(epoch_now)

        # ugly_hack 2.0: set self._last_state:
        super(Planning_Spacecraft, self).propagate(epoch_now)

        # set relative state if chaser-spacecraft
        if hasattr(self, "rel_state"):
            if parent_state is not None:
                try:  # check if states are in same frame
                    assert (self.abs_state.frame == parent_state.frame)
                except AssertionError:
                    raise TypeError("State of " + self.namespace + "is given in frame: " + self.abs_state.frame +
                                    " and parent state is given in frame: " + parent_state.frame + "." +
                                    "Could not compute relative state!")
                # in case of undefined frames print warning that frames could be different
                if parent_state.frame == "UNDEF":
                    mesg = "\033[93m[WARN]: Spacecrafts states have undefined frames. " + \
                        "Cannot determine if states are defined in the same frame!\033[0m"
                    print mesg

                self.rel_state.from_cartesian_pair(self.abs_state, parent_state)
            else:
                raise IOError('Missing target input to initialize relative state!')

    def change_initial_conditions(self, new_state, new_epoch, new_mass):
        """Change the initial conditions given to the propagator without initializing it again.

        Args:
            initial_state (Cartesian): New initial state of the satellite in cartesian coordinates.
            epoch (datetime.datetime): New starting date of the propagator.
            mass (float64): New satellite mass.

        """
        # Create position and velocity vectors as Vector3D
        p = Vector3D(1e3,  # convert to [m]
                     Vector3D(float(new_state.R[0]),
                              float(new_state.R[1]),
                              float(new_state.R[2])))
        v = Vector3D(1e3,  # convert to [m/s]
                     Vector3D(float(new_state.V[0]),
                              float(new_state.V[1]),
                              float(new_state.V[2])))

        # Extract frame from initial/pre-propagated state
        initState = self._propagator._propagator_num.getInitialState()

        orekit_date = to_orekit_date(new_epoch)
        # Evaluate new initial orbit
        # print initState.getOrbit().getType()
        # print initState.getOrbit().getType().toString() == "CARTESIAN"
        initialOrbit = CartesianOrbit(PVCoordinates(p, v), initState.getFrame(), orekit_date, Cst.WGS84_EARTH_MU)

        # Create new spacecraft state
        newSpacecraftState = SpacecraftState(initialOrbit, new_mass)

        # Rewrite propagator initial conditions
        self._propagator._propagator_num.setInitialState(newSpacecraftState)

        # ugly hack for now
        self._last_state[0] = new_state

    def get_osc_oe(self):
        """Return the osculating orbital elements of the spacecraft.

        Returns:
              kep_osc (OscKepOrbElem): Osculating orbital elements.

        """
        kep_osc = OscKepOrbElem()
        kep_osc.from_cartesian(self.abs_state)

        return kep_osc

    def get_mean_oe(self):
        """Return mean orbital elements of the satellite.

        Returns:
            kep_mean (KepOrbElem): Mean orbital elements.

        """
        kep_osc = self.get_osc_oe()

        kep_mean = KepOrbElem()

        if self.prop_type == "real-world":
            kep_mean.from_osc_elems(kep_osc)
        elif self.prop_type == "2-body":
            kep_mean.from_osc_elems(kep_osc, "null")
        else:
            raise TypeError("Propagator type not recognized!")

        return kep_mean

    def set_abs_state_from_target(self, target):
        """Set absolute state given target absolute state and chaser relative state.

        Args:
             target (Satellite): State of the target.

        """
        if hasattr(self, "rel_state"):
            # ugly hack
            tmp = deepcopy(self.abs_state)
            tmp.from_lvlh_frame(target.abs_state, self.rel_state)
            self._last_state[0] = deepcopy(tmp)
        else:
            raise AttributeError("Spacecraft " + self.namespace +
                                 " is not defined as chaser! Cannot set absolute state from target state!")


#############################################################################################
# Write Objects to ROS-messages
#############################################################################################


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
    oe = KepOrbElem()
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
#############################################################################################
