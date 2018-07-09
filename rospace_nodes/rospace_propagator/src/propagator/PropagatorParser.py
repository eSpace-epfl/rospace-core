#!/usr/bin/env python

# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import rospy
import rospace_lib

import sys
import numpy as np
from math import radians

from OrekitPropagator import OrekitPropagator
from FileDataHandler import FileDataHandler
from rospace_msgs.msg import PoseVelocityStamped
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3Stamped
from rospace_msgs.msg import ThrustIsp
from rospace_msgs.msg import SatelitePose
from rospace_msgs.msg import SatelliteTorque


class SpacecraftCommunicator(object):

    @property
    def prop_settings(self):
        return self._parsed_settings

    def __init__(self, namespace):

        self.ns_spacecraft = namespace

        self._set_up_publishers()

        self._parsed_settings = None

    def _set_up_publishers(self):

        self._pub_oe(rospy.Publisher("/" + self.ns_spacecraft + "/oe", SatelitePose, queue_size=10))
        self._pub_pose(rospy.Publisher("/" + self.ns_spacecraft + "/pose", PoseVelocityStamped, queue_size=10))
        self._pub_dtorq(rospy.Publisher("/" + self.ns_spacecraft + "/dist_torque", SatelliteTorque, queue_size=10))
        self._pub_ft(rospy.Publisher("/" + self.ns_spacecraft + "/force_torque", WrenchStamped, queue_size=10))
        self._pub_bfield(rospy.Publisher("/" + self.ns_spacecraft + "/B_field", Vector3Stamped, queue_size=10))

    def publish_messages(self, oe_msg, pose_msg, dtorque_msg, ft_msg, bfield_msg):
        pass

    def load_prop_settings(self):
        pass


# class PropagatorParser(object):

#     def __init__(self):
#         # load yaml file into workspace
#         self.sc_settings = rospy.get_param("scenario/init_coords")
#         self.parsed_spacecraft = []

#     def translate(self):

#         for ns_spacecraft, init_coords in self.sc_settings.items():
#             # created dictionary with spacecraft related information
#             spc_dict = {}

#             prop_settings = rospy.get_param("~" + ns_spacecraft + "/propagator_settings", 0)
#             try:
#                 assert (prop_settings != 0)
#             except AssertionError:
#                 raise AssertionError("Could not find propagator settings." +
#                                      "Check if all spacecraft names-spaces correctly defined in Scenario!")

#             spc_dict["prop_settings"] = rospy.get_param("~" + ns_spacecraft + "/propagator_settings", 0)

#             # get init oe values
#             spc_dict["init_coord"] = getattr(self, "_parse_" + init_coords["coord_type"])()

#             self.parsed_spacecraft.append(spc_dict)

#     def _parse_keplerian(self):
#         pass

#     def _parse_cartesian(self):
#         pass

######################################################################################################################


def parse_configuration_files(spc_obj, ns_spacecraft, init_coords):

    ns_spacecraft = spc_obj.namespace
    spc_obj.propagator_settings = rospy.get_param("/" + ns_spacecraft + "/propagator_settings", 0)
    try:
        assert (spc_obj.propagator_settings != 0)
    except AssertionError:
        raise AssertionError("Could not find propagator settings." +
                             "Check if all spacecraft names-spaces correctly defined in Scenario!")

    # get init oe values
    spc_dict = {}
    pose_method = getattr(sys.modules[__name__], "_pose_" + init_coords["type"] + "_" + init_coords["coord_type"])
    spc_dict["position"] = pose_method(ns_spacecraft, init_coords)

    # parse attitude
    att_method = getattr(sys.modules[__name__], "_attitude_" + init_coords["coord_frame_attitude"])
    spc_dict["attitude"] = att_method(init_coords)

    # parse spin
    spin = init_coords["init_spin"]

    spc_dict["spin"] = [radians(float(spin["w_x"])),
                        radians(float(spin["w_y"])),
                        radians(float(spin["w_z"]))]

    # parse rotation acceleration
    rot_acc = init_coords["init_rot_acceleration"]

    spc_dict["rotation_acceleration"] = [radians(float(rot_acc["w_dot_x"])),
                                         radians(float(rot_acc["w_dot_y"])),
                                         radians(float(rot_acc["w_dot_z"]))]

    spc_obj.init_coords = spc_dict

    return spc_obj


########################################################################################################################
# POSITION PARSER
########################################################################################################################
def _pose_absolute_keplerian(ns_spacecraft, init_coords):
    pos = init_coords["init_coord"]

    init_pose_oe = rospace_lib.KepOrbElem()
    init_pose_oe.a = float(pos["a"])
    init_pose_oe.e = float(pos["e"])
    init_pose_oe.i = float(pos["i"])
    init_pose_oe.O = float(radians(pos["O"]))
    init_pose_oe.w = float(radians(pos["w"]))
    try:
        init_pose_oe.v = float(radians(pos["v"]))
    except KeyError:
        try:
            init_pose_oe.m = float(radians(pos["m"]))
        except KeyError:
            raise ValueError("No Anomaly for initialization of spacecraft: " + ns_spacecraft)

    init_state = rospace_lib.Cartesian()
    init_state.from_keporb(init_pose_oe)

    return init_state


def _pose_absolute_cartesian(ns_spacecraft, init_coords):
    pos = init_coords["init_coord"]

    if init_coords["coord_frame"] == "J2000":
        init_state = rospace_lib.Cartesian()
        init_state.R = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])])

        init_state.V = np.array([float(pos["v_x"]), float(pos["v_y"]), float(pos["v_z"])])
    else:
        raise ValueError("[" + ns_spacecraft + " ]: " + "Conversion from coordinate frame " +
                         init_coords["coord_frame"] +
                         " not implemented. Please provided coordinates in a different reference frame")

    return init_state


########################################################################################################################
# ATTITUDE PARSER
########################################################################################################################
def _attitude_J2000(init_coords):
    att = init_coords["init_attitude"]

    if att == "nadir":
        return "nadir"
    else:
        return [float(att["q0"]), float(att["q1"]), float(att["q2"]), float(att["q3"])]


# def get_init_state_from_param():
#     """
#     Method to get orbital elements from parameters.

#     Depending on which parameters defined in launch file different
#     parameters are extracted.

#     Returns:
#         Object: Initial state of chaser
#         Object: Initial state of target
#     """
#     if rospy.has_param("~oe_ch_init/a"):
#         # mean elements for init
#         a = float(rospy.get_param("~oe_ch_init/a"))
#         e = float(rospy.get_param("~oe_ch_init/e"))
#         i = float(rospy.get_param("~oe_ch_init/i"))
#         O = float(rospy.get_param("~oe_ch_init/O"))
#         w = float(rospy.get_param("~oe_ch_init/w"))

#         init_state_ch = rospace_lib.KepOrbElem()
#         init_state_ch.a = a
#         init_state_ch.e = e
#         init_state_ch.i = radians(i)  # inclination
#         init_state_ch.O = radians(O)
#         init_state_ch.w = radians(w)

#         if rospy.has_param("~oe_ch_init/v"):
#             init_state_ch.v = radians(float(rospy.get_param("~oe_ch_init/v")))
#         elif rospy.has_param("~oe_ch_init/m"):
#             init_state_ch.m = radians(float(rospy.get_param("~oe_ch_init/m")))
#         else:
#             raise ValueError("No Anomaly for initialization of chaser")

#         if rospy.get_param("~oe_ta_rel"):  # relative target state
#             qns_init_ta = rospace_lib.QNSRelOrbElements()
#             # a = 0.001
#             qns_init_ta.dA = float(rospy.get_param("~oe_ta_init/ada"))  # / (a*1000.0)
#             qns_init_ta.dL = float(rospy.get_param("~oe_ta_init/adL"))  # / (a*1000.0)
#             qns_init_ta.dEx = float(rospy.get_param("~oe_ta_init/adEx"))  # / (a*1000.0)
#             qns_init_ta.dEy = float(rospy.get_param("~oe_ta_init/adEy"))  # / (a*1000.0)
#             qns_init_ta.dIx = float(rospy.get_param("~oe_ta_init/adIx"))  # / (a*1000.0)
#             qns_init_ta.dIy = float(rospy.get_param("~oe_ta_init/adIy"))  # / (a*1000.0)

#             init_state_ta = rospace_lib.KepOrbElem()
#             init_state_ta.from_qns_relative(qns_init_ta, init_state_ch)

#         else:  # absolute target state
#             a_t = float(rospy.get_param("~oe_ta_init/a"))
#             e_t = float(rospy.get_param("~oe_ta_init/e"))
#             i_t = float(rospy.get_param("~oe_ta_init/i"))
#             O_t = float(rospy.get_param("~oe_ta_init/O"))
#             w_t = float(rospy.get_param("~oe_ta_init/w"))

#             init_state_ta = rospace_lib.KepOrbElem()
#             init_state_ta.a = a_t
#             init_state_ta.e = e_t
#             init_state_ta.i = radians(i_t)
#             init_state_ta.O = radians(O_t)
#             init_state_ta.w = radians(w_t)

#             if rospy.has_param("~oe_ta_init/v"):
#                 init_state_ta.v = radians(float(rospy.get_param("~oe_ta_init/v")))
#             elif rospy.has_param("~oe_ta_init/m"):
#                 init_state_ta.m = radians(float(rospy.get_param("~oe_ta_init/m")))
#             else:
#                 raise ValueError("No Anomaly for initialization of target")

#     elif rospy.has_param("~oe_ch_init/x"):
#         x = float(rospy.get_param("~oe_ch_init/x"))
#         y = float(rospy.get_param("~oe_ch_init/y"))
#         z = float(rospy.get_param("~oe_ch_init/z"))
#         xDot = float(rospy.get_param("~oe_ch_init/xDot"))
#         yDot = float(rospy.get_param("~oe_ch_init/yDot"))
#         zDot = float(rospy.get_param("~oe_ch_init/zDot"))

#         init_state_ch = rospace_lib.CartesianITRF()
#         init_state_ch.R = np.array([x, y, z])
#         init_state_ch.V = np.array([xDot, yDot, zDot])

#         x = float(rospy.get_param("~oe_ta_init/x"))
#         y = float(rospy.get_param("~oe_ta_init/y"))
#         z = float(rospy.get_param("~oe_ta_init/z"))
#         xDot = float(rospy.get_param("~oe_ta_init/xDot"))
#         yDot = float(rospy.get_param("~oe_ta_init/yDot"))
#         zDot = float(rospy.get_param("~oe_ta_init/zDot"))

#         init_state_ta = rospace_lib.CartesianITRF()
#         init_state_ta.R = np.array([x, y, z])
#         init_state_ta.V = np.array([xDot, yDot, zDot])

#     return [init_state_ch, init_state_ta]
