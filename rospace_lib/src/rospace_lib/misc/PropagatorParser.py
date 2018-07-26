#!/usr/bin/env python

# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import rospy

import sys
import os
import yaml
import numpy as np
from math import radians

from rospace_lib import Cartesian, CartesianTEME, KepOrbElem
from rospace_lib.misc.FileDataHandler import to_orekit_date

from org.orekit.frames import FramesFactory
from org.orekit.utils import PVCoordinates
from org.orekit.utils import IERSConventions as IERS

from org.hipparchus.geometry.euclidean.threed import Vector3D


def parse_configuration_files(spc_obj, init_coords, init_epoch):

    ns_spacecraft = spc_obj.namespace

    if spc_obj.__class__.__name__ == "Simulator_Spacecraft":
        spc_obj.propagator_settings = rospy.get_param("/" + ns_spacecraft + "/propagator_settings", 0)
        try:
            assert (spc_obj.propagator_settings != 0)
        except AssertionError:
            raise AssertionError("Could not find propagator settings." +
                                 "Check if all spacecraft names-spaces correctly defined in Scenario!")

    elif spc_obj.__class__.__name__ == "Planning_Spacecraft":
        # Opening spacecraft file
        abs_path = os.path.dirname(os.path.abspath(__file__))
        scenario_path = os.path.join(
            abs_path, "../../../../rospace_simulator/cfg/Spacecrafts/" + ns_spacecraft + ".yaml")
        scenario_file = file(scenario_path, "r")
        spc_obj.propagator_settings = yaml.load(scenario_file)["propagator_settings"]

    # get init oe values
    spc_dict = {}
    pose_method = getattr(sys.modules[__name__], "_pose_" + init_coords["type"] + "_" + init_coords["coord_type"])
    spc_dict["position"] = pose_method(ns_spacecraft, init_coords, init_epoch)

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
def _pose_absolute_keplerian(ns_spacecraft, init_coords, init_epoch):
    pos = init_coords["init_coord"]
    init_state = Cartesian()

    init_pose_oe = KepOrbElem()
    init_pose_oe.a = float(pos["a"])
    init_pose_oe.e = float(pos["e"])
    init_pose_oe.i = float(radians(pos["i"]))
    init_pose_oe.O = float(radians(pos["O"]))
    init_pose_oe.w = float(radians(pos["w"]))
    try:
        init_pose_oe.v = float(radians(pos["v"]))
    except KeyError:
        try:
            init_pose_oe.m = float(radians(pos["m"]))
        except KeyError:
            raise ValueError("No Anomaly for initialization of spacecraft: " + ns_spacecraft)

    # Coordinates given in J2000 Frame
    # ----------------------------------------------------------------------------------------
    if init_coords["coord_frame"] == "J2000":
        init_state.from_keporb(init_pose_oe)
    # ----------------------------------------------------------------------------------------

    if init_coords["coord_frame"] == "TEME":
        teme_state = CartesianTEME()
        teme_state.from_keporb(init_pose_oe)

        init_frame = FramesFactory.getTEME()
        inertialFrame = FramesFactory.getEME2000()

        TEME2EME = init_frame.getTransformTo(inertialFrame, to_orekit_date(init_epoch))
        p = Vector3D(float(1e3),  # convert to [m]
                     Vector3D(float(teme_state.R[0]),
                              float(teme_state.R[1]),
                              float(teme_state.R[2])))
        v = Vector3D(float(1e3),  # convert to [m/s]
                     Vector3D(float(teme_state.V[0]),
                              float(teme_state.V[1]),
                              float(teme_state.V[2])))

        pv_EME = TEME2EME.transformPVCoordinates(PVCoordinates(p, v))

        init_state.R = np.array([pv_EME.position.x, pv_EME.position.y, pv_EME.position.z]) / 1e3  # convert to [km]
        init_state.V = np.array([pv_EME.velocity.x, pv_EME.velocity.y, pv_EME.velocity.z]) / 1e3  # convert to [km/s]

    # ----------------------------------------------------------------------------------------

    return init_state


def _pose_absolute_cartesian(ns_spacecraft, init_coords, init_epoch):
    pos = init_coords["init_coord"]
    init_state = Cartesian()

    # Coordinates given in J2000 Frame
    # ----------------------------------------------------------------------------------------
    if init_coords["coord_frame"] == "J2000":
        init_state.R = np.array([float(pos["x"]), float(pos["y"]), float(pos["z"])])

        init_state.V = np.array([float(pos["v_x"]), float(pos["v_y"]), float(pos["v_z"])])
    # ----------------------------------------------------------------------------------------

    # Coordinates given in ITRF Frame
    # ----------------------------------------------------------------------------------------
    elif init_coords["coord_frame"] == "ITRF":
        inertialFrame = FramesFactory.getEME2000()
        init_frame = FramesFactory.getITRF(IERS.IERS_2010, False)  # False -> don't ignore tidal effects
        p = Vector3D(float(1e3),  # convert to [m]
                     Vector3D(float(pos["x"]),
                              float(pos["y"]),
                              float(pos["z"])))
        v = Vector3D(float(1e3),  # convert to [m/s]
                     Vector3D(float(pos["v_x"]),
                              float(pos["v_y"]),
                              float(pos["v_z"])))

        ITRF2EME = init_frame.getTransformTo(inertialFrame, to_orekit_date(init_epoch))
        pv_EME = ITRF2EME.transformPVCoordinates(PVCoordinates(p, v))

        init_state.R = np.array([pv_EME.position.x, pv_EME.position.y, pv_EME.position.z]) / 1e3  # convert to [km]
        init_state.V = np.array([pv_EME.velocity.x, pv_EME.velocity.y, pv_EME.velocity.z]) / 1e3  # convert to [km/s]

    else:
        raise ValueError("[" + ns_spacecraft + " ]: " + "Conversion from coordinate frame " +
                         init_coords["coord_frame"] +
                         " not implemented. Please provided coordinates in a different reference frame.")
    # ----------------------------------------------------------------------------------------

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


########################################################################################################################

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

# class KeplerianEME2000(StateFactory):

#     @staticmethod
#     def isApplicable(name):

#         if name == "KeplerianEME2000":
#             return True
#         else:
#             return False

#     @staticmethod
#     def Setup(epoch, earth, state, setup):
#         """
#         Create initial spacecraft state and orbit based on Keplerian elements.

#         Args:
#             epoch: initial epoch or orbital elements
#             state: initial state of satellite
#             setup: additional settings defined in dictionary

#         Returns:
#             inertialFrame: EME2000 as inertial Frame of Orbit
#             initialOrbit: Keplerian orbit
#             initialState: Spacecraft state
#         """

#         satMass = setup['mass']

#         a = float(state.a)
#         e = float(state.e)
#         i = float(state.i)
#         w = float(state.w)
#         O = float(state.O)
#         v = float(state.v)

#         # Inertial frame where the satellite is defined (and earth)
#         inertialFrame = FramesFactory.getEME2000()

#         initialOrbit = KeplerianOrbit(a*1000, e, i, w, O, v,
#                                       PositionAngle.TRUE,
#                                       inertialFrame,
#                                       epoch,
#                                       Cst.WGS84_EARTH_MU)

#         orbit_pv = PVCoordinatesProvider.cast_(initialOrbit)
#         satAtt = _build_satellite_attitude(setup, orbit_pv, inertialFrame,
#                                            earth, epoch)

#         initialState = SpacecraftState(initialOrbit, satAtt, satMass)

#         return [inertialFrame, initialOrbit, initialState]


# class CartesianITRF(StateFactory):

#     @staticmethod
#     def isApplicable(name):

#         if name == "CartesianITRF":
#             return True
#         else:
#             return False

#     @staticmethod
#     def Setup(epoch, earth, state, setup):
#         """
#         Create initial spacecraft state and orbit using PV-Coordinates in ITRF2008 Frame.

#         Args:
#             epoch: initial epoch or orbital elements
#             state: initial state of satellite [Position, Velocity]
#             setup: additional settings defined in dictionary

#         Returns:
#             inertialFrame: EME2000 as inertial Frame of Orbit
#             initialOrbit: Cartesian orbit
#             initialState: Spacecraft state
#         """

#         satMass = setup['mass']

#         p = Vector3D(float(state.R[0]),
#                      float(state.R[1]),
#                      float(state.R[2]))
#         v = Vector3D(float(state.V[0]),
#                      float(state.V[1]),
#                      float(state.V[2]))

#         # Inertial frame where the satellite is defined (and earth)
#         inertialFrame = FramesFactory.getEME2000()
#         # False bool -> don't ignore tidal effects
#         orbitFrame = FramesFactory.getITRF(IERS.IERS_2010, False)
#         ITRF2EME = orbitFrame.getTransformTo(inertialFrame, epoch)
#         pv_EME = ITRF2EME.transformPVCoordinates(PVCoordinates(p, v))

#         initialOrbit = CartesianOrbit(pv_EME,
#                                       inertialFrame,
#                                       epoch,
#                                       Cst.WGS84_EARTH_MU)

#         orbit_pv = PVCoordinatesProvider.cast_(initialOrbit)
#         satAtt = _build_satellite_attitude(setup, orbit_pv, inertialFrame,
#                                            earth, epoch)

#         initialState = SpacecraftState(initialOrbit, satAtt, satMass)

#         return [inertialFrame, initialOrbit, initialState]
