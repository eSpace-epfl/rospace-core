# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import rospace_lib
import numpy as np
import PropagatorBuilder as PB
import os

import orekit

from java.io import File

from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.data import DataProvidersManager, ZipJarCrawler
from org.orekit.time import TimeScalesFactory, AbsoluteDate

from org.hipparchus.geometry.euclidean.threed import Vector3D


def write_satellite_state(state):
    """
    Method to extract satellite orbit state vector from Java object and put it
    in a numpy array.

    Args:
        PVCoordinates: satellite state vector

    Returns:
        numpy.array: cartesian state vector in TEME frame
        numpy.array: current attitude rotation in quaternions
    """

    pv = state.getPVCoordinates()
    cart_teme = rospace_lib.CartesianTEME()
    cart_teme.R = np.array([pv.position.x,
                            pv.position.y,
                            pv.position.z]) / 1000
    cart_teme.V = np.array([pv.velocity.x,
                            pv.velocity.y,
                            pv.velocity.z]) / 1000

    # if hasAttitudeProp:
    rot_ch = state.getAttitude().getRotation()
    att = np.array([rot_ch.q1,
                    rot_ch.q2,
                    rot_ch.q3,
                    rot_ch.q0])

    return [cart_teme, att]


class OrekitPropagator(object):
    """
    Class building numerical propagator using the orekit library methods.
    """

    @staticmethod
    def init_jvm():
        """
        Initialize Java virtual machine for Orekit.

        This method also loads the orekit-data.zip from the pythonpath or
        form the current directory and sets up the orekit DataProviers to
        access it (how to do so has been copied from setup_orekit_curdir()).
        """

        orekit.initVM()

        path_to_file = None
        # search for file on pythonpath
        for paths in os.environ['PYTHONPATH'].split(os.pathsep):
            for root, dirs, files in os.walk(paths):
                if 'orekit-data.zip' in files:
                    path_to_file = os.path.join(root, 'orekit-data.zip')

        if path_to_file is None:
            # desperate search on current directory
            setup_orekit_curdir()
        else:
            DM = DataProvidersManager.getInstance()
            datafile = File(path_to_file)
            crawler = ZipJarCrawler(datafile)
            DM.clearProviders()
            DM.addProvider(crawler)

    def __init__(self):
        self._propagator_num = None

    def initialize(self, propSettings, state, epoch):
        """
        Method builds propagator object based on settings defined in arguments.

        Propagator object is build using PropagatorBuilder. This takes
        information from a python dictionary to add and build propagator
        settings correctly.

        After build this method checks if Attitude propagation and Thrusting
        has been set as active. If yes the following object variables are set
        to True:
            - hasAttitudeProp
            - hasThrust

        Args:
            propSettings: dictionary containing info about propagator settings
            state: initial state of spacecraft
            epoch: initial epoch @ which state is defined
        """

        OrEpoch = self.to_orekit_date(epoch)

        _builder = PB.PropagatorBuilder(propSettings, state, OrEpoch)
        _builder._build_state()
        _builder._build_integrator()
        _builder._build_propagator()
        _builder._build_gravity()
        _builder._build_attitude_propagation()
        _builder._build_thirdBody()
        _builder._build_drag_and_solar_pressure()

        _builder._build_solid_tides()
        _builder._build_ocean_tides()
        _builder._build_relativity()
        _builder._build_thrust()

        self._earth = _builder.get_earth()
        self.ThrustModel = _builder.get_thrust_model()
        self._propagator_num = _builder.get_propagator()

        if propSettings['attitudeProvider']['type'] == 'AttPropagation':
            self.hasAttitudeProp = True
        else:
            self.hasAttitudeProp = False

        if self.ThrustModel is not None:
            self.hasThrust = True
        else:
            self.hasThrust = False

    def propagate(self, epoch):
        """
        Propagate satellite to given epoch.

        Method calculates if external forces and torques are acting on
        satellite (Thrust), then propagates its state.

        The newly propagated state is set as the satellite's new initial state.

        Args:
            epoch: epoch to which propagator has to propagate

        Returns:
            numpy.array: cartesian state vector in TEME frame
            numpy.array: current attitude rotation in quaternions
        """

        orekit_date = self.to_orekit_date(epoch)

        if self.hasAttitudeProp:
            self.calculate_torque()

        if self.hasThrust:
            # self.change_attitude_provider()
            self.calculate_thrust()

        # Place where all the magic happens:
        state = self._propagator_num.propagate(orekit_date)

        # return new state in numpy arrays
        return write_satellite_state(state)

    def to_orekit_date(self, epoch):
        """
        Method to convert UTC simulation time from python's datetime object to
        orekits AbsoluteDate object

        Args:
            epoch: UTC simulation time as datetime object

        Returns:
            AbsoluteDate: simulation time in UTC
        """
        seconds = float(epoch.second) + float(epoch.microsecond) / 1e6
        orekit_date = AbsoluteDate(epoch.year,
                                   epoch.month,
                                   epoch.day,
                                   epoch.hour,
                                   epoch.minute,
                                   seconds,
                                   TimeScalesFactory.getUTC())
        return orekit_date

    def calculate_thrust(self):
        """
        Method that updates parameter in Thrust Model.

        Method checks if first message from propulsion node received (stored in
         self.F_T), then computes the norm of the thrust vector and updates the
        parameters in the Thrust force model.

        If Thrust is zero, the contribution of the force model is not
        calculated.
        """

        # if simulation hasn't started attributes don't exist yet
        # set them to None in this case
        F_T = getattr(self, 'F_T', None)

        if F_T is not None:
            F_T_norm = np.linalg.norm(self.F_T, 2)

            if F_T_norm > 2 * np.finfo(np.float32).eps:
                # print self.F_T
                F_T_dir = self.F_T / F_T_norm
                F_T_dir = Vector3D(float(F_T_dir[0]),
                                   float(F_T_dir[1]),
                                   float(F_T_dir[2]))

                self.ThrustModel.direction = F_T_dir
                self.ThrustModel.thrust = float(F_T_norm)
                self.ThrustModel.isp = self.Isp
                self.ThrustModel.ChangeParameters(float(F_T_norm), self.Isp)
                self.ThrustModel.firing = True
            else:
                self.ThrustModel.firing = False

    def add_thrust_callback(self, thrust_force, thrust_ispM):
        """
        Callback function for subscriber to the propulsion node.

        Function saves the vlaue for the mean specific impulse and
        the force, torque vector to class objects.

        If calback used it is necessary that the propulsion node publishes
        the  current thrust force at every timestep (even if it is zero)!

        Messages for thrust/torque and specific impulse are synchronized.

        Args:
            thrust_force: WrenchStamped message cotaining torque and force
            thrust_ispM: ThrustIsp message with specific impulse of maneuver
        """

        # this currently only called in by chaser object
        self.F_T = np.array([thrust_force.wrench.force.x,
                             thrust_force.wrench.force.y,
                             thrust_force.wrench.force.z])

        self.Isp = thrust_ispM.Isp_val

        # Torque:
        self.torque = np.array([thrust_force.wrench.torque.x,
                                thrust_force.wrench.torque.y,
                                thrust_force.wrench.torque.z])
