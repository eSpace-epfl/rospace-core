# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import space_tf
import numpy as np
import PropagatorBuilder as PB
import os

import orekit

from java.io import File

from org.orekit.python import PythonAttitudePropagation as PAP
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.data import DataProvidersManager, ZipJarCrawler
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.attitudes import Attitude, FixedRate, InertialProvider

from org.hipparchus.geometry.euclidean.threed import Vector3D


class DisturbanceTorques(object):
    def __init__(self):
        self._add = []
        self._dtorque = []

    @property
    def add(self):
        return self._add

    @add.setter
    def add(self, addedDTorques):
        self._add = []
        for added in addedDTorques:
            self._add.append(added)

    @property
    def dtorque(self):
        return self._dtorque

    @dtorque.setter
    def dtorque(self, new_dtorques):
        self._dtorque = []
        for vector in new_dtorques:
            t_array = np.array([vector.getX(),
                               vector.getY(),
                               vector.getZ()])
            self._dtorque.append(t_array)


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

    @staticmethod
    def to_orekit_date(epoch):
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

    @staticmethod
    def write_satellite_state(state):
        """
        Method to extract satellite orbit state vector from Java object and put
        it in a numpy array.

        Args:
            PVCoordinates: satellite state vector

        Returns:
            numpy.array: cartesian state vector in TEME frame
            numpy.array: current attitude rotation in quaternions
        """

        pv = state.getPVCoordinates()
        cart_teme = space_tf.CartesianTEME()
        cart_teme.R = np.array([pv.position.x,
                                pv.position.y,
                                pv.position.z]) / 1000
        cart_teme.V = np.array([pv.velocity.x,
                                pv.velocity.y,
                                pv.velocity.z]) / 1000

        rot_ch = state.getAttitude().getRotation()
        att = np.array([rot_ch.q1,
                        rot_ch.q2,
                        rot_ch.q3,
                        rot_ch.q0])

        return [cart_teme, att]

    def __init__(self):
        self._propagator_num = None
        self._hasAttitudeProp = False
        self._hasThrust = False

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

        Raises:
            AssertionError: if propagator settings file was not found
        """

        OrEpoch = self.to_orekit_date(epoch)

        try:
            assert propSettings != 0
        except AssertionError:
            print "ERROR: Propagator settings file could not be found!"
            raise

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
            self._hasAttitudeProp = True
            self.attProv = PAP.cast_(self._propagator_num.getAttitudeProvider())

        if self.ThrustModel is not None:
            self._hasThrust = True

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

        if self._hasAttitudeProp:
            self.calculate_external_torque()

        if self._hasThrust:
            self.calculate_thrust()

        # Place where all the magic happens:
        state = self._propagator_num.propagate(orekit_date)

        # return and store output of propagation
        dtorque = self._write_d_torques()
        [cart_teme, att] = self.write_satellite_state(state)
        return [cart_teme, att, dtorque]

    def _write_d_torques(self):

        dtorque = DisturbanceTorques()

        if self._hasAttitudeProp:
            dtorque.add = self.attProv.getAddedDisturbanceTorques()
            dtorque.dtorque = self.attProv.getDisturbanceTorques()
        else:
            dtorque.add = [False]*5
            zeros_vector = [Vector3D.ZERO]*5
            dtorque.dtorque = orekit.JArray('object')(zeros_vector, Vector3D)

        return dtorque

    def change_attitude_provider(self):
        """
        Method changing propagator's provider between FixedRate and Inertia.

        Rate and axis are obtained from Propulsion node and stored by
        callback method. If rotation is required the callback method
        sets trigger to true and the propagator's attitude provider is set
        to FixedRate. Otherwise an InertialProvider is used so that propagation
        keeps attitude constant.

        After every change in attitude provider the name is also stored in the
        variable attProv_name to ensure that the provider is only set once
        (when changes was requested).
        """

        if self.attitude_trigger is True:
            sat_state = self._propagator_num.getInitialState()
            att_prov = self.provide_fixed_rate_attitude(sat_state)
            self._propagator_num.setAttitudeProvider(att_prov)

            self.attProv_name = self._propagator_num.getAttitudeProvider()
            self.attitude_trigger = False

        elif (self.attProv_name !=
              self._propagator_num.getAttitudeProvider().toString()):
            curr_rot = self._propagator_num.getInitialState() \
                                           .getAttitude() \
                                           .getRotation()
            self._propagator_num.setAttitudeProvider(InertialProvider(curr_rot))

            self.attProv_name = self._propagator_num.getAttitudeProvider()\
                                                    .toString()
        else:
            pass

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

    def calculate_external_torque(self):
        """
        Method which feeds external torques to attitude propagation provider.

        Method checks if torque has already been set by message from propulsion
        node (stored in self.torque), and then feeds them to the provider.
        """

        N_T = getattr(self, 'torque', None)

        if N_T is not None:
            N_T = Vector3D(float(N_T[0]),
                           float(N_T[1]),
                           float(N_T[2]))
            attProv = PAP.cast_(self._propagator_num.getAttitudeProvider())
            attProv.setExternalTorque(N_T)

    def init_fixed_rate_attitude(self):
        """
        Method adds Fixed rate provider to propagator's attitude provider.

        Spin rate and spin acceleration is set to zero and trigger with
        attitude provider name is initialized, so that it can be used in
        the method change_attitude_provider().
        """

        self.sat_spin = [0, 0, 0]
        sat_state = self._propagator_num.getInitialState()
        att_prov = self.provide_fixed_rate_attitude(sat_state)
        self._propagator_num.setAttitudeProvider(att_prov)
        self.attitude_trigger = False  # trigger to change attitude in prop
        self.attProv_name = self._propagator_num.getAttitudeProvider() \
                                                .toString()

    def provide_fixed_rate_attitude(self, sat_state):
        """
        Method creates FixedRate attitude provider object.

        Args:
            sat_state: current state of satellite

        Returns:
            FixedRate: Orekit's FixedRate provider
        """
        sat_spin = Vector3D(float(self.sat_spin[0]),
                            float(self.sat_spin[1]),
                            float(self.sat_spin[2]))
        sat_acc = Vector3D.ZERO

        start_date = sat_state.getDate()
        sat_frame = sat_state.getFrame()
        sat_rotation = sat_state.getAttitude().getRotation()

        rot_attitude = Attitude(start_date,
                                sat_frame,
                                sat_rotation,
                                sat_spin,
                                sat_acc)

        return FixedRate(rot_attitude)

    def thrust_torque_callback(self, thrust_force, thrust_ispM):
        """
        Callback function for subscriber to the propulsion node.

        Propulsion has to set torque and force back to Zero if no
        forces/torques present.

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

    def attitude_fixed_rot_callback(self, att_parameter):
        """
        Callback for attitude control using Orekit's Fixed Rate provider.

        Normalizes the rotation axis vector if necessary and sets spin rate and
        an attitude trigger indicating attitude change.
        Is used with  method: simple_attitude_control()

        Args:
            att_parameter: attitude_ctrl message containing spin axis vector
                           and angular rate
        """

        sat_spin = np.array([att_parameter.axis.x,
                             att_parameter.axis.y,
                             att_parameter.axis.z])

        norm = np.linalg.norm(sat_spin)
        if norm != 0:
            sat_spin = sat_spin / norm

        self.sat_spin = sat_spin * att_parameter.angle_rate
        self.attitude_trigger = True
