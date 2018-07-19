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
from math import degrees, sin, cos

from FileDataHandler import FileDataHandler, to_orekit_date

import orekit

from java.io import File

from org.orekit.python import PythonAttitudePropagation as PAP
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.data import DataProvidersManager, ZipJarCrawler
from org.orekit.frames import TopocentricFrame
from org.orekit.propagation import SpacecraftState

from org.hipparchus.geometry.euclidean.threed import Vector3D


class DisturbanceTorqueStorage(object):
    '''Stores disturbance torques in satellite frame and their activation
    status in numpy array.

    The torques are stored in following order:
        - gravity gradient
        - magnetic torque
        - solar radiation pressure torque
        - drag torque
        - external torque
        - sum of all torques
    '''
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
        torque_sum = np.array([0.0, 0.0, 0.0])
        for vector in new_dtorques:
            t_array = np.array([vector.getX(),
                               vector.getY(),
                               vector.getZ()])
            torque_sum += t_array
            self._dtorque.append(t_array)

        self._dtorque.append(torque_sum)


class OrekitPropagator(object):
    """
    Class building numerical propagator using the orekit library methods.
    """

    # _data_checklist = dict()
    # """Holds dates for which data from orekit-data folder is loaded"""
    # _mag_field_coll = None
    # """Java Collection holding all loaded magnetic field models"""
    # _mag_field_model = None
    # """Currently used magnetic field model, transformed to correct year"""

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
    def _write_satellite_state(state):
        """
        Method to extract satellite orbit state vector, rotation and force
        from Java object and put it in a numpy array.

        Args:
            SpacecraftState: satellite state vector

        Returns:
            numpy.array: cartesian state vector in TEME frame
            numpy.array: current attitude rotation in quaternions
            numpy.array: force in satellite body frame
        """

        pv = state.getPVCoordinates()
        cart_teme = rospace_lib.CartesianTEME()
        cart_teme.R = np.array([pv.position.x,
                                pv.position.y,
                                pv.position.z]) / 1000
        cart_teme.V = np.array([pv.velocity.x,
                                pv.velocity.y,
                                pv.velocity.z]) / 1000

        att = state.getAttitude()

        a_sF = att.getRotation().applyTo(pv.acceleration)
        force_sF = np.array([a_sF.x,
                             a_sF.y,
                             a_sF.z]) * state.getMass()

        return [cart_teme, att, force_sF]

    def _write_d_torques(self):

        dtorque = DisturbanceTorqueStorage()

        if self._hasAttitudeProp:
            dtorque.add = self.attProv.getAddedDisturbanceTorques()
            dtorque.dtorque = self.attProv.getDisturbanceTorques()
        else:
            dtorque.add = [False]*6
            zeros_vector = [Vector3D.ZERO]*6
            dtorque.dtorque = orekit.JArray('object')(zeros_vector, Vector3D)

        return dtorque

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
            epoch: initial epoch @ which state is defined as datetime object

        Raises:
            AssertionError: if propagator settings file was not found
        """

        OrEpoch = to_orekit_date(epoch)

        try:
            assert propSettings != 0
        except AssertionError:
            ass_err = "ERROR: Propagator settings file could not be found!"
            raise AssertionError(ass_err)

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
            numpy.array: force acting on satellite in body frame
            numpy.array: disturbance torques acting on satellite in body frame
            numpy.array: Magnetic field at satellite position in TEME frame
        """

        orekit_date = to_orekit_date(epoch)

        if self._hasAttitudeProp:
            self.calculate_external_torque()

        if self._hasThrust:
            self.calculate_thrust()

        # Place where all the magic happens:
        state = self._propagator_num.propagate(orekit_date)

        if self._hasAttitudeProp:
            # get attitude of last time step at add it to new state
            # propagator does update of last step before new orbit propagation step
            old_state = self._propagator_num.getInitialState()
            orbit = old_state.getOrbit()
            date = old_state.getDate()
            frame = old_state.getFrame()

            att_corr = self._propagator_num.getAttitudeProvider()\
                           .getAttitude(orbit, date, frame)
            state = SpacecraftState(orbit, att_corr, old_state.getMass())
            self._propagator_num.setInitialState(state)

        # return and store output of propagation
        dtorque = self._write_d_torques()
        [cart_teme, att, force_sF] = self._write_satellite_state(state)
        B_field_b = self._calculate_magnetic_field(orekit_date)

        return [cart_teme, att, force_sF, dtorque, B_field_b]

    def _calculate_magnetic_field(self, oDate):
        space_state = self._propagator_num.getInitialState()
        satPos = space_state.getPVCoordinates().getPosition()
        inertial2Sat = space_state.getAttitude().getRotation()
        frame = space_state.getFrame()

        gP = self._earth.transform(satPos, frame, oDate)

        topoframe = TopocentricFrame(self._earth, gP, 'ENU')
        topo2inertial = topoframe.getTransformTo(frame, oDate)

        lat = gP.getLatitude()
        lon = gP.getLongitude()
        alt = gP.getAltitude() / 1e3  # Mag. Field needs degrees and [km]

        # get B-field in geodetic system (X:East, Y:North, Z:Nadir)
        B_geo = FileDataHandler.mag_field_model.calculateField(
                            degrees(lat), degrees(lon), alt).getFieldVector()

        # convert geodetic frame to inertial and from [nT] to [T]
        B_i = topo2inertial.transformVector(Vector3D(1e-9, B_geo))

        return inertial2Sat.applyTo(B_i)

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
        Isp = getattr(self, 'Isp', None)

        if F_T is not None and Isp is not None:
            F_T_norm = np.linalg.norm(F_T, 2)

            if F_T_norm > 2 * np.finfo(np.float32).eps:
                F_T_dir = F_T / F_T_norm
                F_T_dir = Vector3D(float(F_T_dir[0]),
                                   float(F_T_dir[1]),
                                   float(F_T_dir[2]))

                self.ThrustModel.direction = F_T_dir
                self.ThrustModel.thrust = float(F_T_norm)
                self.ThrustModel.isp = Isp
                self.ThrustModel.ChangeParameters(float(F_T_norm), Isp)
                self.ThrustModel.firing = True
            else:
                self.ThrustModel.firing = False

    def calculate_external_torque(self):
        """
        Method which feeds external torques to attitude propagation provider.

        Method checks if torque has already been set by message from propulsion
        node (stored in self.torque), and then feeds them to the provider.
        """

        N_T_thrust = getattr(self, 'thrust_torque', None)
        N_T_actuator = getattr(self, 'actuator_torque', None)

        if N_T_thrust is not None and N_T_actuator is not None:
            N_T = N_T_thrust + N_T_actuator
        elif N_T_thrust is not None:
            N_T = N_T_thrust
        elif N_T_actuator is not None:
            N_T = N_T_actuator
        else:
            N_T = None

        if N_T is not None:
            N_T = Vector3D(float(N_T[0]),
                           float(N_T[1]),
                           float(N_T[2]))
            attProv = PAP.cast_(self._propagator_num.getAttitudeProvider())
            attProv.setExternalTorque(N_T)

    def thrust_callback(self, force_torque, thrust_ispM):
        """
        Callback function for subscriber to the propulsion node.

        The propulsion node has to set torque and force back to zero if no
        force/torque is present.

        Function saves the value for the mean specific impulse and
        the force- & torque-vector to class objects.

        If the callback is used it is necessary that the propulsion node publishes
        the current thrust force at every timestep (even if it is zero)!

        Messages for thrust/torque and specific impulse are synchronized.

        Args:
            thrust_force (geometry_msgs.msg.WrenchStamped): WrenchStamped message containing torque and force
            thrust_ispM (rospace_msgs.msg.ThrustIsp): ThrustIsp message with specific impulse of maneuver
        """

        # this currently only called in by chaser object
        self.F_T = np.array([force_torque.wrench.force.x,
                             force_torque.wrench.force.y,
                             force_torque.wrench.force.z])

        self.Isp = thrust_ispM.Isp_val

        # Torque:
        self.thrust_torque = np.array([force_torque.wrench.torque.x,
                                       force_torque.wrench.torque.y,
                                       force_torque.wrench.torque.z])

    def magnetotorque_callback(self, torque_msg):
        """Callback function for subscriber to the magnetotorquer node.

        The magnetotorquer has to set the torque back to zero if no torque present.

        Args:
            torque_msg (geometry_msgs.msg.WrenchStamped): message containing torque in spacecraft body frame
        """
        self.actuator_torque = np.array([torque_msg.wrench.torque.x,
                                         torque_msg.wrench.torque.y,
                                         torque_msg.wrench.torque.z])
