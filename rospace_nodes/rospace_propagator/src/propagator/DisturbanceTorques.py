# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

# #####################################################################
# PRELIMINARY CODE: This code is still under construction.
# Methods used from here could result to unexpected behavior and should
# therefore be used with care.
#######################################################################

import abc
import numpy as np
from math import sqrt, degrees

from FileDataHandler import FileDataHandler
from DipoleModel import DipoleModel

from org.orekit.bodies import BodyShape
from org.orekit.forces import ForceModel
from org.orekit.utils import PVCoordinatesProvider
from org.orekit.forces.drag.atmosphere import Atmosphere
from org.orekit.frames import TopocentricFrame


from org.hipparchus.geometry.euclidean.threed import Rotation, Vector3D
from org.hipparchus.util import Precision


class DisturbanceTorqueInterface(object):
    """
    Base class for Disturbance Torques.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_satellite_state(self, state_observer):
        pass

    @abc.abstractmethod
    def compute_torques(self, rotation, omega):
        pass

    @abc.abstractmethod
    def _compute_gravity_torque(self):
        pass

    @abc.abstractmethod
    def _compute_magnetic_torque(self):
        pass

    @abc.abstractmethod
    def _compute_solar_torque(self):
        pass

    @abc.abstractmethod
    def _compute_aero_torque(self, omega):
        pass

    @abc.abstractmethod
    def _set_external_torque(self, e_torque):
        pass

    @abc.abstractproperty
    def gTorque(self):
        pass

    @abc.abstractproperty
    def mTorque(self):
        pass

    @abc.abstractproperty
    def aTorque(self):
        pass

    @abc.abstractproperty
    def sTorque(self):
        pass

    @abc.abstractproperty
    def eTorque(self):
        pass

    @abc.abstractproperty
    def to_add(self):
        pass


class DisturbanceTorqueArray(DisturbanceTorqueInterface):

    @property
    def gTorque(self):
        return self._gTorque

    @property
    def mTorque(self):
        return self._mTorque

    @property
    def sTorque(self):
        return self._sTorque

    @property
    def aTorque(self):
        return self._aTorque

    @property
    def eTorque(self):
        return self._eTorque

    @property
    def to_add(self):
        return self._to_add

    @to_add.setter
    def to_add(self, add):
        self._to_add = add

    def __init__(self, observer, in_frame, in_date, inCub, AttitudeFM, earth, sun, meshDA):
        self.state_observer = observer

        self.in_frame = in_frame

        self.curr_date = in_date

        self._gTorque = None

        self._mTorque = None

        self._sTorque = None

        self._aTorque = None

        self._eTorque = None

        self._to_add = None

        self.inCub = inCub
        '''Dictionary of inner cuboid obtained from discretization'''

        self.meshDA = meshDA
        '''Dictionary of surfaces obtained from discretization'''

        self.GravityModel = None
        '''Gravity Model for gravity torque computation.
        Torque will only be computed if one is provided.'''

        self.dipleM = None
        '''Dipole model for magnetic torque computation.
        Torque will only be computed if one is provided.'''

        self.SolarModel = None
        '''Solar Model for computation of of torque due to solar pressure.
        Torque will only be computed if one is provided.'''

        self.AtmoModel = None
        '''Atmospheric Model for computation of torque due to drag.
        Torque will only be computed if one is provided.'''

        self.sun = None
        '''PVCoordinatesProvider object of Sun.'''

        self.earth = None
        '''BodyShape object of the Earth.'''

        self.K_REF = None
        '''Reference flux normalized for a 1m distance (N). [Taken from Orekit]'''

        if 'GravityModel' in AttitudeFM:
            self.GravityModel = AttitudeFM['GravityModel']
            self.muGM = ForceModel.cast_(self.GravityModel).getParameters()[0]

        if 'MagneticModel' in AttitudeFM:
            # self.MagneticModel = AttitudeFM['MagneticModel']
            self.earth = BodyShape.cast_(AttitudeFM['Earth'])
            self._initialize_dipole_model(AttitudeFM['MagneticModel'])

        if 'SolarModel' in AttitudeFM:
            self.SolarModel = AttitudeFM['SolarModel']
            self.sun = PVCoordinatesProvider.cast_(AttitudeFM['Sun'])
            # Reference distance for the solar radiation pressure (m).
            D_REF = float(149597870000.0)
            # Reference solar radiation pressure at D_REF (N/m^2).
            P_REF = float(4.56e-6)
            # Reference flux normalized for a 1m distance (N).
            self.K_REF = float(P_REF * D_REF * D_REF)

        if 'AtmoModel' in AttitudeFM:
            self.AtmoModel = Atmosphere.cast_(AttitudeFM['AtmoModel'])

        # # update and store computed values
        # self._to_add[0] = False if self.GravityModel is None else True
        # self._to_add[1] = False if self.dipoleM is None else True
        # self._to_add[2] = False if self.SolarModel is None else True
        # self._to_add[3] = False if self.AtmoModel is None else True

    def _initialize_dipole_model(self, model):
        self.dipoleM = DipoleModel()

        for key, hyst in model['Hysteresis'].items():
            direction = np.array([float(x) for x in hyst['dir'].split(" ")])
            self.dipoleM.addHysteresis(direction, hyst['vol'], hyst['Hc'], hyst['Bs'], hyst['Br'])

        # initialize values for Hysteresis
        spacecraft_state = self.state_observer.spacecraftState
        self.inertial2Sat = spacecraft_state.getAttitude().getRotation()
        self.satPos_i = spacecraft_state.getPVCoordinates().getPosition()

        gP = self.earth.transform(self.satPos_i, self.in_frame, self.curr_date)

        topoframe = TopocentricFrame(self.earth, gP, 'ENU')
        topo2inertial = topoframe.getTransformTo(self.in_frame, self.curr_date)

        lat = gP.getLatitude()
        lon = gP.getLongitude()
        alt = gP.getAltitude() / 1e3  # Mag. Field needs degrees and [km]

        # get B-field in geodetic system (X:East, Y:North, Z:Nadir)
        B_geo = FileDataHandler.mag_field_model.calculateField(
                            degrees(lat), degrees(lon), alt).getFieldVector()

        # convert geodetic frame to inertial and from [nT] to [T]
        B_i = topo2inertial.transformVector(Vector3D(1e-9, B_geo))

        B_b = self.inertial2Sat.applyTo(B_i)
        B_field = np.array([B_b.x, B_b.y, B_b.z])

        self.dipoleM.initializeHysteresisModel(B_field)

        for key, bar in model['BarMagnet'].items():
            direction = np.array([float(x) for x in bar['dir'].split(" ")])
            self.dipoleM.addBarMagnet(direction, bar['m'])

    def update_satellite_state(self, current_date):
        '''call before integration'''
        self.curr_date = current_date
        self.spacecraft_state = self.state_observer.spacecraftState
        self.inertial2Sat = self.spacecraft_state.getAttitude().getRotation()

        self.satPos_i = self.spacecraft_state.getPVCoordinates().getPosition()
        self.satVel_i = self.spacecraft_state.getPVCoordinates().getVelocity()

    def compute_torques(self, rotation, omega):
        # override old orientation of satellite:
        # only orientation changes, not position & velocity
        self.inertial2Sat = rotation
        self.satPos_s = self.inertial2Sat.applyTo(self.satPos_i)
        self.satPos_s = np.array([self.satPos_s.x,
                                  self.satPos_s.y,
                                  self.satPos_s.z], dtype='float64')

        self._compute_gravity_torque()
        self._compute_magnetic_torque()
        self._compute_solar_torque()
        self._compute_aero_torque(omega)

        # external torque has to be set separately because it is received
        # through a ros subscriber
        return self._gTorque.add(
                self._mTorque.add(
                 self._sTorque.add(
                  self._aTorque)))

    def _compute_gravity_torque(self):
        """Compute gravity gradient torque if gravity model provided.

        This method is declared in the Orekit wrapper in
        PythonAttitudePropagation.java and overridden here, so that
        an OrekitException is thrown if something goes wrong.

        It computes the Newtonian attraction and the perturbing part
        of the gravity gradient for every cuboid defined in dictionary
        inCub at time refDate (= time of current satellite position).
        The gravity torque is computed in the inertial frame in which the
        spacecraft is defined. The perturbing part is calculated using Orekit's
        methods defined in the GravityModel object.

        The current position, rotation and mass of the satellite is obtained
        from the StateObserver object.

        Returns:
            Vector3D: gravity gradient torque at curr_date in satellite frame
        """

        if self._to_add[0]:
            # return gravity gradient torque in satellite frame
            body2inertial = self.earth.getBodyFrame().getTransformTo(self.in_frame, self.curr_date)
            body2sat = self.inertial2Sat.applyTo(body2inertial.getRotation())
            body2satRot = PyRotation(body2sat.q0,
                                     body2sat.q1,
                                     body2sat.q2,
                                     body2sat.q3)
            sat2bodyRot = body2satRot.revert()
            body2sat = body2satRot.getMatrix()
            sat2body = sat2bodyRot.getMatrix()

            satM = self.spacecraft_state.getMass()
            mCub = self.inCub['mass_frac'] * satM
            CoM = self.inCub['CoM_np']

            dmPos_s = CoM + self.satPos_s

            gNewton = (-self.muGM / np.linalg.norm(dmPos_s,
                                                   axis=1,
                                                   keepdims=True)**3) * dmPos_s

            # rotate vectors:
            dmPos_b = np.einsum('ij,kj->ki', sat2body, dmPos_s)

            gDist = np.empty(dmPos_b.shape)
            for i in xrange(0, dmPos_b.shape[0]):
                gDist[i, :] = np.asarray(
                    self.GravityModel.gradient(self.curr_date,
                                               Vector3D(float(dmPos_b[i, 0]),
                                                        float(dmPos_b[i, 1]),
                                                        float(dmPos_b[i, 2])),
                                               self.muGM))

            gDist_s = np.einsum('ij,kj->ki', body2sat, gDist)

            gT = np.sum(np.cross(CoM, mCub*(gNewton + gDist_s)), axis=0)

            self._gTorque = Vector3D(float(gT[0]), float(gT[1]), float(gT[2]))

        else:
            self._gTorque = Vector3D.ZERO

    def _compute_magnetic_torque(self):
        """Compute magnetic torque if magnetic model provided.

        This method is declared in the Orekit wrapper in
        PythonAttitudePropagation.java and overridden here, so that
        an OrekitException is thrown if something goes wrong.

        It gets the satellites dipole vector which is stored in the base class,
        converts the satellite's position into Longitude, Latitude, Altitude
        representation to determine the geo. magnetic field at that position
        and then computes base on those values the magnetic torque.

        Returns:
            Vector3D: magnetic torque at satellite position in satellite frame
        """
        if self._to_add[1]:
            gP = self.earth.transform(self.satPos_i, self.in_frame, self.curr_date)

            topoframe = TopocentricFrame(self.earth, gP, 'ENU')
            topo2inertial = topoframe.getTransformTo(self.in_frame, self.curr_date)

            lat = gP.getLatitude()
            lon = gP.getLongitude()
            alt = gP.getAltitude() / 1e3  # Mag. Field needs degrees and [km]

            # get B-field in geodetic system (X:East, Y:North, Z:Nadir)
            B_geo = FileDataHandler.mag_field_model.calculateField(
                                degrees(lat), degrees(lon), alt).getFieldVector()

            # convert geodetic frame to inertial and from [nT] to [T]
            B_i = topo2inertial.transformVector(Vector3D(1e-9, B_geo))

            B_b = self.inertial2Sat.applyTo(B_i)
            B_b = np.array([B_b.x, B_b.y, B_b.z])

            dipoleVector = self.dipoleM.getDipoleVectors(B_b)

            torque = np.sum(np.cross(dipoleVector, B_b), axis=0)

            self._mTorque = Vector3D(float(torque[0]), float(torque[1]), float(torque[2]))
        else:
            self._mTorque = Vector3D.ZERO

    def _compute_solar_torque(self):
        """Compute torque acting on satellite due to solar radiation pressure.

        This method is declared in the Orekit wrapper in
        PythonAttitudePropagation.java and overridden here, so that
        an OrekitException is thrown if something goes wrong.

        This method uses the getLightingRatio method defined in Orekit and
        copies parts of the acceleration() method of the SolarRadiationPressure
        and radiationPressureAcceleration() of the BoxAndSolarArraySpacecraft
        class to to calculate the solar radiation pressure on the discretized
        surface of the satellite. This is done, since the necessary Orekit
        methods cannot be accessed directly without creating an Spacecraft
        object.

        Returns:
            Vector3D: solar radiation pressure torque in satellite frame along principle axes

        Raises:
            AssertionError: if (Absorption Coeff + Specular Reflection Coeff) > 1
        """
        if self._to_add[2]:
            ratio = self.SolarModel.getLightingRatio(self.satPos_i,
                                                     self.in_frame,
                                                     self.curr_date)

            sunPos = self.inertial2Sat.applyTo(
                    self.sun.getPVCoordinates(self.curr_date,
                                              self.in_frame).getPosition())
            sunPos = np.array([sunPos.x, sunPos.y, sunPos.z], dtype='float64')

            CoM = self.meshDA['CoM_np']
            normal = self.meshDA['Normal_np']
            area = self.meshDA['Area_np']
            coefs = self.meshDA['Coefs_np']

            sunSatVector = self.satPos_s + CoM - sunPos
            r = np.linalg.norm(sunSatVector, axis=1)
            rawP = ratio * self.K_REF / (r**2)
            flux = (rawP / r)[:, None] * sunSatVector
            # eliminate arrays where zero flux
            fluxNorm = np.linalg.norm(flux, axis=1)
            Condflux = fluxNorm**2 > Precision.SAFE_MIN
            flux = flux[Condflux]
            normal = normal[Condflux]

            # dot product for multidimensional arrays:
            dot = np.einsum('ij,ij->i', flux, normal)
            dot[dot > 0] = dot[dot > 0] * (-1.0)
            if dot.size > 0:
                normal[dot > 0] = normal[dot > 0] * (-1.0)

                cN = 2 * area * dot * (coefs[:, 2] / 3 - coefs[:, 1] * dot / fluxNorm)
                cS = (area * dot / fluxNorm) * (coefs[:, 1] - 1)
                force = cN[:, None] * normal + cS[:, None] * flux

                sT = np.sum(np.cross(CoM, force), axis=0)

                self._sTorque = Vector3D(float(sT[0]), float(sT[1]), float(sT[2]))

        else:
            self._sTorque = Vector3D.ZERO

    def _compute_aero_torque(self, omega):
        """Compute torque acting on satellite due to drag.

        This method is declared in the Orekit wrapper in
        PythonAttitudePropagation.java and overridden here, so that
        an OrekitException is thrown if something goes wrong.

        This method copies parts of the acceleration() method of the
        DragForce and dragAcceleration() of the BoxAndSolarArraySpacecraft
        class to to calculate the pressure on the discretized surface of the
        satellite. This is done, since the necessary Orekit methods cannot be
        accessed directly without creating an Spacecraft object.

        Returns:
            Vector3D: torque due to drag along principle axes in satellite frame
        """
        if self._to_add[3]:
            # assuming constant atmosphere condition over spacecraft
            # error is of order of 10^-17
            rho = self.AtmoModel.getDensity(self.curr_date, self.satPos_i, self.in_frame)
            vAtm_i = self.AtmoModel.getVelocity(self.curr_date, self.satPos_i, self.in_frame)

            satVel = self.inertial2Sat.applyTo(self.satVel_i)
            vAtm = self.inertial2Sat.applyTo(vAtm_i)

            dragCoeff = self.meshDA['Cd']
            liftRatio = 0.0  # no lift considered

            CoM = self.meshDA['CoM_np']
            normal = self.meshDA['Normal_np']
            area = np.asarray(self.meshDA['Area'])
            satVel = np.array([satVel.x, satVel.y, satVel.z])
            vAtm = np.array([vAtm.x, vAtm.y, vAtm.z])

            relativeVelocity = vAtm - (satVel + (np.cross(omega, CoM)))
            vNorm = np.linalg.norm(relativeVelocity, axis=1)
            vDir = np.reciprocal(vNorm[:, None]) * relativeVelocity

            dot = np.einsum('ij,ij->i', normal, vDir)

            dotCondition = dot < 0
            dot = dot[dotCondition]
            if dot.size > 0:
                vDir = vDir[dotCondition]
                vNorm = vNorm[dotCondition]
                normal = normal[dotCondition]
                area = area[dotCondition]
                CoM = CoM[dotCondition]

                coeff = 0.5 * rho * dragCoeff * (vNorm**2)
                oMr = 1.0 - liftRatio
                f = (coeff * area * dot)[:, None]

                aT = np.sum(np.cross(CoM, oMr * np.absolute(f) * vDir + 2 * liftRatio * f * normal), axis=0)

                self._aTorque = Vector3D(float(aT[0]), float(aT[1]), float(aT[2]))

        else:
            self._aTorque = Vector3D.ZERO

    def _set_external_torque(self, eTorque):
        self._eTorque = eTorque


class PyRotation(object):
    '''This class uses the Rotation class methods from the Hipparchus library
    rewritten in Python to create and obtain the same rotation matrix which
    would be returned by the Hipparchus library, only as a numpy array.
    '''
    def __init__(self, q0, q1, q2, q3):
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def getMatrix(self):
        # products
        q0q0 = self.q0 * self.q0
        q0q1 = self.q0 * self.q1
        q0q2 = self.q0 * self.q2
        q0q3 = self.q0 * self.q3
        q1q1 = self.q1 * self.q1
        q1q2 = self.q1 * self.q2
        q1q3 = self.q1 * self.q3
        q2q2 = self.q2 * self.q2
        q2q3 = self.q2 * self.q3
        q3q3 = self.q3 * self.q3

        # create the matrix
        m = np.empty([3, 3], dtype='float64')

        m[0, 0] = 2.0 * (q0q0 + q1q1) - 1.0
        m[1, 0] = 2.0 * (q1q2 - q0q3)
        m[2, 0] = 2.0 * (q1q3 + q0q2)

        m[0, 1] = 2.0 * (q1q2 + q0q3)
        m[1, 1] = 2.0 * (q0q0 + q2q2) - 1.0
        m[2, 1] = 2.0 * (q2q3 - q0q1)

        m[0, 2] = 2.0 * (q1q3 - q0q2)
        m[1, 2] = 2.0 * (q2q3 + q0q1)
        m[2, 2] = 2.0 * (q0q0 + q3q3) - 1.0

        return m

    def revert(self):
        return PyRotation(-self.q0, self.q1, self.q2, self.q3)

    def applyTo(self, v):
        s = self.q1 * v[0] + self.q2 * v[1] + self.q3 * v[2]

        return np.array([2 * (self.q0 * (v[0] * self.q0 - (self.q2 * v[2] - self.q3 * v[1])) + s * self.q1) - v[0],
                         2 * (self.q0 * (v[1] * self.q0 - (self.q3 * v[0] - self.q1 * v[2])) + s * self.q2) - v[1],
                         2 * (self.q0 * (v[2] * self.q0 - (self.q1 * v[1] - self.q2 * v[0])) + s * self.q3) - v[2]])

    def applyInverseTo(self, v):
        s = self.q1 * v[0] + self.q2 * v[1] + self.q3 * v[2]
        m0 = -self.q0

        return np.array([2 * (m0 * (v[0] * m0 - (self.q2 * v[2] - self.q3 * v[1])) + s * self.q1) - v[0],
                         2 * (m0 * (v[1] * m0 - (self.q3 * v[0] - self.q1 * v[2])) + s * self.q2) - v[1],
                         2 * (m0 * (v[2] * m0 - (self.q1 * v[1] - self.q2 * v[0])) + s * self.q3) - v[2]])
