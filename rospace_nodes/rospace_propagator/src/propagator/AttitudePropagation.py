# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import orekit
import numpy as np
import itertools
from math import sqrt, sin, cos
from math import degrees
import sys  # for errors
import traceback

from org.orekit.attitudes import Attitude
from org.orekit.python import PythonAttitudePropagation as PAP
from org.orekit.python import PythonStateEquation as PSE
from org.orekit.bodies import BodyShape
from org.orekit.forces import ForceModel
from org.orekit.forces.drag.atmosphere import Atmosphere
from org.orekit.utils import PVCoordinatesProvider

from org.hipparchus.util import Precision
from org.hipparchus.ode import OrdinaryDifferentialEquation
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator
from org.hipparchus.geometry.euclidean.threed import Rotation, Vector3D


def _log_value(name, value, date, f):
    f.write(name + ": " + str(value) + " " + str(date) + '\n')


def _compare_results(y, y_s, f):
    # y = new.getPrimaryState()

    f.write("#######################\n")
    f.write("Integration Comparison:\n")
    # f.write(str(y[0]) + " ?=? " + str(y_s[0]) + " err: " + str(y[0] - y_s[0]) + "\n")
    # f.write(str(y[1]) + " ?=? " + str(y_s[1]) + " err: " + str(y[1] - y_s[1]) + "\n")
    # f.write(str(y[2]) + " ?=? " + str(y_s[2]) + " err: " + str(y[2] - y_s[2]) + "\n")
    # f.write(str(y[3]) + " ?=? " + str(y_s[3]) + " err: " + str(y[3] - y_s[3]) + "\n")
    # f.write(str(y[4]) + " ?=? " + str(y_s[4]) + " err: " + str(y[4] - y_s[4]) + "\n")
    # f.write(str(y[5]) + " ?=? " + str(y_s[5]) + " err: " + str(y[5] - y_s[5]) + "\n")
    # f.write(str(y[6]) + " ?=? " + str(y_s[6]) + " err: " + str(y[6] - y_s[6]) + "\n")
    f.write(str(y.x) + " ?=? " + str(y_s.x) + " err: " + str(y.x - y_s.x) + "\n")
    f.write(str(y.y) + " ?=? " + str(y_s.y) + " err: " + str(y.y - y_s.y) + "\n")
    f.write(str(y.z) + " ?=? " + str(y_s.z) + " err: " + str(y.z - y_s.z) + "\n")
    f.write("#######################\n")


class AttitudePropagation(PAP):
    """Implements an attitude propagation which is called by Orekit's attitude provider."""

    @staticmethod
    def _set_up_attitude_DormandPrice(intSettings, tol):
        '''Set up integrator for attitude propagation.

        If settings negative use same tolerances used in orbit propagation'''

        minStep = intSettings['minStep']
        maxStep = intSettings['maxStep']
        initStep = intSettings['initStep']

        if intSettings['absTol'] < 0.0:
            intAbsTol = orekit.JArray_double.cast_(tol[0])
        else:
            intAbsTol = orekit.JArray_double.cast_(intSettings['absTol'])
        if intSettings['relTol'] < 0.0:
            intRelTol = orekit.JArray_double.cast_(tol[1])
        else:
            intRelTol = orekit.JArray_double.cast_(intSettings['relTol'])
        integrator = \
            DormandPrince853Integrator(minStep, maxStep, intAbsTol, intRelTol)
        integrator.setInitialStepSize(initStep)

        return integrator

    def __init__(self,
                 attitude,
                 referenceDate,
                 inertiaT,
                 tol,
                 intSettings,
                 inCub,
                 meshDA,
                 AttitudeFM):
        super(AttitudePropagation, self).__init__(attitude)

        self.omega = Vector3D.ZERO
        '''Angular velocity of satellite in direction of principle axes.'''

        self.rotation = attitude.getRotation()

        self.state = StateEquation(7)
        '''StateEquation object. Holds 7 equations to be integrated.'''

        self.one_step_state = SingleStepEq()

        self.integrator = self._set_up_attitude_DormandPrice(intSettings, tol)
        '''DormandPrince853Integrator object for attitude propagation.'''

        self.StateObserver = AttitudeFM['StateObserver']
        '''Dictionary of force models for gravity torques & inertia tensor.'''

        self.inertiaT = inertiaT / self.StateObserver.spacecraftState.getMass()
        '''Inertial tensor linearly dependent on mass given for principal axes.'''

        self.refDate = referenceDate
        '''Date at current satellite state'''

        self.refFrame = attitude.getReferenceFrame()
        '''Reference frame in which attitude is computed'''

        self.inCub = inCub
        '''Dictionary of inner cuboid obtained from discretization'''

        self.meshDA = meshDA
        '''Dictionary of surfaces obtained from discretization'''

        self.GravityModel = None
        '''Gravity Model for gravity torque computation.
        Torque will only be computed if one is provided.'''

        self.MagneticModel = None
        '''World Magnetic Model for magnetic torque computation.
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
            self.MagneticModel = AttitudeFM['MagneticModel']
            self.earth = BodyShape.cast_(AttitudeFM['Earth'])

        if 'SolarModel' in AttitudeFM:
            self.SolarModel = AttitudeFM['SolarModel']
            self.sun = PVCoordinatesProvider.cast_(AttitudeFM['Sun'])
            # Reference distance for the solar radiation pressure (m).
            D_REF = float(149597870000.0)
            # Reference solar radiation pressure at D_REF (N/m^2).
            P_REF = float(4.56e-6)
            self.K_REF = float(P_REF * D_REF * D_REF)

        if 'AtmoModel' in AttitudeFM:
            self.AtmoModel = Atmosphere.cast_(AttitudeFM['AtmoModel'])

        self.V3_cross = Vector3D.crossProduct
        self.V3_dot = Vector3D.dotProduct

        # update and store computed values
        addG = False if self.GravityModel is None else True
        addM = False if self.MagneticModel is None else True
        addSP = False if self.SolarModel is None else True
        addD = False if self.AtmoModel is None else True
        self.setAddedDisturbanceTorques(addG, addM, addSP, addD)

        # self.f = open('/home/christian/Documents/ETH/MasterThesis/Profiling/Vectorization/output.txt', 'a+')

    def getAttitude(self, pvProv, date, frame):
        """Method called by Orekit at every state integration step.

        This Method calculates the disturbance torques if initialized
        and then integrates the 7 state equations to determine
        the satellites attitude at the next time step.

        The parameters of the state equation are the angular rate along
        the satellite's principle axes (1:3) and the rotation given in
        quaternions (4:7)

        The date provided as argument must no necessarily be advancing in
        time. Also earlier time can be given, resulting in a negative time
        difference

        Args:
            pvProv: the satellites orbit given as PVCoordinatesProvider object
            date: AbsoluteDate object to which attitude should be integrated
            frame: Frame in which attitude should be given

        Returns:
            Attitude: new attitude @ date

        Raises:
            MathIllegalArgumentException: if integration step is too small
            MathIllegalArgumentException: if the location of an event cannot be bracketed
            MathIllegalStateException: if the number of functions evaluations is exceeded
            Exception: if anything else went wrong (should never raise this)
        """
        try:
            if self.refDate.equals(date):
                return self.getReferenceAttitude()

            else:
                self.state.inertiaT = self.inertiaT * \
                    self.StateObserver.spacecraftState.getMass()

                self.state.torque_control = self.getExternalTorque()
                self.to_add = self.getAddedDisturbanceTorques()

                self._initialize_disturbance_calculation()
                gTorque = self.getGravTorqueArray()
                mTorque = self.getMagTorque()
                sTorque = self.getSolarTorqueArray()
                aTorque = self.getAeroTorqueArray()
                self.setDisturbanceTorques(gTorque, mTorque, sTorque, aTorque)

                # _compare_results(gTorque, gTorqueRef, self.f)

                self.state.torque_dist = gTorque.add(
                    mTorque.add(
                        sTorque.add(
                            aTorque)))

                y = orekit.JArray('double')(7)
                y[0] = self.omega.x  # angular momentum
                y[1] = self.omega.y
                y[2] = self.omega.z
                # get rotation in quaternions:
                # scalar part is Q0 in Orekit, but q3 in integration
                y[3] = self.rotation.q1
                y[4] = self.rotation.q2
                y[5] = self.rotation.q3
                y[6] = self.rotation.q0
                dt = date.durationFrom(self.refDate)  # refDate - date

                try:
                    new_state = self.one_step_state.integrate(
                            OrdinaryDifferentialEquation.cast_(self.state),
                            float(0.0),
                            y,
                            dt)
                except Exception:
                    raise RuntimeError("Single Step integration of Attitude failed!")

                self.rotation = Rotation(float(new_state[6]),
                                         float(new_state[3]),
                                         float(new_state[4]),
                                         float(new_state[5]), True)
                self.omega = Vector3D([new_state[0], new_state[1], new_state[2]])
                newAttitude = Attitude(date,
                                       self.refFrame,
                                       self.rotation,
                                       self.omega,
                                       Vector3D.ZERO)
                self.refDate = date
                self.setReferenceAttitude(newAttitude)

                return newAttitude

        except Exception:  # should never get here
            print traceback.print_exc()
            raise

    def _initialize_disturbance_calculation(self):
        spacecraftState = self.StateObserver.spacecraftState
        self.inertial2Sat = spacecraftState.getAttitude().getRotation()

        self.satPos_i = spacecraftState.getPVCoordinates().getPosition()
        self.satVel_i = spacecraftState.getPVCoordinates().getVelocity()
        self.satPos_s = self.inertial2Sat.applyTo(self.satPos_i)
        self.satPos_s = np.array([self.satPos_s.x,
                                  self.satPos_s.y,
                                  self.satPos_s.z], dtype='float64')

    def getGravTorque(self):
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
            Vector3D: gravity gradient torque at refDate in satellite frame
        """

        if self.to_add[0]:
            body2inertial = self.earth.getBodyFrame().getTransformTo(self.refFrame, self.refDate)
            body2sat = self.inertial2Sat.applyTo(body2inertial.getRotation())
            sat2body = body2sat.revert()

            satM = self.StateObserver.spacecraftState.getMass()
            mCub = self.inCub['mass_frac'] * satM

            gTorque = Vector3D.ZERO

            for CoM in self.inCub['CoM']:

                S_dmPos = self.satPos_s.add(CoM)

                r2 = S_dmPos.getNormSq()
                gNewton = Vector3D(-self.muGM / (sqrt(r2) * r2), S_dmPos)

                B_dmPos = sat2body.applyTo(S_dmPos)

                gDist = Vector3D(self.GravityModel.gradient(self.refDate,
                                                            B_dmPos,
                                                            self.muGM))

                g_Dist_s = body2sat.applyTo(gDist)

                dmForce = Vector3D(mCub, gNewton.add(g_Dist_s))
                gTorque = gTorque.add(self.V3_cross(CoM, dmForce))

            return gTorque
        else:
            return Vector3D.ZERO

    def getGravTorqueArray(self):
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
            Vector3D: gravity gradient torque at refDate in satellite frame
        """

        if self.to_add[0]:
            # return gravity gradient torque in satellite frame
            body2inertial = self.earth.getBodyFrame().getTransformTo(self.refFrame, self.refDate)
            body2sat = self.inertial2Sat.applyTo(body2inertial.getRotation())
            body2satRot = PyRotation(body2sat.q0,
                                     body2sat.q1,
                                     body2sat.q2,
                                     body2sat.q3)
            sat2bodyRot = body2satRot.revert()
            body2sat = body2satRot.getMatrix()
            sat2body = sat2bodyRot.getMatrix()

            satM = self.StateObserver.spacecraftState.getMass()
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
                    self.GravityModel.gradient(self.refDate,
                                               Vector3D(float(dmPos_b[i, 0]),
                                                        float(dmPos_b[i, 1]),
                                                        float(dmPos_b[i, 2])),
                                               self.muGM))

            gDist_s = np.einsum('ij,kj->ki', body2sat, gDist)

            gTorque = np.sum(np.cross(CoM, mCub*(gNewton + gDist_s)), axis=0)

            return Vector3D(float(gTorque[0]), float(gTorque[1]), float(gTorque[2]))

        else:
            return Vector3D.ZERO

    def getMagTorque(self):
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
        if self.to_add[1]:
            gP = self.earth.transform(self.satPos_i, self.refFrame, self.refDate)
            lat = gP.getLatitude()
            lon = gP.getLongitude()
            alt = gP.getAltitude() / 1e3  # Mag. Field needs degrees and [km]
            geo2inertial = np.array([
                            [-sin(lon), -cos(lon)*sin(lat), cos(lon)*cos(lat)],
                            [cos(lon), -sin(lon)*sin(lat), sin(lon)*cos(lat)],
                            [0., cos(lat), sin(lat)]])

            # get B-field in geodetic system (X:East, Y:North, Z:Nadir)
            B_geo = self.MagneticModel.calculateField(
                                degrees(lat), degrees(lon), alt).getFieldVector()

            # convert geodetic frame to inertial and from [nT] to [T]
            B_i = geo2inertial.dot(np.array([B_geo.getX(),
                                             B_geo.getY(),
                                             B_geo.getZ()])) * 1e-9

            B_b = self.inertial2Sat.applyTo(Vector3D(float(B_i[0]),
                                                     float(B_i[1]),
                                                     float(B_i[2])))
            dipole = self.getDipoleVector()

            return self.V3_cross(dipole, B_b)
        else:
            return Vector3D.ZERO

    def getSolarTorqueArray(self):
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
        if self.to_add[2]:
            ratio = self.SolarModel.getLightingRatio(self.satPos_i,
                                                     self.refFrame,
                                                     self.refDate)

            sunPos = self.inertial2Sat.applyTo(
                    self.sun.getPVCoordinates(self.refDate,
                                              self.refFrame).getPosition())
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

                spTorque = np.sum(np.cross(CoM, force), axis=0)
            else:
                spTorque = np.zeros(3)

            spTorque = Vector3D(float(spTorque[0]), float(spTorque[1]), float(spTorque[2]))

            return spTorque

        else:
            return Vector3D.ZERO

    def getSolarTorque(self):
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
        if self.to_add[2]:
            spacecraftState = self.StateObserver.spacecraftState
            inertial2Sat = spacecraftState.getAttitude().getRotation()

            ratio = self.SolarModel.getLightingRatio(self.satPos_i,
                                                     self.refFrame,
                                                     self.refDate)

            sunPos = inertial2Sat.applyTo(
                    self.sun.getPVCoordinates(self.refDate,
                                              self.refFrame).getPosition())
            spTorque = Vector3D.ZERO

            iterator = itertools.izip(self.meshDA['CoM'],
                                      self.meshDA['Normal'],
                                      self.meshDA['Area'],
                                      self.meshDA['Coefs'])

            for CoM, normal, area, coefs in iterator:
                position = self.satPos_s.add(CoM)

                # compute flux in inertial frame
                sunSatVector = \
                    position.subtract(sunPos)
                r2 = sunSatVector.getNormSq()

                rawP = ratio * self.K_REF / r2
                flux = Vector3D(rawP / sqrt(r2), sunSatVector)

                # compute Radiation Pressure Force:
                if flux.getNormSq() > Precision.SAFE_MIN:
                    # illumination (we are not in umbra)
                    # rotate flux to spacecraft frame:
                    dot = self.V3_dot(normal, flux)

                    if dot > 0:
                        # the solar array is illuminated backward,
                        # fix signs to compute contribution correctly
                        dot = -dot
                        normal = normal.negate()
                    absorbCoeff = coefs[0]
                    specularReflCoeff = coefs[1]
                    diffuseReflCoeff = 1 - (absorbCoeff + specularReflCoeff)
                    try:
                        assert(diffuseReflCoeff >= 0)
                    except AssertionError:
                        raise AssertionError(
                            "Negative diffuse reflection coefficient not possible!")
                    psr = flux.getNorm()
                    # Vallado's equation uses different parameters which are
                    # related to our parameters as:
                    # cos (phi) = - dot / (psr*area)
                    # n         = N                     (n...unit vector)
                    # s         = -fluxSat / psr        (s...unit vector)
                    cN = 2 * area * dot * (diffuseReflCoeff / 3 -
                                           specularReflCoeff * dot / psr)
                    cS = (area * dot / psr) * (specularReflCoeff - 1)
                    Force = Vector3D(float(cN), normal, float(cS), flux)
                    # Force already in spacecraft frame. No need to convert
                    spTorque = spTorque.add(self.V3_cross(CoM, Force))

            return spTorque

        else:
            return Vector3D.ZERO

    def getAeroTorque(self):
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

        if self.to_add[3]:
            # assuming constant atmosphere condition over spacecraft
            # error is of order of 10^-17
            rho = self.AtmoModel.getDensity(self.refDate, self.satPos_i, self.refFrame)
            vAtm_i = self.AtmoModel.getVelocity(self.refDate, self.satPos_i, self.refFrame)

            satVel = self.inertial2Sat.applyTo(self.satVel_i)
            vAtm = self.inertial2Sat.applyTo(vAtm_i)

            aTorque = Vector3D.ZERO

            dragCoeff = self.meshDA['Cd']
            liftRatio = 0.0  # no lift considered

            iterator = itertools.izip(self.meshDA['CoM'],
                                      self.meshDA['Normal'],
                                      self.meshDA['Area'])

            for CoM, Normal, Area in iterator:
                CoMVelocity = satVel.add(self.V3_cross(self.omega, CoM))
                relativeVelocity = vAtm.subtract(CoMVelocity)

                vNorm2 = relativeVelocity.getNormSq()
                vNorm = sqrt(vNorm2)
                vDir = relativeVelocity.scalarMultiply(1.0 / vNorm)

                dot = self.V3_dot(Normal, vDir)
                if (dot < 0):
                    coeff = 0.5 * rho * dragCoeff * vNorm2
                    oMr = 1.0 - liftRatio
                    # dA intercepts the incoming flux
                    f = coeff * Area * dot
                    force = Vector3D(float(oMr * abs(f)), vDir,
                                     float(liftRatio * f * 2), Normal)
                    aTorque = aTorque.add(self.V3_cross(CoM, force))

            return aTorque

        else:
            return Vector3D.ZERO

    def getAeroTorqueArray(self):
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

        if self.to_add[3]:
            # assuming constant atmosphere condition over spacecraft
            # error is of order of 10^-17
            rho = self.AtmoModel.getDensity(self.refDate, self.satPos_i, self.refFrame)
            vAtm_i = self.AtmoModel.getVelocity(self.refDate, self.satPos_i, self.refFrame)

            satVel = self.inertial2Sat.applyTo(self.satVel_i)
            vAtm = self.inertial2Sat.applyTo(vAtm_i)

            dragCoeff = self.meshDA['Cd']
            liftRatio = 0.0  # no lift considered

            CoM = self.meshDA['CoM_np']
            normal = self.meshDA['Normal_np']
            area = np.asarray(self.meshDA['Area'])
            omega = np.array([self.omega.x, self.omega.y, self.omega.z])
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

                torque = np.sum(np.cross(CoM, oMr * np.absolute(f) * vDir + 2 * liftRatio * f * normal), axis=0)
            else:
                torque = np.zeros(3)

            aTorque = Vector3D(float(torque[0]), float(torque[1]), float(torque[2]))

            return aTorque

        else:
            return Vector3D.ZERO


class StateEquation(PSE):
    """Class in format to be used with Hipparchus library for integration.

    Integrates satellite's state equations and returns its new attitude
    in quaternions and it's angular rate along the principal axes
    """

    def __init__(self, Dimension):
        PSE.__init__(self, Dimension)

        self.torque_control = Vector3D.ZERO
        '''External torque provided by ROS Node.'''

        self.torque_dist = Vector3D.ZERO
        '''Disturbance torque computed before integration'''

        self.inertiaT = None
        '''Inertia Tensor given for principal axes.'''

    def init(self, t0, y0, finalTime):
        """No initialization needed"""

    def computeDerivatives(self, t, y):
        try:
            yDot = orekit.JArray('double')(self.getDimension())

            # angular velocity body rates (omega):
            yDot[0] = 1.0 / self.inertiaT[0][0] * \
                (self.torque_control.getX() + self.torque_dist.getX() +
                 (self.inertiaT[1][1] - self.inertiaT[2][2]) *
                 y[1] * y[2])

            yDot[1] = 1.0 / self.inertiaT[1][1] * \
                (self.torque_control.getY() + self.torque_dist.getY() +
                 (self.inertiaT[2][2] - self.inertiaT[0][0]) *
                 y[2] * y[0])

            yDot[2] = 1.0 / self.inertiaT[2][2] * \
                (self.torque_control.getZ() + self.torque_dist.getZ() +
                 (self.inertiaT[0][0] - self.inertiaT[1][1]) *
                 y[0] * y[1])

            # attitude quaternion:
            yDot[3] = 0.5 * (y[2] * y[4] - y[1] * y[5] + y[0] * y[6])
            yDot[4] = 0.5 * (-y[2] * y[3] + y[0] * y[5] + y[1] * y[6])
            yDot[5] = 0.5 * (y[1] * y[3] - y[0] * y[4] + y[2] * y[6])
            yDot[6] = 0.5 * (-y[0] * y[3] - y[1] * y[4] - y[2] * y[5])

            return yDot

        except Exception as err:
            print str(err)
            raise


class SingleStepEq(object):
    '''Class performs RK-single step.

    Used for profiler, so that it shows how much time spend in this class'''

    def __init__(self):
        self.RK = ClassicalRungeKuttaIntegrator(float(1.0))

    def integrate(self, equations, t0, y0, t):
        result = self.RK.singleStep(equations, t0, y0, t)

        return result


class PyRotation(object):
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
