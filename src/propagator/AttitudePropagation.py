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
from math import sqrt
from math import degrees
import sys  # for errors

from org.orekit.attitudes import Attitude
from org.orekit.python import PythonAttitudePropagation as PAP
from org.orekit.python import PythonStateEquation as PSE
from org.orekit.bodies import BodyShape
from org.orekit.forces import ForceModel
from org.orekit.forces.drag.atmosphere import Atmosphere
from org.orekit.utils import PVCoordinatesProvider

from org.hipparchus.util import Precision
from org.hipparchus.ode import ODEState
from org.hipparchus.ode import ExpandableODE, OrdinaryDifferentialEquation
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Rotation, Vector3D
from org.hipparchus.exception import MathIllegalArgumentException
from org.hipparchus.exception import MathIllegalStateException


class AttitudePropagation(PAP):
    """Implements an attitude propagation which is called by Orekit's attitude provider."""

    @staticmethod
    def _set_up_attitude_integrator(intSettings, tol):
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

        self.omega = np.zeros(3)
        '''Angular velocity of satellite in direction of principle axes.'''

        self.state = StateEquation(7)
        '''StateEquation object. Holds 7 equations to be integrated.'''

        self.integrator = self._set_up_attitude_integrator(intSettings, tol)
        '''DormandPrince853Integrator object.'''

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
                gTorque = self.getGravTorque()
                mTorque = self.getMagTorque()
                sTorque = self.getSolarTorque()
                aTorque = self.getAeroTorque()

                self.state.torque_dist = gTorque.add(
                    mTorque.add(
                        sTorque.add(
                            aTorque)))
                self.state.omega = self.omega

                ode = ExpandableODE(
                    OrdinaryDifferentialEquation.cast_(self.state))
                y = self._convert_initial_state_to_JArray()
                initial_state = ODEState(float(0.0), y)
                dt = date.durationFrom(self.refDate)  # refDate - date
                try:
                    new_state = self.integrator.integrate(ode,
                                                          initial_state,
                                                          dt)

                except MathIllegalArgumentException as illArg:
                    raise illArg
                except MathIllegalStateException as illStat:
                    raise illStat

                y = new_state.getPrimaryState()  # primary state from ODEState

                # update and store computed values
                addG = False if self.GravityModel is None else True
                addM = False if self.MagneticModel is None else True
                addSP = False if self.SolarModel is None else True
                addD = False if self.AtmoModel is None else True
                self.setAddedDisturbanceTorques(addG, addM, addSP, addD)
                self.setDisturbanceTorques(gTorque, mTorque, sTorque, aTorque)

                # spin = Vector3D(y[0], y[1], y[2])
                # torque = self.state.torque_dist.add(self.state.torque_control)
                rot = Rotation(float(y[6]),
                               float(y[3]),
                               float(y[4]),
                               float(y[5]), True)
                newAttitude = Attitude(date,
                                       self.refFrame,
                                       rot,
                                       Vector3D.ZERO,
                                       Vector3D.ZERO)
                self.refDate = date
                self.omega = np.array([y[0], y[1], y[2]])
                self.setReferenceAttitude(newAttitude)

                return newAttitude

        except Exception:  # should never get here
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print exc_type
            print exc_value
            raise

    def _convert_initial_state_to_JArray(self):
        """Method to get current initial state and convert it to JArray object

        Returns:
            JArray('double'): state vector to be integrated
        """

        y = np.zeros(7)

        y[0:3] = self.omega  # angular momentum
        rotation = self.getReferenceAttitude().getRotation()

        # get rotation in quaternions:
        # scalar part is Q0 in Orekit, but q4 in integration
        y[3] = rotation.getQ1()
        y[4] = rotation.getQ2()
        y[5] = rotation.getQ3()
        y[6] = rotation.getQ0()

        return orekit.JArray('double')(y)

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

        if self.GravityModel is not None:
            spacecraftState = self.StateObserver.spacecraftState
            satPos = spacecraftState.getPVCoordinates().getPosition()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            satM = spacecraftState.getMass()

            muGM = ForceModel.cast_(self.GravityModel).getParameters()[0]
            mCub = self.inCub['mass_frac'] * satM

            gTorque = Vector3D.ZERO

            for CoM in self.inCub['CoM']:
                # get absolute coordinates of CoM in inertial frame:
                I_dm = inertial2Sat.applyInverseTo(CoM)
                I_dmPos = satPos.add(I_dm)

                # Newtonian Attraction:
                r2 = I_dmPos.getNormSq()
                gNewton = Vector3D(-muGM / (sqrt(r2) * r2), I_dmPos)

                # Perturbing part of gravity gradient:
                gDist = Vector3D(self.GravityModel.gradient(self.refDate,
                                                            I_dmPos,
                                                            muGM))
                dmForce = Vector3D(mCub, gNewton.add(gDist))
                gTorque = gTorque.add(Vector3D.crossProduct(I_dm, dmForce))

            # return gravity gradient torque in satellite frame
            return inertial2Sat.applyTo(gTorque)

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

        if self.SolarModel is not None:

            spacecraftState = self.StateObserver.spacecraftState
            satPos = spacecraftState.getPVCoordinates().getPosition()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            frame = spacecraftState.getFrame()

            mesh_CoM = self.meshDA['CoM']
            mesh_N = self.meshDA['Normal']
            mesh_A = self.meshDA['Area']
            mesh_Coef = self.meshDA['Coefs']

            iterator = itertools.izip(mesh_CoM, mesh_N, mesh_A, mesh_Coef)

            for CoM, normal, area, coefs in iterator:
                position = satPos.add(inertial2Sat.applyInverseTo(CoM))

                # compute flux in inertial frame
                sunSatVector = \
                    position.subtract(self.sun.getPVCoordinates(self.refDate,
                                                                frame).getPosition())
                r2 = sunSatVector.getNormSq()
                ratio = self.SolarModel.getLightingRatio(position,
                                                         frame,
                                                         self.refDate)
                rawP = ratio * self.K_REF / r2
                flux = Vector3D(rawP / sqrt(r2), sunSatVector)

                spTorque = Vector3D.ZERO
                # compute Radiation Pressure Force:
                if flux.getNormSq() > Precision.SAFE_MIN:
                    # illumination (we are not in umbra)
                    # rotate flux to spacecraft frame:
                    fluxSat = inertial2Sat.applyTo(flux)
                    dot = Vector3D.dotProduct(normal, fluxSat)

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
                    psr = fluxSat.getNorm()
                    # Vallado's equation uses different parameters which are
                    # related to our parameters as:
                    # cos (phi) = - dot / (psr*area)
                    # n         = N                     (n...unit vector)
                    # s         = -fluxSat / psr        (s...unit vector)
                    cN = 2 * area * dot * (diffuseReflCoeff / 3 -
                                           specularReflCoeff * dot / psr)
                    cS = (area * dot / psr) * (specularReflCoeff - 1)
                    Force = Vector3D(float(cN), normal, float(cS), fluxSat)
                    # Force already in spacecraft frame. No need to convert
                    spTorque = spTorque.add(Vector3D.crossProduct(CoM, Force))

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

        if self.AtmoModel is not None:
            spacecraftState = self.StateObserver.spacecraftState
            satPos = spacecraftState.getPVCoordinates().getPosition()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            satVel = spacecraftState.getPVCoordinates().getVelocity()
            frame = spacecraftState.getFrame()

            omega = Vector3D(float(self.omega[0]),
                             float(self.omega[1]),
                             float(self.omega[2]))

            # assuming constant atmosphere condition over spacecraft
            # error is of order of 10^-17
            rho = self.AtmoModel.getDensity(self.refDate, satPos, frame)
            vAtm = self.AtmoModel.getVelocity(self.refDate, satPos, frame)

            force = Vector3D.ZERO
            aTorque = Vector3D.ZERO

            dragCoeff = self.meshDA['Cd']
            liftRatio = 0.0  # no lift considered

            mesh_CoM = self.meshDA['CoM']
            mesh_N = self.meshDA['Normal']
            mesh_A = self.meshDA['Area']

            iterator = itertools.izip(mesh_CoM, mesh_N, mesh_A)

            for CoM, Normal, Area in iterator:
                CoMVelocity = satVel.add(Vector3D.crossProduct(omega, CoM))
                relativeVelocity = vAtm.subtract(CoMVelocity)

                vNorm2 = relativeVelocity.getNormSq()
                vNorm = sqrt(vNorm2)
                vDir = inertial2Sat.applyTo(
                    relativeVelocity.scalarMultiply(1.0 / vNorm))

                coeff = 0.5 * rho * dragCoeff * vNorm2
                oMr = 1.0 - liftRatio

                dot = Vector3D.dotProduct(Normal, vDir)
                if (dot < 0):
                    # dA intercepts the incoming flux
                    f = coeff * Area * dot
                    force = Vector3D(float(1.0), force,
                                     float(oMr * abs(f)), vDir,
                                     float(liftRatio * f * 2), Normal)
                    aTorque = aTorque.add(Vector3D.crossProduct(CoM, force))

            return aTorque

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
        if self.MagneticModel is not None:
            spacecraftState = self.StateObserver.spacecraftState
            satPos = spacecraftState.getPVCoordinates().getPosition()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            frame = spacecraftState.getFrame()

            gP = self.earth.transform(satPos, frame, self.refDate)
            lat = degrees(gP.getLatitude())
            lon = degrees(gP.getLongitude())
            alt = gP.getAltitude() / 1e3  # Mag. Field needs degrees and [km]

            B_i = self.MagneticModel.calculateField(lat, lon, alt)
            B_b = inertial2Sat.applyTo(B_i.getFieldVector())
            B_b = Vector3D(float(1e-9), B_b)   # convert B_i from [nT] to [T]
            dipole = self.getDipoleVector()

            return dipole.crossProduct(B_b)

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

        self.omega = None
        '''Rotation rates along principal axes'''

    def init(self, t0, y0, finalTime):
        """No initialization needed"""
        pass

    def computeDerivatives(self, t, y):
        try:
            yDot = np.zeros(self.getDimension())

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

            return orekit.JArray('double')(yDot)

        except Exception as err:
            print str(err)
            raise
