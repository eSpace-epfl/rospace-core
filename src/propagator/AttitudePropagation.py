import orekit
import numpy as np
import itertools
from math import sqrt
import sys  # for errors

from org.orekit.attitudes import Attitude
from org.orekit.python import PythonAttitudePropagation as PAP
from org.orekit.python import PythonStateEquation as PSE
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
    def __init__(self, attitude, referenceDate, inertiaT, tol, intSettings, inCub, mesh_dA, AttitudeFM):
        PAP.__init__(self, attitude)

        self.omega = np.zeros(3)

        # Force Models needed for gravity gradient Torques & inertia tensor:
        self.StateObserver = AttitudeFM['StateObserver']

        # assuming inertia Tensor given for principal axes and
        # linerly dependent on mass
        self.inertiaT = inertiaT / self.StateObserver.spacecraftState.getMass()
        self.refDate = referenceDate
        self.refFrame = attitude.getReferenceFrame()

        if 'GravityModel' in AttitudeFM:  # gravitational disturbance active
            self.add_gDist = True
            self.GravityModel = AttitudeFM['GravityModel']
        else:
            self.add_gDist = False
            self.GravityModel = None

        self.inCub = inCub
        self.mesh_dA = mesh_dA

        if 'SolarModel' in AttitudeFM:  # Torque due to Solar Pressure active
            self.add_solarP = True
            self.SolarModel = AttitudeFM['SolarModel']
            self.Sun = PVCoordinatesProvider.cast_(AttitudeFM['Sun'])

            # taken from orekit:
            # Reference distance for the solar radiation pressure (m).
            D_REF = float(149597870000.0)
            # Reference solar radiation pressure at D_REF (N/m^2).
            P_REF = float(4.56e-6)
            # Reference flux normalized for a 1m distance (N).
            self.K_REF = float(P_REF * D_REF * D_REF)
        else:
            self.add_solarP = False
            self.SolarModel = None

        if 'AtmoModel' in AttitudeFM:
            self.add_aeroD = True
            self.AtmoModel = Atmosphere.cast_(AttitudeFM['AtmoModel'])
        else:
            self.add_aeroD = False

        minStep = intSettings['minStep']
        maxStep = intSettings['maxStep']
        initStep = intSettings['initStep']
        if intSettings['absTol'] < 0.0:  # use tol from orbit propagator
            intAbsTol = orekit.JArray_double.cast_(tol[0])
        else:
            intAbsTol = orekit.JArray_double.cast_(intSettings['absTol'])
        if intSettings['relTol'] < 0.0:  # use tol from orbit propagator
            intRelTol = orekit.JArray_double.cast_(tol[1])
        else:
            intRelTol = orekit.JArray_double.cast_(intSettings['relTol'])

        self.integrator = \
            DormandPrince853Integrator(minStep, maxStep, intAbsTol, intRelTol)
        self.integrator.setInitialStepSize(initStep)

        self.state = StateEquation(7)

    def getAttitude(self, pvProv, date, frame):
        try:
            if self.refDate.equals(date):
                return self.getReferenceAttitude()
            else:
                self.state.inertiaT = self.inertiaT * \
                                self.StateObserver.spacecraftState.getMass()
                self.state.torque_control = self.getExternalTorque()
                gTorque = self.getGravTorque(date)
                sTorque = self.getSolarTorque(date)
                aTorque = self.getAeroTorque(date)
                self.state.torque_dist = gTorque.add(sTorque.add(aTorque))
                self.state.omega = self.omega

                # subtract reference Date from date
                dt = date.durationFrom(self.refDate)
                y = self.getInitialState()
                # start at t zero and go integrate over time difference dt
                initial_state = ODEState(float(0.0), y)
                ode = ExpandableODE(OrdinaryDifferentialEquation.cast_(
                                                                   self.state))

                try:
                    new_state = self.integrator.integrate(ode,
                                                          initial_state,
                                                          dt)
                except MathIllegalArgumentException as illArg:
                    raise illArg
                except MathIllegalStateException as illStat:
                    raise illStat

                y = new_state.getPrimaryState()

                # update values
                spin = Vector3D(y[0], y[1], y[2])
                torque = self.state.torque_dist.add(self.state.torque_control)
                rot = Rotation(float(y[6]),
                               float(y[3]),
                               float(y[4]),
                               float(y[5]), True)
                newAttitude = Attitude(date,
                                       self.refFrame,
                                       rot,
                                       spin,
                                       torque)
                self.refDate = date
                self.omega = np.array([y[0], y[1], y[2]])
                self.setReferenceAttitude(newAttitude)

                return newAttitude

        except Exception:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print exc_type
                print exc_value
                raise

    def getInitialState(self):

        y = np.zeros(7)

        y[0:3] = self.omega  # angular momentum
        rotation = self.getReferenceAttitude().getRotation()
        # get attitude in quaternions:
        # scalar part is Q0 in orekit, but q4 in integration
        y[3] = rotation.getQ1()
        y[4] = rotation.getQ2()
        y[5] = rotation.getQ3()
        y[6] = rotation.getQ0()

        return orekit.JArray('double')(y)

    def getGravTorque(self, date):
        if self.add_gDist:
            spacecraftState = self.StateObserver.spacecraftState
            muGM = ForceModel.cast_(self.GravityModel).getParameters()[0]

            satPos = spacecraftState.getPVCoordinates().getPosition()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            satM = spacecraftState.getMass()

            mCub = self.inCub['mass_frac'] * satM
            gTorque = Vector3D.ZERO

            for CoM in self.inCub['CoM']:
                # get absolute coordinates of CoM in inertial frame:
                I_dm = inertial2Sat.applyInverseTo(CoM)
                I_dmPos = satPos.add(I_dm)

                # Newtonian Attraction:
                r2 = I_dmPos.getNormSq()
                gNewton = Vector3D(-muGM / (sqrt(r2) * r2), I_dmPos)
                # Perturbating part of gravity gradient:
                gDist = Vector3D(self.GravityModel.gradient(date,
                                                            I_dmPos,
                                                            muGM))

                dmForce = Vector3D(mCub, gNewton.add(gDist))
                gTorque = gTorque.add(Vector3D.crossProduct(I_dm, dmForce))

            # return gravity gradient torque in satellite frame
            return inertial2Sat.applyTo(gTorque)

        else:
            return Vector3D.ZERO

    """
    since necessary orekit methods cannot be accessed directly without
    creating an Spacecraft object, and even then information would not be
    complete this method copies parts of the acceleration() method of
    the SolarRadiationPressure and radiationPressureAcceleration() of
    the BoxAndSolarArraySpacecraft class
    """
    def getSolarTorque(self, date):
        if self.add_solarP:
            spacecraftState = self.StateObserver.spacecraftState
            frame = spacecraftState.getFrame()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            satPos = spacecraftState.getPVCoordinates().getPosition()

            mesh_CoM = self.mesh_dA['CoM']
            mesh_N = self.mesh_dA['Normal']
            mesh_A = self.mesh_dA['Area']
            mesh_Coef = self.mesh_dA['Coefs']
            iterator = itertools.izip(mesh_CoM, mesh_N, mesh_A, mesh_Coef)

            # this could be very costly for large mesh. -> parallelize ?
            for CoM, normal, area, coefs in iterator:
                position = satPos.add(inertial2Sat.applyInverseTo(CoM))
                sunSatVector = position.subtract(
                        self.Sun.getPVCoordinates(date, frame).getPosition())
                r2 = sunSatVector.getNormSq()

                # compute flux in inertial frame
                ratio = self.SolarModel.getLightingRatio(position, frame, date)
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
                    if diffuseReflCoeff < 0:
                        print "Error: Negative Coefficient not possible!"
                        print "diffuseReflCoeff= ", diffuseReflCoeff, " < 0 !"
                        raise

                    psr = fluxSat.getNorm()

                    # Vallado's equation uses different arameters which are
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

    """
    since necessary orekit methods cannot be accessed directly without
    creating an Spacecraft object, and even then information would not be
    complete this method copies parts of the acceleration() method of
    the DragForce and dragAcceleration() of the BoxAndSolarArraySpacecraft
    class
    """
    def getAeroTorque(self, date):
        if self.add_aeroD:
            spacecraftState = self.StateObserver.spacecraftState
            frame = spacecraftState.getFrame()
            inertial2Sat = spacecraftState.getAttitude().getRotation()
            satPos = spacecraftState.getPVCoordinates().getPosition()
            satVel = spacecraftState.getPVCoordinates().getVelocity()

            # rotation rate of spacecraf:
            omega = Vector3D(float(self.omega[0]),
                             float(self.omega[1]),
                             float(self.omega[2]))

            # assuming constant atmosphere condition over spacecraft
            # error is of order of 10^-17
            rho = self.AtmoModel.getDensity(date, satPos, frame)
            vAtm = self.AtmoModel.getVelocity(date, satPos, frame)

            force = Vector3D.ZERO
            aTorque = Vector3D.ZERO

            dragCoeff = self.mesh_dA['Cd']
            liftRatio = 0.0  # no lift considered

            mesh_CoM = self.mesh_dA['CoM']
            mesh_N = self.mesh_dA['Normal']
            mesh_A = self.mesh_dA['Area']
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

    # def writeToFile(self, direct, writelist):
    #     with open(direct, "w") as f:
    #             for s in writelist:
    #                 f.write(str(s) + "\n")


# Differential Equations object:
class StateEquation(PSE):
    def __init__(self, Dimension):
        PSE.__init__(self, Dimension)

        self.torque_control = None
        self.torque_dist = None
        self.inertiaT = None
        self.omega = None

    def init(self, t0, y0, finalTime):
        pass

    def computeDerivatives(self, t, y):
        try:
            yDot = np.zeros(self.getDimension())

            # angular velocity body rates (omega):
            yDot[0] = 1.0 / self.inertiaT[0][0] * \
                      (self.torque_control.getX() + self.torque_dist.getX() +
                       (self.inertiaT[1][1] - self.inertiaT[2][2]) *
                       y[1]*y[2])

            yDot[1] = 1.0 / self.inertiaT[1][1] * \
                     (self.torque_control.getY() + self.torque_dist.getY() +
                      (self.inertiaT[2][2] - self.inertiaT[0][0]) *
                      y[2]*y[0])

            yDot[2] = 1.0 / self.inertiaT[2][2] * \
                      (self.torque_control.getZ() + self.torque_dist.getZ() +
                       (self.inertiaT[0][0] - self.inertiaT[1][1]) *
                       y[0]*y[1])

            # attitude quaternion:
            yDot[3] = 0.5 * (y[2]*y[4] - y[1]*y[5] + y[0]*y[6])
            yDot[4] = 0.5 * (-y[2]*y[3] + y[0]*y[5] + y[1]*y[6])
            yDot[5] = 0.5 * (y[1]*y[3] - y[0]*y[4] + y[2]*y[6])
            yDot[6] = 0.5 * (-y[0]*y[3] - y[1]*y[4] - y[2]*y[5])

            return orekit.JArray('double')(yDot)

        except Exception as err:
            print str(err)
            raise
