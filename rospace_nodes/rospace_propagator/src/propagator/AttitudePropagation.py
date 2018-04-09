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

import orekit
import numpy as np
import traceback

from DisturbanceTorques import DisturbanceTorqueArray as DTarray

from org.orekit.attitudes import Attitude
from org.orekit.python import PythonAttitudePropagation as PAP
from org.orekit.python import PythonStateEquation as PSE
from org.orekit.bodies import BodyShape
from org.orekit.forces import ForceModel
from org.orekit.forces.drag.atmosphere import Atmosphere
from org.orekit.utils import PVCoordinatesProvider

from org.hipparchus.ode import ODEState
from org.hipparchus.ode import OrdinaryDifferentialEquation, ExpandableODE
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator
from org.hipparchus.geometry.euclidean.threed import Rotation, Vector3D

from org.hipparchus.exception import MathIllegalArgumentException
from org.hipparchus.exception import MathIllegalStateException


class AttitudePropagation(PAP):
    """Implements an attitude propagation which is called by Orekit's attitude provider."""

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

        self.omega = attitude.getSpin()
        '''Angular velocity of satellite in direction of principle axes.'''

        self.rotation = attitude.getRotation()

        self.integrator = Integrators(intSettings, tol)
        '''DormandPrince853 & singleStep RK object for attitude propagation.'''

        self.StateObserver = AttitudeFM['StateObserver']
        '''Dictionary of force models for gravity torques & inertia tensor.'''

        self.inertiaT = inertiaT / self.StateObserver.spacecraftState.getMass()
        '''Inertial tensor linearly dependent on mass given for principal axes.'''

        self.refDate = referenceDate
        '''Date at current satellite state'''

        self.refFrame = attitude.getReferenceFrame()
        '''Reference frame in which attitude is computed'''

        self.DT = DTarray(self.StateObserver,
                          self.refFrame,
                          self.refDate,
                          inCub,
                          AttitudeFM,
                          meshDA)
        '''Disturbance torque object'''

        self.state = StateEquation(7, self.DT)
        '''StateEquation object. Holds 7 equations to be integrated.'''

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

                # set control torque
                self.state.torque_control = self.getExternalTorque()

                # prepare computation of disturbance torques
                self.DT.to_add = self.getAddedDisturbanceTorques()
                self.DT.update_satellite_state(self.refDate)

                y = orekit.JArray('double')(7)
                y[0] = self.omega.x  # angular momentum
                y[1] = self.omega.y
                y[2] = self.omega.z
                # get rotation in quaternions:
                # scalar part is Q0 in Orekit, but q4 in integration
                y[3] = self.rotation.q1
                y[4] = self.rotation.q2
                y[5] = self.rotation.q3
                y[6] = self.rotation.q0
                dt = date.durationFrom(self.refDate)  # refDate - date

                new_state = self.integrator.integrate(self.state, y, dt)

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

                # for publisher
                self.setDisturbanceTorques(self.DT.gTorque, self.DT.mTorque, self.DT.sTorque, self.DT.aTorque)

                self.refDate = date
                self.setReferenceAttitude(newAttitude)

                return newAttitude

        except Exception:  # should never get here
            print traceback.print_exc()
            raise


class StateEquation(PSE):
    """Class in format to be used with Hipparchus library for integration.

    Integrates satellite's state equations and returns its new attitude
    in quaternions and it's angular rate along the principal axes
    """

    def __init__(self, Dimension, DT_instance):
        PSE.__init__(self, Dimension)

        self.torque_control = Vector3D.ZERO
        '''External torque provided by ROS Node.'''

        self.inertiaT = None
        '''Inertia Tensor given for principal axes. Constant through integration'''

        self.in2Sat_rotation = None

        self.omega = None

        self.DistTorque = DT_instance

    def init(self, t0, y0, finalTime):
        """No initialization needed"""

    def computeDerivatives(self, t, y):
        try:
            # update rotation and compute torque at new attitude
            self.in2Sat_rotation = Rotation(float(y[6]),
                                            float(y[3]),
                                            float(y[4]),
                                            float(y[5]), True)

            self.omega = np.array([y[0], y[1], y[2]])
            DT = self.DistTorque.compute_torques(self.in2Sat_rotation, self.omega)

            yDot = orekit.JArray('double')(self.getDimension())

            # angular velocity body rates (omega):
            yDot[0] = 1.0 / self.inertiaT[0][0] * \
                (self.torque_control.getX() + DT.getX() +
                 (self.inertiaT[1][1] - self.inertiaT[2][2]) *
                 y[1] * y[2])

            yDot[1] = 1.0 / self.inertiaT[1][1] * \
                (self.torque_control.getY() + DT.getY() +
                 (self.inertiaT[2][2] - self.inertiaT[0][0]) *
                 y[2] * y[0])

            yDot[2] = 1.0 / self.inertiaT[2][2] * \
                (self.torque_control.getZ() + DT.getZ() +
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


class Integrators(object):
    '''Class performs RK-single step.'''

    def __init__(self, intSettings, tol):
        self.maxDT = intSettings['maxDT']

        self.integrator_RK = ClassicalRungeKuttaIntegrator(float(1.0))  # 1.0 chosen arbitrarily

        minStep = intSettings['minStep']
        maxStep = intSettings['maxStep']
        initStep = intSettings['initStep']

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
        self.integrator_DP853 = \
            DormandPrince853Integrator(minStep, maxStep, intAbsTol, intRelTol)
        self.integrator_DP853.setInitialStepSize(initStep)

    def integrate(self, state, y0, dt):

        if abs(dt) <= self.maxDT:
            equations = OrdinaryDifferentialEquation.cast_(state)
            result = self.integrator_RK.singleStep(equations, float(0.0), y0, dt)
        else:
            ode = ExpandableODE(
                        OrdinaryDifferentialEquation.cast_(state))
            initial_state = ODEState(float(0.0), y0)
            try:
                new_state = self.integrator_DP853. \
                                 integrate(ode, initial_state, dt)

            except MathIllegalArgumentException as illArg:
                raise illArg
            except MathIllegalStateException as illStat:
                raise illStat

            result = new_state.getPrimaryState()  # primary state from ODEState

        return result
