# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import orekit
import numpy as np
import traceback

from DisturbanceTorques import DisturbanceTorqueArray as DTarray

from org.orekit.attitudes import Attitude
from org.orekit.python import PythonAttitudePropagation as PAP
from org.orekit.python import PythonStateEquation as PSE

from org.hipparchus.ode import ODEState
from org.hipparchus.ode import OrdinaryDifferentialEquation, ExpandableODE
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator
from org.hipparchus.geometry.euclidean.threed import Rotation, Vector3D

from org.hipparchus.exception import MathIllegalArgumentException
from org.hipparchus.exception import MathIllegalStateException


class AttitudePropagation(PAP):
    """Implements an attitude propagation which is called by Orekit's attitude provider.

    The integration of the attitude is embedded into Orekit's orbit integration
    and the computation is done before every orbit integration step.
    For attitude integration two integrators from the Hipparchus library are
    being initialized for performance reasons (see :class:`AttitudePropagation.Integrators`).
    For small orbit time steps a single Runge-Kutta integrator is used. For larger orbit
    integration steps the Dormand Prince integrator solves the equation of motion.
    The maximal step size for which the single step RK integrator is used can be
    set in the settings yaml-file.

    The maximal step size should not be set to a larger value since the simulation
    could fail due to numerical instability.

    Args:
        attitude: initial attitude object from spacecraft
        referenceDate: initial epoch as AbsoluteDate object
        inertiaT: numpy array of diagonal entries of inertia tensor
        tol: tolerances for Dormand Prince integrator
        intSettings: dictionary with integrator settings
        inCub: dictionary with information about satellite's inner body discretization
        meshDA: dictionary with information about satellite's surface discretization
        AttitdueFM: dictionary with Force Model objects needed for disturbance torques

    """

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
        '''Rotation object of spacecrafts attitude'''

        self.integrator = Integrators(intSettings, tol)
        '''DormandPrince853 & classical RK object for attitude propagation.'''

        self.StateObserver = AttitudeFM['StateObserver']
        '''State observer object added to satellite's orbit propagator as Force Model.'''

        self.inertiaT = inertiaT / self.StateObserver.spacecraftState.getMass()
        '''Inertial tensor linearly dependent on mass given for principal axes.'''

        self.refDate = referenceDate
        '''Date at current satellite state'''

        self.refFrame = attitude.getReferenceFrame()
        '''Reference frame in which attitude is computed'''

        self.DT = DTarray(self.refFrame,
                          self.refDate,
                          inCub,
                          meshDA,
                          AttitudeFM)
        '''Disturbance torque object'''

        self.state = StateEquation(7, self.DT)
        '''StateEquation object. Holds 7 equations to be integrated.'''

    def getAttitude(self, pvProv, date, frame):
        """Method called by Orekit at every orbit integration step.

        This Method updates the the satellite state and rotations in the
        disturbance torque class, calls the integrators with the updated
        state vector and stores and returns the new Attitude with the new
        spin vector.

        The parameters of the state equation are the angular rate along
        the satellite's principle axes (1:3) and the rotation given in
        quaternions (4:7)

        The date provided as argument must no necessarily be advancing in
        time. Also earlier time can be given, resulting in a negative time
        difference.

        Args:
            pvProv: the satellites orbit given as PVCoordinatesProvider object
            date: AbsoluteDate object to which attitude should be integrated
            frame: Frame in which attitude should be given

        Returns:
            Attitude: new attitude @ date

        Raises:
            Exception: if something went wrong in JAVA classes the traceback is printed
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
                # scalar part is q0 in Orekit, but q4 in integration
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
    in quaternions and it's angular rate along the principal axes.

    The computeDerivatives method overridden in this class, computes the
    EoM of satellite's attitude based on equations (17-1a), (17-1b), (17-2),
    (17-3) from the book:
    'SPACECRAFT ATTITUDE DETERMINATION AND CONTROL' by James R. Wertz

    The inertial tensor provided has to be a diagonal matrix.

    Constructor Args:
        Dimension: dimension of state vector
        DT_instance: disturbance torque object
    """

    def __init__(self, Dimension, DT_instance):
        PSE.__init__(self, Dimension)

        self.torque_control = Vector3D.ZERO
        '''External torque provided by ROS Node.'''

        self.inertiaT = None
        '''Inertia Tensor given for principal axes. Constant through integration'''

        self.DistTorque = DT_instance
        '''Disturbance torque object'''

    def init(self, t0, y0, finalTime):
        """No initialization needed"""

    def computeDerivatives(self, t, y):
        '''Get the current time derivative of the state vector.

        Args:
            t: current value of the independent time variable
            y: array containing the current value of the state vector
        Returns:
            JArray: time derivative of the state vector
        '''
        try:
            # update rotation and compute torque at new attitude
            in2Sat_rotation = Rotation(float(y[6]),
                                       float(y[3]),
                                       float(y[4]),
                                       float(y[5]), True)

            omega = np.array([y[0], y[1], y[2]])
            DT = self.DistTorque.compute_torques(in2Sat_rotation, omega, t)

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

            self.t_last = t

            return yDot

        except Exception as err:
            print str(err)
            raise


class Integrators(object):
    """Class holding a classical Runge Kutta and a DormandPrinve853 integrator.

    This class is integrating the state equations either using a single step
    integration with a Runge Kutta integrator or using multiple variable steps
    with a DormandPrince853 integrator from the Hipparchus library.

    For small orbit integration steps a single step Runge-Kutta integrator is
    used. For larger steps the DormandPrince.
    For time steps smaller than variable maxDT a single step integration is performed,
    for time steps larger than maxDT a multiple variable steps integration is used.

    Constructor Args:
        intSettings: dictionary holding DormandPrince853 integrator settings
        tol: tolerances for DormandPrince853 integrator
    """

    def __init__(self, intSettings, tol):
        self.maxDT = intSettings['maxDT']
        '''Value representing maximal step size for which a single step integrator is used.'''

        self.integrator_RK = ClassicalRungeKuttaIntegrator(float(1.0))  # 1.0 chosen arbitrarily
        '''Classical Runge Kutta integrator object'''

        self.integrator_DP853 = None
        '''DormandPrince853 integrator object'''

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
        """Intermediate method switching between integrator object depending on
        orbit time step and calling their integrate methods.

        For small orbit integration steps a single step Runge-Kutta integrator is
        used. For larger steps the DormandPrince integrator used be used otherwise
        simulation could become numerically instable.

        Args:
            state: state object holding differential equations to be integrated
            y0: JArray with initial values of state vector
            dt: time to which should be integrated

        Returns:
            JArray object: integrated state vector

        Raises:
            MathIllegalArgumentException: if integration step too small
            MathIllegalStateException: if the number of functions evaluations is exceeded
        """

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
