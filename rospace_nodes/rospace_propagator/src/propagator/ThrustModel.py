# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import orekit  # need thist otherwise cannot import other orekit stuff

from orekit import JArray
from org.orekit.python import BaseForceModel
from org.orekit.utils import Constants, ParameterDriver
from org.orekit.forces import ForceModel
from org.hipparchus.geometry.euclidean.threed import Vector3D
from java.util.stream import Stream

################################################################
# Java-Wrapper Hacks:                                          #
#                                                              #
#   How to call method from interface:                         #
#   (example ForceModel -> getParameters() method)             #
#       def getParameters_test(self):                          #
#       return ForceModel.getParameters(ForceModel.cast_(self))#
#                                                              #
#                                                              #
# from Java it is only possible to change python class         #
# objects if method is defined as native!                      #
#                                                              #
################################################################


class ThrustModel(BaseForceModel):
    """
    Thrust Model based on Orekit's ConstantThrustManeuver class.

    Thrust Model inherits from the Java class BaseForceModel, with
    same properties and methods like Orekit's ConstantThrustManeuver.
    Only difference is that the variables holding the magnitude of the Thrust
    and its direction are not defined private. This enables to change thrust
    without rebuilding all Force models and adding them to propagator.

    To the BaseForceModel the string "ThrustModel" is given as argument.
    This ensures that this ForceModel can be distinguished from other classes
    inheriting from BaseForceModel.

    To change direction of Thrust the variable direction can be accessed
    directly through class as it is not set private.

    Args:
        Thrust: thrust magnitude in [N]
        Isp: mean specific impulse of maneuver
    """

    @property
    def mDot(self):
        return self._mDot

    def __init__(self, Thrust=None, Isp=None):
        BaseForceModel.__init__(self, "ThrustModel")

        # direction has to be normalized!
        self.direction = Vector3D(float(1), float(0), float(0))
        self.firing = False

        self._mDot = tuple([0., 0.])  # tuple for attitude propagation

        if Thrust is None:
            thrust = float(0.0)
        else:
            thrust = float(Thrust)

        if Isp is None:
            isp = 0
        else:
            isp = Isp

        if isp == 0:
            flowRate = 0.0
            thrust = 0.0
        else:
            flowRate = - thrust / (Constants.G0_STANDARD_GRAVITY * isp)

        # defined as in ConstantManeuverClass by Orekit:
        FLOW_RATE_SCALE = 1.0*(2**-12)
        THRUST_RATE_SCALE = 1.0*(2**-5)

        self.thrustDriver = ParameterDriver('thrust',
                                            thrust,
                                            float(FLOW_RATE_SCALE),
                                            float(0.0),
                                            float(100.0))

        self.flowRateDriver = ParameterDriver('flowRate',
                                              float(flowRate),
                                              float(THRUST_RATE_SCALE),
                                              float(-100.0),
                                              float(0.0))

    def dependsOnPositionOnly(self):
        """
        Doesn't depend on position only.

        Returns:
            False
        """

        return False

    def init(self, s0, t):
        """
        No initialization is set.

        Method only for completion here, since orekit's propagator calls
        init, before each propagation.

        Args:
            s0: initial spacecraft state
            t: time at start of integration
        """

        pass

    def getThrust(self):
        """
        Get magnitude of thrust.

        Returns:
            float: magnitude of thrust
        """

        return self.thrustDriver.getValue()

    def getFlowRate(self):
        """
        Get flow rate

        Returns:
            float: flow rate
        """

        return self.flowRateDriver.getValue()

    def getISP(self):
        """
        Get specific impulse

        Returns:
            float: specific impulse or 0 if not thrusting
        """

        thrust = self.getThrust()
        flowRate = self.getFlowRate()
        if flowRate < 0:
            return -thrust / (Constants.G0_STANDARD_GRAVITY * flowRate)
        else:
            return 0

    def addContribution(self, s, adder):
        """
        Compute acceleration from thrust and add it to derivatives.

        Args:
            s: spacecraft state
            adder: additional derivatives which are being integrated
        """

        if self.firing:
            parameter = ForceModel.getParameters(ForceModel.cast_(self))
            # thrust and flow rate should be on when firing..
            # no flow = no thrust
            if parameter[0] > 0.0 and parameter[1] < 0.0:
                adder.addNonKeplerianAcceleration(self.acceleration(s, parameter))
                adder.addMassDerivative(parameter[1])
                self._mDot = tuple([s.getDate(), parameter[1]])

    def getEventsDetectors(self):
        """
        No events detectors set by class.

        Returns:
            stream: empty Java stream
        """
        return Stream.empty()

    def acceleration(self, state, parameters):
        """
        Compute acceleration due to thrust.

        Args:
            state: spacecraft state
            parameters: parameters of Force Model (here: thrust and flow rate)

        Returns:
            Vector3D: acceleration vector based on thrust and mass of satellite
        """
        if self.firing:
            thrust = parameters[0]
            return Vector3D(thrust / state.getMass(),
                            state.getAttitude()
                                 .getRotation()
                                 .applyInverseTo(self.direction))
        else:
            return Vector3D.ZERO

    def getParametersDrivers(self):
        """
        Get Parameter drivers of class.

        Parameter drivers of ThrustModel class:
            - Thrust
            - Flow Rate

        Returns:
            JArray('object'): array with parameter drivers
        """

        pD = JArray('object')(2, ParameterDriver)
        pD[0] = self.thrustDriver
        pD[1] = self.flowRateDriver
        return pD

    def ChangeParameters(self, thrust, isp):
        """
        Change values of Parameter Drivers.

        If isp is set to zero also flow rate and thrust are set to zero.

        Args:
            thrust: magnitude of thrust
            isp: specific impulse
        """

        if isp == 0:
            flowRate = 0.0
            thrust = 0.0  # cannot thrust if no flow
        else:
            flowRate = -thrust / (Constants.G0_STANDARD_GRAVITY * isp)

        self.thrustDriver.setValue(float(thrust))
        self.flowRateDriver.setValue(float(flowRate))
