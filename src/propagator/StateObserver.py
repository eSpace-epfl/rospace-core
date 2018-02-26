import orekit  # need thist otherwise cannot import other stuff

from org.orekit.python import BaseForceModel

from java.util.stream import Stream

###############################################################
# This class is needed to get spacecraft state during		  #
# propagation (at every integration step), which is needed	  #
# for attitude propagation									  #
# ------------------------------------------------------------#
# addContribution() here doesn't change the state of the 	  #
# spacecraft. It just updates a object which can be obtained  #
# by extern methods   										  #
###############################################################


class StateObserver(BaseForceModel):
    """OwnThrustModel"""
    def __init__(self, initialState):
        BaseForceModel.__init__(self, "StateObserver")

        self.spacecraftState = initialState

    def dependsOnPositionOnly(self):
        return False

    def init(self, s0, t):
        pass

    # this method is called by propagator during integration at
    # every timestep (after orbit, attitude and mass update)
    # spacecraft state stored here
    def addContribution(self, s, adder):
        self.spacecraftState = s

    # this method is called by propagator once before propagation
    def getEventsDetectors(self):
        return Stream.empty()

    def acceleration(self, state, parameters):
        pass

    # this method should never be called
    # only called by Force Model interface, if! getParameters()
    # method called
    def getParametersDrivers(self):
        return None
