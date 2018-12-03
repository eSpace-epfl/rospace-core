# Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# SPDX-License-Identifier: Zlib
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details. The contributors to this file maybe
# found in the SCM logs or in the AUTHORS.md file.

import orekit  # need thist otherwise cannot import other stuff

from org.orekit.python import BaseForceModel

from java.util.stream import Stream


class StateObserver(BaseForceModel):
    """
    This class is needed to get the spacecraft state during
    propagation (at every integration step), which is needed
    for attitude propagation

    addContribution() here doesn't change the state of the
    spacecraft. It just updates a object which can be obtained
    by extern methods
    """

    def __init__(self, initialState):
        BaseForceModel.__init__(self, "StateObserver")

        self.spacecraftState = initialState

    def dependsOnPositionOnly(self):
        return False

    def init(self, s0, t):
        pass

    def addContribution(self, s, adder):
        '''Store current spacecraft state.

        This method is called by propagator during integration at
        every timestep (after orbit, attitude and mass update)
        spacecraft state stored here
        '''
        self.spacecraftState = s

    # this method is called by propagator once before propagation
    def getEventsDetectors(self):
        '''Returns empty stream.

        This method is called by propagator once before propagation
        '''
        return Stream.empty()

    def acceleration(self, state, parameters):
        '''This method should never be called.

        only called by Force Model interface, if! getParameters()
        method called
        '''
        pass

    def getParametersDrivers(self):
        return None
