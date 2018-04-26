# @copyright Copyright (c) 2017, Davide Frey (frey.davide.ae@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

"""Class containing a propagator object that holds the orekit propagator itself and other useful information."""

import sys
import yaml

from datetime import datetime

from propagator.FileDataHandler import FileDataHandler
from propagator.OrekitPropagator import OrekitPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.utils import PVCoordinates
from org.orekit.utils import Constants as Cst
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory


class Propagator(object):
    """
        Class that holds the definition of the orekit propagator.

    Attributes:
        orekit_prop (OrekitPropagator): The propagator created from orekit.
        prop_type (str): Propagator type.
        date (datetime): Propagator actual date.

    """

    def __init__(self):
        # Initialize Propagator settings
        OrekitPropagator.init_jvm()

        self.orekit_prop = OrekitPropagator()
        self.prop_type = ''
        self.date = None

    def initialize_propagator(self, name, initial_state, prop_type, start_date=datetime.utcnow()):
        """
            Initialize the propagator.

        Args:
            name (str): Name of the satellite (connected to cfg files).
            initial_state (KepOrbElem): Initial osculating orbital elements of the satellite.
            prop_type (str): Propagator type. Can be either 2-body or real-world (depending on which one a cfg file is
                chosen).
            start_date (datetime): Initial date of the propagator.
        """
        # Set date
        self.date = start_date

        # Set type
        self.prop_type = prop_type

        if prop_type == '2-body':
            name += '_2body'

        # Set magnetic field to the propagator date
        # FileDataHandler.load_magnetic_field_models(self.date)
        # FileDataHandler.create_data_validity_checklist()

        # Open the configuration file
        abs_path = sys.argv[0]
        path_idx = abs_path.find('planning')
        abs_path = abs_path[0:path_idx]
        settings_path = abs_path + 'rdv-cap-sim/rospace_simulator/cfg/' + name + '.yaml'
        settings = file(settings_path, 'r')
        propSettings = yaml.load(settings)

        # Initialize propagator
        self.orekit_prop.initialize(propSettings['propagator_settings'], initial_state, self.date)

    def change_initial_conditions(self, initial_state, date, mass):
        """
            Allows to change the initial conditions given to the propagator without initializing it again.

        Args:
            initial_state (Cartesian): New initial state of the satellite in cartesian coordinates.
            date (datetime): New starting date of the propagator.
            mass (float64): New satellite mass.
        """
        # Redefine the start date
        self.date = date

        # Create position and velocity vectors as Vector3D
        p = Vector3D(float(initial_state.R[0]) * 1e3, float(initial_state.R[1]) * 1e3,
                     float(initial_state.R[2]) * 1e3)
        v = Vector3D(float(initial_state.V[0]) * 1e3, float(initial_state.V[1]) * 1e3,
                     float(initial_state.V[2]) * 1e3)

        # Initialize orekit date
        seconds = float(date.second) + float(date.microsecond) / 1e6
        orekit_date = AbsoluteDate(date.year,
                                   date.month,
                                   date.day,
                                   date.hour,
                                   date.minute,
                                   seconds,
                                   TimeScalesFactory.getUTC())

        # Extract frame
        inertialFrame = FramesFactory.getEME2000()

        # Evaluate new initial orbit
        initialOrbit = CartesianOrbit(PVCoordinates(p, v), inertialFrame, orekit_date, Cst.WGS84_EARTH_MU)

        # Create new spacecraft state
        newSpacecraftState = SpacecraftState(initialOrbit, mass)

        # Rewrite propagator initial conditions
        self.orekit_prop._propagator_num.setInitialState(newSpacecraftState)
