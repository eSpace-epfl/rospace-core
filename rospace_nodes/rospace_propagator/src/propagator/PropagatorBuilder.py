# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import abc
import numpy as np
import math

import FileDataHandler
from ThrustModel import ThrustModel
from AttitudePropagation import AttitudePropagation
from StateObserver import StateObserver
from SatelliteDiscretization import DiscretizationInterface as DiscInterface

from org.orekit.python import PythonEventHandler

from org.orekit.utils import PVCoordinatesProvider

import orekit
from orekit import JArray_double, JArray
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants as Cst
from org.orekit.utils import IERSConventions as IERS
from org.orekit.utils import PVCoordinates
from org.orekit.time import TimeScalesFactory
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.bodies import CelestialBody
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngle
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.forces import BoxAndSolarArraySpacecraft
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.propagation.events import EclipseDetector
from org.orekit.propagation.events.handlers import EventHandler
from org.orekit.forces.radiation import IsotropicRadiationClassicalConvention
from org.orekit.forces.drag import DragForce
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.gravity import SolidTides, OceanTides, Relativity
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity.potential import EGMFormatReader, ICGEMFormatReader
from org.orekit.forces.drag.atmosphere import DTM2000
from org.orekit.forces.drag.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.forces.drag.atmosphere.data import CelesTrackWeather
from org.orekit.attitudes import NadirPointing, Attitude
from org.orekit.data import DataProvidersManager

from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D, Rotation


def _build_default_gravity_Field(methodName):
    """
    Build gravity field using Normalized Provider with degree and order of 10.

    Gravity model used: eigen-6s

    Args:
        methodName: name of method calling this function (for printing warning)

    Returns:
        NormalizedSphericalHarmonicsProvider: provids norm. spherical harmonics

    """
    gfReader = EGMFormatReader('eigen-6s.gfc', False)
    GravityFieldFactory.addPotentialCoefficientsReader(gfReader)

    mesg = "\033[93m  [WARN] [Builder." + methodName\
           + "]: No Gravity Field defined. Creating default using"\
           + " NormalizedProvider of degree and order 10.\033[0m"

    print mesg

    return GravityFieldFactory.getNormalizedProvider(10, 10)


def _build_default_earth():
    '''
    Build earth object using ReferenceElliposoid and GTOD as body frame.

    Uses Constants based on WGS84 Standard from Orekit library.

    This method is called when PropagatorBuilder object is created.

    Args:
        methodName: name of method calling this function (for printing warning)

    Returns:
        ReferenceEllipsoid: Earth Body with rotating body frame

    '''
    return ReferenceEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                              Cst.WGS84_EARTH_FLATTENING,
                              FramesFactory.getGTOD(IERS.IERS_2010, False),
                              Cst.WGS84_EARTH_MU,
                              Cst.WGS84_EARTH_ANGULAR_VELOCITY)


class Builder(object):
    """
    Base class for Propagator builder.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _build_state(self):
        pass

    @abc.abstractmethod
    def _build_integrator(self):
        pass

    @abc.abstractmethod
    def _build_propagator(self):
        pass

    @abc.abstractmethod
    def get_propagator(self):
        pass


class PropagatorBuilder(Builder):
    """
    Class building Orekit propagator based on provided settings.

    Besides the settings also an initial satellite state and epoch
    have to be provided.

    The settings have to be provided as a python dictionary. Every method
    accesses in this nested dictionary its corresponding section which further
    has to have the following two keys:
        - type -> name of the subclass to use
        - settings -> dictionary with settings needed for the build

    If for no subclass is found for corresponding type, no propagator attributes
    will be changed.

    Here is an possible example for the _build_state method:
        State:
            type: 'Keplerian'
            settings:
                mass: 82.0
                epoch: 19990101T16:16:16
                attitude: 'grumpy'
                .
                .
                .

    Following methods use following keys to access its type and settings
    attributes:
        _build_state:                   State
        _build_gravity:                 Gravity
        _build_thrust:                  Thrust
        _build_drag_and_solar_pressure: SatShape, DragModel, SolarModel

    Every of the Classes which are being loop through have exactly
    2 methods:
        - isApplicable: checks if class name corresponds to type in dictionary
        - Setup: executes build if isApplicable returned true
    """

    def __init__(self, settings, state, epoch):
        self.propagator = Propagator()

        self.SatType = settings['name']

        print "Building %s:" % (self.SatType)
        self.orbSettings = settings['orbitProp']
        self.attSettings = settings['attitudeProvider']
        self.stateElements = state

        self.initialOrbit = None
        self.initialState = None
        self.gravField = None
        self.earth = _build_default_earth()

        self.thrustM = None

        # Frame in which orbit AND Body is defined
        # needs to be the same, because it is assumed in attitude prop
        self.inertialFrame = None
        self.refDate = epoch

    def _build_state(self):
        """
        Create initial Spacecraft State based on satellite's initial state
        and mass.
        """

        StatFactory = [cls() for cls in StateFactory.__subclasses__()]
        ST = self.orbSettings['State']
        for model in StatFactory:
            if model.isApplicable(ST['type']):
                [inFrame, inOrbit, inState] = model.Setup(self.refDate,
                                                          self.earth,
                                                          self.stateElements,
                                                          ST['settings'])

                self.inertialFrame = inFrame
                self.initialOrbit = inOrbit
                self.initialState = inState
                break

        if self.initialState is None:
            raise ValueError("No existing state class defined. " +
                             "State could not be build!")
        else:
            print "  [INFO]: Satellite state created in %s Frame."\
                  % (self.inertialFrame.getName())

    def _build_integrator(self):
        """
        Build orbit integrator based on settings specified in settings.

        Integrator used: DormandPrince853Integrator
        Orbit Type of propagation transformed to Cartesian.
        """

        intSettings = self.orbSettings['integrator']
        minStep = intSettings['minStep']
        maxStep = intSettings['maxStep']
        initStep = intSettings['initStep']
        positionTolerance = intSettings['positionTolerance']

        self.orbitType = OrbitType.CARTESIAN
        self.tol = NumericalPropagator.tolerances(positionTolerance,
                                                  self.initialOrbit,
                                                  self.orbitType)

        self.integrator = \
            DormandPrince853Integrator(float(minStep), float(maxStep),
                                       JArray_double.cast_(self.tol[0]),
                                       # Double array of doubles
                                       # needs to be casted in Python
                                       JArray_double.cast_(self.tol[1]))

        self.integrator.setInitialStepSize(float(initStep))

    def _build_propagator(self):
        """
        Build propagator object and add integrator and initial State to it.
        """

        self.propagator = NumericalPropagator(self.integrator)
        self.propagator.setOrbitType(self.orbitType)
        self.propagator.setInitialState(self.initialState)

        print "  [INFO]: Propagator created with %s integrator & %s orbit."\
              % (str(self.integrator.getName()), str(self.orbitType))

    def _build_attitude_propagation(self):
        """
        Set an attitude provider based on the type specified in settings.
        """

        AttFactory = [cls() for cls in AttitudeFactory.__subclasses__()]
        for provider in AttFactory:
            if provider.isApplicable(self.attSettings['type']):
                self.propagator = provider.Setup(self)
                break

    def _build_gravity(self):
        """
        Add gravity model to orbit propagation based on settings.

        Spacecraft will follow Newtonian orbit if type not matching any
        available Factory Class.
        """

        GMFactory = [cls() for cls in GravityFactory.__subclasses__()]
        GM = self.orbSettings['Gravity']
        for model in GMFactory:
            if model.isApplicable(GM['type']):
                [prop, gravField, file] = model.Setup(self.propagator,
                                                      GM['settings'],
                                                      self.earth)
                self.propagator = prop
                self.gravField = gravField

                print "  [INFO]: Gravity perturbation added. Using \'%s\' file."\
                      % (file[0])
                break

    def _build_thirdBody(self):
        """
        Adds Third body perturbation of Sun and/or Moon based on settings.
        """

        bodies = self.orbSettings['ThirdBody']
        for Body, addBody in bodies.items():
            if addBody:
                if str(Body) == 'Sun':
                    self.propagator.addForceModel(
                                        ThirdBodyAttraction(
                                           CelestialBodyFactory.getSun()))
                elif str(Body) == 'Moon':
                    self.propagator.addForceModel(
                                        ThirdBodyAttraction(
                                           CelestialBodyFactory.getMoon()))

                print "  [INFO]: added Attraction of %s." % (str(Body))

    def _build_solid_tides(self):
        """
        Adds Solid tides force model to propagation if specified in settings.

        Uses IERS2010 conventions.

        This method needs a gravity Field and Earth object. If one of those is
        not defined, they are default object is build by the build_default
        methods.
        """

        STides = self.orbSettings['SolidTides']
        if STides['add']:
            if self.gravField is None:
                gravField = _build_default_gravity_Field('_build_solid_tides')
            else:
                gravField = self.gravField

            body_list = []
            for Body, addBody in STides['settings'].items():
                if addBody:
                    if str(Body) == 'Sun':
                        body_list.append(CelestialBodyFactory.getSun())
                    if str(Body) == 'Moon':
                        body_list.append(CelestialBodyFactory.getMoon())

                    print "  [INFO]: Solid Tidal Force generated by %s added."\
                          % (str(Body))

            if body_list:  # empty list is false
                bl = JArray('object')(body_list, CelestialBody)
                conventions = IERS.IERS_2010
                # create Solid Tides force model including pole tides and
                # using default step and default number of point for integrator
                # tidal effects are ignored when interpolating EOP
                ST = SolidTides(self.earth.getBodyFrame(),
                                gravField.getAe(),
                                gravField.getMu(),
                                gravField.getTideSystem(),
                                conventions,
                                TimeScalesFactory.getUT1(conventions, True),
                                bl)
                self.propagator.addForceModel(ST)

    def _build_ocean_tides(self):
        """
        Adds ocean tide force model to propagation if specified in settings.

        Uses IERS2010 conventions.

        This method needs a gravity Field and Earth object. If one of those is
        not defined, they are default object is build by the build_default
        methods.
        """

        OTides = self.orbSettings['OceanTides']
        if OTides['add']:
            if self.gravField is None:
                gravField = _build_default_gravity_Field('_build_ocean_tides')
            else:
                gravField = self.gravField

            conventions = IERS.IERS_2010

            OT = OceanTides(self.earth.getBodyFrame(),
                            gravField.getAe(),
                            gravField.getMu(),
                            OTides['settings']['degree'],
                            OTides['settings']['order'],
                            conventions,
                            TimeScalesFactory.getUT1(conventions, True))

            self.propagator.addForceModel(OT)

            print "  [INFO]: Ocean Tides force model added."

    def _build_relativity(self):
        """
        Adds relativity force model to propagator if specified in settings.
        """

        if self.orbSettings['addRelativity']:
            if self.gravField is None:
                gravField = _build_default_gravity_Field('_build_relativity')
            else:
                gravField = self.gravField

            self.propagator.addForceModel(Relativity(gravField.getMu()))
            print "  [INFO]: Relativity Correction force added."

        else:
            pass

    def _build_thrust(self):
        """
        Adds a thrust model to propagation based on type defined in settings.

        Thrust Model can be obtained using get_thrust_model() method.
        """

        TFactory = [cls() for cls in ThrustFactory.__subclasses__()]
        T = self.orbSettings['Thrust']
        for model in TFactory:
            if model.isApplicable(T['type']):
                [propagator, TM] = model.Setup(self.propagator, T['settings'])
                self.propagator = propagator
                self.thrustM = TM

                print "  [INFO]: Thrust added."

    def _build_drag_and_solar_pressure(self):
        """
        Create Spacecraft shape and then build solar and drag force models.

        Drag and solar radiation pressure force models are added to propagation
        if to type defined in settings corresponding subclasses are found and
        shape is defined properly.
        If no satellite shape defined force models cannot be created.
        """

        SatModelSettings = self.orbSettings['SatShape']
        DragModelSettings = self.orbSettings['DragModel']
        SolarModelSettings = self.orbSettings['SolarModel']

        BoxFactory = [cls() for cls in SpacecraftModelFactory.__subclasses__()]
        DMFactory = [cls() for cls in DragFactory.__subclasses__()]
        SPFactory = [cls() for cls in SolarPressureFactory.__subclasses__()]

        for model in BoxFactory:
            if model.isApplicable(SatModelSettings['type']):
                self.spacecraft = model.Setup(SatModelSettings['settings'])
                break

        for model in DMFactory:
            if model.isApplicable(DragModelSettings['type']):
                [self.propagator, self.atmosphere] = model.Setup(self)
                print "  [INFO]: added Drag Force."
                break

        for model in SPFactory:
            if model.isApplicable(SolarModelSettings['type']):
                self.propagator = model.Setup(self)
                print "  [INFO]: added Radiation Pressure Force."
                break

    def get_propagator(self):
        """
        Call after build has been completed.

        Returns:
            Object: finished propagator.
        """

        # Print all force models which are being integrated
        print "[%s]: added Force models: \n%s"\
              % (self.SatType, str(self.propagator.getAllForceModels()))

        return self.propagator

    def get_earth(self):
        """
        Returns:
            Object: Earth object which was created during gravity build. If
                    none was created a default Earth object is created.
        """
        return self.earth

    def get_thrust_model(self):
        """
        Returns:
            Object: Thrust model if build and added to propagator
                    (otherwise returns None).
        """

        return self.thrustM


#####################################################################
class StateFactory(object):
    """
    Base Class for different builds of initial states
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """check for desired state build type"""

    @abc.abstractmethod
    def Setup(epoch, earth, state, setup):
        """Build spacecraft state based on type selected"""


def _build_satellite_attitude(setup, orbit_pv, inertialFrame, earth, epoch):
    '''Creates the initial attitude of the spacecraft based on the provided settings.

    If nadir pointing is defined the method takes the initial position and defines the correct
    rotation by using OREKIT's NadirPointing attiude provider.

    Args:
        setup: additional settings defined in dictionary
        orbit_pv: PVCoordinatesProvider from Orbit object
        inertialFrame: inertial Frame of propagation
        earth: Earth body object
        epoch: initial epoch as AbsoluteDate object

    Returns:
        Attitude: OREKIT's atttiude object with the correct initial attiude
    '''
    if setup['rotation'] == 'nadir':
            satRot = NadirPointing(inertialFrame, earth). \
                      getAttitude(orbit_pv, epoch, inertialFrame). \
                      getRotation()
    else:
        satRot = [float(x) for x in setup['rotation'].split(" ")]
        satRot = Rotation(satRot[0], satRot[1], satRot[2], satRot[3], False)

    spin = [math.radians(float(x)) for x in setup['spin'].split(" ")]
    spin = Vector3D(float(spin[0]), float(spin[1]), float(spin[2]))

    acc = [math.radians(float(x)) for x in setup['acceleration'].split(" ")]
    acc = Vector3D(float(acc[0]), float(acc[1]), float(acc[2]))

    return Attitude(epoch, inertialFrame, satRot, spin, acc)


class KeplerianEME2000(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "KeplerianEME2000":
            return True
        else:
            return False

    @staticmethod
    def Setup(epoch, earth, state, setup):
        """
        Create initial spacecraft state and orbit based on Keplerian elements.

        Args:
            epoch: initial epoch or orbital elements
            state: initial state of satellite
            setup: additional settings defined in dictionary

        Returns:
            inertialFrame: EME2000 as inertial Frame of Orbit
            initialOrbit: Keplerian orbit
            initialState: Spacecraft state
        """

        satMass = setup['mass']

        a = float(state.a)
        e = float(state.e)
        i = float(state.i)
        w = float(state.w)
        O = float(state.O)
        v = float(state.v)

        # Inertial frame where the satellite is defined (and earth)
        inertialFrame = FramesFactory.getEME2000()

        initialOrbit = KeplerianOrbit(a*1000, e, i, w, O, v,
                                      PositionAngle.TRUE,
                                      inertialFrame,
                                      epoch,
                                      Cst.WGS84_EARTH_MU)

        orbit_pv = PVCoordinatesProvider.cast_(initialOrbit)
        satAtt = _build_satellite_attitude(setup, orbit_pv, inertialFrame,
                                           earth, epoch)

        initialState = SpacecraftState(initialOrbit, satAtt, satMass)

        return [inertialFrame, initialOrbit, initialState]


class CartesianITRF(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "CartesianITRF":
            return True
        else:
            return False

    @staticmethod
    def Setup(epoch, earth, state, setup):
        """
        Create initial spacecraft state and orbit using PV-Coordinates in ITRF2008 Frame.

        Args:
            epoch: initial epoch or orbital elements
            state: initial state of satellite [Position, Velocity]
            setup: additional settings defined in dictionary

        Returns:
            inertialFrame: EME2000 as inertial Frame of Orbit
            initialOrbit: Cartesian orbit
            initialState: Spacecraft state
        """

        satMass = setup['mass']

        p = Vector3D(float(state.R[0]),
                     float(state.R[1]),
                     float(state.R[2]))
        v = Vector3D(float(state.V[0]),
                     float(state.V[1]),
                     float(state.V[2]))

        # Inertial frame where the satellite is defined (and earth)
        inertialFrame = FramesFactory.getEME2000()
        # False bool -> don't ignore tidal effects
        orbitFrame = FramesFactory.getITRF(IERS.IERS_2010, False)
        ITRF2EME = orbitFrame.getTransformTo(inertialFrame, epoch)
        pv_EME = ITRF2EME.transformPVCoordinates(PVCoordinates(p, v))

        initialOrbit = CartesianOrbit(pv_EME,
                                      inertialFrame,
                                      epoch,
                                      Cst.WGS84_EARTH_MU)

        orbit_pv = PVCoordinatesProvider.cast_(initialOrbit)
        satAtt = _build_satellite_attitude(setup, orbit_pv, inertialFrame,
                                           earth, epoch)

        initialState = SpacecraftState(initialOrbit, satAtt, satMass)

        return [inertialFrame, initialOrbit, initialState]


class CartesianEME2000(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "CartesianEME2000":
            return True
        else:
            return False

    @staticmethod
    def Setup(epoch, earth, state, setup):
        """
        Create initial spacecraft state and orbit using PV-Coordinates in J2000 Frame.

        Args:
            epoch: initial epoch or orbital elements
            state: initial state of satellite [Position, Velocity]
            setup: additional settings defined in dictionary

        Returns:
            inertialFrame: EME2000 as inertial Frame of Orbit
            initialOrbit: Cartesian orbit
            initialState: Spacecraft state
        """

        satMass = setup['mass']

        p = Vector3D(float(state.R[0]),
                     float(state.R[1]),
                     float(state.R[2]))
        v = Vector3D(float(state.V[0]),
                     float(state.V[1]),
                     float(state.V[2]))

        # Inertial frame where the satellite is defined (and earth)
        inertialFrame = FramesFactory.getEME2000()

        initialOrbit = CartesianOrbit(PVCoordinates(p, v),
                                      inertialFrame,
                                      epoch,
                                      Cst.WGS84_EARTH_MU)

        orbit_pv = PVCoordinatesProvider.cast_(initialOrbit)
        satAtt = _build_satellite_attitude(setup, orbit_pv, inertialFrame,
                                           earth, epoch)

        initialState = SpacecraftState(initialOrbit, satAtt, satMass)

        return [inertialFrame, initialOrbit, initialState]
#####################################################################


#####################################################################
class GravityFactory(object):
    """
    Base Class for different Gravity Force Models
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """check for desired Gravity Model"""

    @abc.abstractmethod
    def Setup(propagator, setup, earth):
        """Create gravity field and add force model to propagator."""


class EigenGravityWGS84(GravityFactory):

    @staticmethod
    def isApplicable(name):
        if name == "EigenGravityWGS84":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, setup, earth):
        """
        Add gravity perturbation using the HolmesFeatherstoneAttractionModel.

        As Earth model a ReferenceEllipsoid is used, with GTOD as body frame
        (tidal effects are not ignored when interpolating EOP).

        Uses WGS84 norm for the equatorial radius, flattening, standard
        grav. parameter and angular velocity.

        The Eigen-6s gravity model is used to compute the gravity field.

        Args:
            propagator: propagator object
            setup: additional settings defined in dictionary

        Returns:
            Propagator: propagator
            ReferenceEllipsoid: Earth body with GTOD as body frame
            NormalizedSphericalHarmonicsProvider: gravity field
            String: name of file used for the calculation of gravity potential
        """
        GravityFieldFactory.clearPotentialCoefficientsReaders()
        supported = GravityFieldFactory.ICGEM_FILENAME
        gfReader = ICGEMFormatReader(supported, False)
        GravityFieldFactory.addPotentialCoefficientsReader(gfReader)

        degree = setup['degree']
        order = setup['order']
        gravField = GravityFieldFactory.getNormalizedProvider(degree, order)
        gravModel = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(),
                                                      gravField)

        propagator.addForceModel(gravModel)

        file_name = FileDataHandler._get_name_of_loaded_files('Potential')
        if len(file_name) > 1:
            file_name = file_name[0]  # Orekit uses first loaded file
        elif len(file_name) == 0:
            # error should be thrown before this when creating gravModel!
            raise ValueError('No gravity potential file loaded!')

        return [propagator, gravField, file_name]


class EGM96GravityWGS84(GravityFactory):

    @staticmethod
    def isApplicable(name):
        if name == "EGM96GravityWGS84":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, setup, earth):
        """
        Add gravity perturbation using the HolmesFeatherstoneAttractionModel.

        As Earth model a ReferenceEllipsoid is used, with GTOD as body frame
        (tidal effects are not ignored when interpolating EOP).

        Uses WGS84 norm for the equatorial radius, flattening, standard
        grav. parameter and angular velocity.

        The EGM96 gravity model is used to compute the gravity field.

        Args:
            propagator: propagator object
            setup: additional settings defined in dictionary

        Returns:
            Propagator: propagator
            ReferenceEllipsoid: Earth body with GTOD body frame
            NormalizedSphericalHarmonicsProvider: gravity field
            String: name of file used for the calculation of gravity potential
        """

        GravityFieldFactory.clearPotentialCoefficientsReaders()
        supported = GravityFieldFactory.EGM_FILENAME
        gfReader = EGMFormatReader(supported, False)
        GravityFieldFactory.addPotentialCoefficientsReader(gfReader)

        degree = setup['degree']
        order = setup['order']

        gravField = GravityFieldFactory.getNormalizedProvider(degree, order)
        gravModel = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(),
                                                      gravField)

        propagator.addForceModel(gravModel)

        file_name = FileDataHandler._get_name_of_loaded_files('Potential')
        if len(file_name) > 1:
            file_name = file_name[0]  # orekit uses first loaded file
        elif len(file_name) == 0:
            # error should be trhown before this when creating gravModel!
            raise ValueError('No gravity potential file loaded!')

        return [propagator, gravField, file_name]
#####################################################################


#####################################################################
class AttitudeFactory(object):
    """
    Base class which implements different attitude providers.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """Check for desired Attitude Provider"""

    @abc.abstractmethod
    def Setup(builderInstance):
        """Create setup for attitude provider and add it to propagator."""


class AttNadir(AttitudeFactory):

    @staticmethod
    def isApplicable(name):
        if name == "AttNadir":
            return True
        else:
            return False

    @staticmethod
    def Setup(builderInstance):
        """
        Adding Orekit's Attitude Provider aligning the z-axis of the spacecraft with nadir.

        Args:
            builderInstance: Instance of propagator builder

        Returns:
            Propagator: propagator
        """
        propagator = builderInstance.propagator
        earth = builderInstance.earth

        attitude = NadirPointing(builderInstance.inertialFrame,
                                 earth)

        propagator.setAttitudeProvider(attitude)

        return propagator


class AttPropagation(AttitudeFactory):

    @staticmethod
    def isApplicable(name):
        if name == "AttPropagation":
            return True
        else:
            return False

    @staticmethod
    def Setup(builderInstance):
        """
        Implements Attitude propagation and sets it as attitude provider.

        The attitude propagator accounts for disturbance torques defined in settings cfg file.

        Args:
            builderInstance: Instance of propagator builder

        Returns:
            Propagator: propagator
        """
        propagator = builderInstance.propagator
        setup = builderInstance.attSettings['settings']
        earth = builderInstance.earth

        iT_dict = setup['inertiaTensor']
        int_dict = setup['integrator']
        discretization = setup['Discretization']

        gravitySettings = setup['GravityGradient']
        solarSettings = setup['SolarPressure']
        dragSettings = setup['AeroDrag']
        magSettings = setup['MagneticTorque']
        innerCuboids = None
        surfaceMesh = None

        AttitudeFM = dict()
        AttitudeFM['Earth'] = earth

        to_add_list = [gravitySettings['add'], magSettings['add'],
                       solarSettings['add'], dragSettings['add']]

        # add Spacecraft State observer as force model to be able
        # to extract spacecraft state during integration
        StateOb = StateObserver(propagator.getInitialState())
        propagator.addForceModel(StateOb)
        AttitudeFM['StateObserver'] = StateOb

        # Inertia Tensor of Spacecraft:
        Ix = [float(x) for x in iT_dict['Ix'].split(" ")]
        Iy = [float(x) for x in iT_dict['Iy'].split(" ")]
        Iz = [float(x) for x in iT_dict['Iz'].split(" ")]
        inertiaT = np.array([Ix, Iy, Iz])

        # find discretization class if needed
        if 'type' in discretization:
            type_disc = None
            types_clases = [cls() for cls in DiscInterface.__subclasses__()]
            for disc in types_clases:
                # look for subclasses with same name and store correct class
                if discretization['type'] == disc.__class__.__name__:
                    type_disc = 'type'
                    break
            if type_disc is None and \
               (gravitySettings['add'] or solarSettings['add'] or dragSettings['add']):
                # discretization type not defined but discretization needed..
                raise ValueError("No discretization type defined in settings file!")

        # add Gravity Gradient Torque to Attitude Propagation:
        if gravitySettings['add']:
            innerCuboids = disc.discretize_inner_body(discretization['settings'])

            # use own Gravity Model with own Field Coefficients
            degree = gravitySettings['FC_degree']
            order = gravitySettings['FC_order']
            gravField = GravityFieldFactory.getNormalizedProvider(degree,
                                                                  order)
            AttitudeFM['GravityModel'] = \
                HolmesFeatherstoneAttractionModel(
                                        earth.getBodyFrame(),
                                        gravField)

        if solarSettings['add'] or dragSettings['add']:
            surfaceMesh = disc.discretize_outer_surface(solarSettings,
                                                        discretization['settings'])
            sun = CelestialBodyFactory.getSun()

            if solarSettings['add']:
                AttitudeFM['Sun'] = sun
                # dummy spacecraft which enables access to methods
                #  needed to create SolarRadiationPressure object
                dummy = IsotropicRadiationClassicalConvention(
                                                1.0,
                                                solarSettings['AbsorbCoeff'],
                                                solarSettings['ReflectCoeff'])

                # Force Model needed to get Lighting ratio:
                eqRad = earth.getEquatorialRadius()
                AttitudeFM['SolarModel'] = SolarRadiationPressure(sun,
                                                                  eqRad,
                                                                  dummy)

                # add two Event handlers for solar eclipse (one when entering eclipse other when leaving)
                # solar pressure torque only computed if not in umbra
                dayNightEvent = EclipseDetector(PVCoordinatesProvider.cast_(sun),
                                                696000000.,  # radius of the sun
                                                CelestialBodyFactory.getEarth(),
                                                Cst.WGS84_EARTH_EQUATORIAL_RADIUS)
                dayNightEvent = dayNightEvent.withHandler(NightEclipseDetector().of_(EclipseDetector))

            if dragSettings['add']:
                # check if atmosphere was already created
                # for orbit propagation, otherwise create one:
                atmo = getattr(builderInstance, 'atmosphere', None)
                if atmo is None:
                    msafe = \
                        MarshallSolarActivityFutureEstimation(
                         "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)" +
                         "\\p{Digit}\\p{Digit}\\p{Digit}\\p{Digit}F10\\" +
                         ".(?:txt|TXT)",
                         MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)

                    manager = DataProvidersManager.getInstance()
                    manager.feed(msafe.getSupportedNames(), msafe)

                    atmo = DTM2000(msafe, sun, earth)

                else:
                    atmo = builderInstance.atmosphere

                AttitudeFM['AtmoModel'] = atmo
                surfaceMesh['Cd'] = dragSettings['DragCoeff']

        if magSettings['add']:
            AttitudeFM['MagneticModel'] = magSettings['settings']

        provider = AttitudePropagation(builderInstance.initialState.getAttitude(),
                                       builderInstance.refDate,
                                       inertiaT,
                                       builderInstance.tol,
                                       int_dict,
                                       innerCuboids,
                                       surfaceMesh,
                                       AttitudeFM)

        # add torques which are set to true in yaml file
        provider.setAddedDisturbanceTorques(to_add_list[0], to_add_list[1],
                                            to_add_list[2], to_add_list[3])

        if solarSettings['add']:
            # add night eclipse detector to turn of solar pressure torque when in umbra
            NightEclipseDetector.attitudeProvider = provider
            propagator.addEventDetector(dayNightEvent)
            # disable torque if starting at umbra:
            if dayNightEvent.g(propagator.getInitialState()) < 0:
                DT = NightEclipseDetector.attitudeProvider.getAddedDisturbanceTorques()
                NightEclipseDetector.attitudeProvider.setAddedDisturbanceTorques(DT[0], DT[1], False, DT[3])

        propagator.setAttitudeProvider(provider)

        return propagator


# Helper classes for attitude propagation:
class NightEclipseDetector(PythonEventHandler):
    attitudeProvider = None

    def eventOccurred(self, s, detector, increasing):
        if not increasing:
            # in umbra
            DT = NightEclipseDetector.attitudeProvider.getAddedDisturbanceTorques()
            NightEclipseDetector.attitudeProvider.setAddedDisturbanceTorques(DT[0], DT[1], False, DT[3])

        if increasing:
            # exiting umbra
            DT = NightEclipseDetector.attitudeProvider.getAddedDisturbanceTorques()
            NightEclipseDetector.attitudeProvider.setAddedDisturbanceTorques(DT[0], DT[1], True, DT[3])

        return EventHandler.Action.CONTINUE

    def resetState(self, detector, oldState):
        return oldState
#####################################################################


#####################################################################
class SpacecraftModelFactory(object):
    """
    Base Class for different types of BoxAndSolarArraySpacecraft objects.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """Check for desired spacecraft model factory"""

    @abc.abstractmethod
    def Setup(SatModelSettings):
        """Set up BoxAndSolarArraySpacecraft object"""


def load_facets_from_dict(Facets):
    """
    Give me Facets I give you JArray object.

    Args:
        Facets: dictionary with facets are and normal vector

    Example of Facets:
            front:
              area: 0.313956 # 612x513
              nVector: 1 0 0
            back:
              area: 0.313956
              nVector: -1 0 0
            .
            .
            .

    Returns:
        JArray('object'): Filled with facet list for Orekit to create
                          BoxAndSolarArraySpacecraft
    """

    facet_list = []
    for val in Facets.itervalues():
        area = float(val['area'])
        vector = [float(x) for x in val['nVector'].split(" ")]
        nVector = Vector3D(float(vector[0]),
                           float(vector[1]),
                           float(vector[2]))
        new_Facet = BoxAndSolarArraySpacecraft.Facet(nVector, area)
        facet_list.append(new_Facet)

    return JArray('object')(facet_list, BoxAndSolarArraySpacecraft.Facet)


class FacetsAndFixedSolarArray(SpacecraftModelFactory):

    @staticmethod
    def isApplicable(name):
        if name == "FacetsAndFixedSolarArray":
            return True
        else:
            return False

    @staticmethod
    def Setup(SatModelSettings):
        """
        Creates a spacecraft model based on the facets defined in settings.

        Solar Arrays rotate around fixed axis defined in settings with best
        lightning.

        Any number of facets can be added to settings. However shape of
        spacecraft has to be convex.

        Args:
            SatModelSettings: dictionary needed for build

        Example of SatModelSettings:
          Facets: area in [m2]
            front:
              area: 0.313956 # 612x513
              nVector: 1 0 0
            back:
              area: 0.313956
              nVector: -1 0 0
            left:
              area: 0.485316 # 612x793
              nVector: 0 0 1
            right:
              area: 0.485316
              nVector: 0 0 -1
            top:
              area: 0.406809 # 513x793
              nVector: 0 -1 0
            bottom:
              area: 0.406809
              nVector: 0 1 0
          SolarArea: 0.60096 # 313x960 * 2
          SolarDir: 0 0 1
          absCoeff: 0.98
          refCoeff: 0.02
          dragCoeff: 1.3
          liftRatio: 0.0

        Returns:
            BoxAndSolarArraySpacecraft: satellite model
        """

        Facets = SatModelSettings['Facets']

        facet_list = load_facets_from_dict(Facets)

        sun = CelestialBodyFactory.getSun()

        solarDir = [float(x) for x in SatModelSettings['SolarDir'].split(" ")]
        solarDir = Vector3D(float(solarDir[0]),
                            float(solarDir[1]),
                            float(solarDir[2]))
        starfighter = BoxAndSolarArraySpacecraft(facet_list,
                                                 sun,
                                                 SatModelSettings['SolarArea'],
                                                 solarDir,
                                                 SatModelSettings['dragCoeff'],
                                                 SatModelSettings['liftRatio'],
                                                 SatModelSettings['absCoeff'],
                                                 SatModelSettings['refCoeff'])

        return starfighter
#####################################################################


#####################################################################
class DragFactory(object):
    """
    Base class for different drag and atmosphere models
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """Check for desired drag factory"""

    @abc.abstractmethod
    def Setup(builderInstance):
        """Create drag instance and add it to propagator"""


class DragDTM2000MSAFE(DragFactory):

    @staticmethod
    def isApplicable(name):
        if name == "DragDTM2000MSAFE":
            return True
        else:
            return False

    @staticmethod
    def Setup(builderInstance):
        """
        Adds a Drag Force to orbit propagation using DTM2000 model as atmosphere.

        Uses a BoxAndSolarArraySpacecraft object and aerodynamic parameters
        configured in the settings.

        Returns:
            propagator: Propagator
            atmosphere: DTM2000 model of atmosphere
        """

        propagator = builderInstance.propagator
        earth = builderInstance.earth
        starfighter = builderInstance.spacecraft
        dragModel = builderInstance.orbSettings['DragModel']
        sun = CelestialBodyFactory.getSun()

        # if drag coefficient estimated so select its driver
        # (coeff will not be fixed):
        if dragModel['settings']['cD_Estimated']:
            for coef in starfighter.getDragParametersDrivers():
                if coef.getName() == "drag coefficient":
                    coef.setSelected(True)

        # load data from orekit-data zip file and
        # atmosphere from data of the Marshall Space Flight Center
        # (needed for the DTM-2000 atmosphere model)
        # txt have the format Jan2017F10.txt
        msafe = \
            MarshallSolarActivityFutureEstimation(
             "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)" +
             "\\p{Digit}\\p{Digit}\\p{Digit}\\p{Digit}F10\\" +
             ".(?:txt|TXT)",
             MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE)

        manager = DataProvidersManager.getInstance()
        manager.feed(msafe.getSupportedNames(), msafe)

        atmosphere = DTM2000(msafe, sun, earth)
        propagator.addForceModel(DragForce(atmosphere, starfighter))

        return [propagator, atmosphere]


class DragDTM2000CELESTRACK(DragFactory):

    @staticmethod
    def isApplicable(name):
        if name == "DragDTM2000CELESTRACK":
            return True
        else:
            return False

    @staticmethod
    def Setup(builderInstance):
        """
        Adds a Drag Force to orbit propagation using DTM2000 model as atmosphere.

        Uses a BoxAndSolarArraySpacecraft object and aerodynamic parameters
        configured in the settings.

        Returns:
            propagator: Propagator
            atmosphere: DTM2000 model of atmosphere
        """
        propagator = builderInstance.propagator
        earth = builderInstance.earth
        starfighter = builderInstance.spacecraft
        dragModel = builderInstance.orbSettings['DragModel']
        sun = CelestialBodyFactory.getSun()

        # if drag coefficient estimated so select its driver
        # (coeff will not be fixed):
        if dragModel['settings']['cD_Estimated']:
            for coef in starfighter.getDragParametersDrivers():
                if coef.getName() == "drag coefficient":
                    coef.setSelected(True)

        # load data from celestrack-weather
        ctw = CelesTrackWeather("(:?sw|SW)\\p{Digit}+\\.(?:txt|TXT)")

        manager = DataProvidersManager.getInstance()
        manager.feed(ctw.getSupportedNames(), ctw)

        atmosphere = DTM2000(ctw, sun, earth)
        propagator.addForceModel(DragForce(atmosphere, starfighter))

        return [propagator, atmosphere]
#####################################################################


class SolarPressureFactory(object):
    """
    Base class for different solar pressure models used in orbit propagation.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """Check for desired solar pressure factory"""

    @abc.abstractmethod
    def Setup(builderInstance):
        """Create Solar pressure force model and add it to propagator"""


class SolarPressureBoxModel(SolarPressureFactory):

    @staticmethod
    def isApplicable(name):
        if name == "SolarPressureBoxModel":
            return True
        else:
            return False

    @staticmethod
    def Setup(builderInstance):
        """
        Adds solar radiation pressure force model to orbit propagation.

        Needs a that a satellite shape has been created beforehand in builder
        instance.

        Args:
            builderInstance: Instance of propagator builder

        Returns:
            propagator: Propagator
        """

        propagator = builderInstance.propagator
        earth = builderInstance.earth

        starfighter = builderInstance.spacecraft
        solarSett = builderInstance.orbSettings['SolarModel']
        sun = CelestialBodyFactory.getSun()

        # drag coefficient estimated so select its driver
        # (coeff will not be fixed):
        for coef in starfighter.getRadiationParametersDrivers():
            if coef.getName() == "reflection coefficient" \
               and \
               solarSett['settings']['abs_Estimated']:
                coef.setSelected(True)
            if coef.getName() == "absorption coefficient" \
               and \
               solarSett['settings']['ref_Estimated']:
                coef.setSelected(True)

        propagator.addForceModel(
              SolarRadiationPressure(sun,
                                     earth.getEquatorialRadius(),
                                     starfighter))

        return propagator
#####################################################################


#####################################################################
class ThrustFactory(object):
    """
    Base class for different solar pressure models used in orbit propagation
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def isApplicable(name):
        """Check for desired thrust model"""

    @abc.abstractmethod
    def Setup(propagator, thrustSettings):
        """Create thrust model and add it to propagator"""


class ThrustModelVariable(ThrustFactory):

    @staticmethod
    def isApplicable(name):
        if name == "ThrustModelVariable":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, thrustSettings):
        """
        Creates thrust model based on orekit's ConstantThrustManeuver class.

        Direction of thrust and magnitude is changeable after creation
        of the object, meaning that Force Models in propagator don't have to be
        reset when changing thrusting maneuver (in contrast to orekit's class)

        Returns:
            propagator: Propagator
            thrustM: Thrust model
        """

        thrustM = ThrustModel()
        propagator.addForceModel(thrustM)

        return [propagator, thrustM]
#####################################################################


class Propagator(object):
    """Propagator object to be build"""
