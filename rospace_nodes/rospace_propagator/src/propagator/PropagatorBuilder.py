# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import abc
import itertools
import numpy as np

from ThrustModel import ThrustModel
from AttitudePropagation import AttitudePropagation
from StateObserver import StateObserver

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
from org.orekit.attitudes import NadirPointing
from org.orekit.data import DataProvidersManager
from org.orekit.models.earth import GeoMagneticModelLoader

from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D


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


def _build_default_earth(methodName):
    '''
    Build earth object using OneAxisElliposoid and GTOD as body frame.

    Uses Constants based on WGS84 Standard from Orekit library.

    Args:
        methodName: name of method calling this function (for printing warning)

    Returns:
        ReferenceEllipsoid: Earth Body with rotating body frame

    '''
    mesg = "\033[93m  [WARN] [Builder." + methodName \
           + "]: No earth defined. Creating default Earth using" \
           + " ReferenceElliposoid with GTOD and IERS2010 conventions as"\
           + " body frame and constants based on the WGS84 Standard.\033[0m"
    print mesg

    return ReferenceEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                              Cst.WGS84_EARTH_FLATTENING,
                              FramesFactory.getGTOD(IERS.IERS_2010, False),
                              Cst.WGS84_EARTH_MU,
                              Cst.WGS84_EARTH_ANGULAR_VELOCITY)


def _get_name_of_loaded_files(folder_name):
    '''
    Gets names of files in defined folder loaded by the data provider.

    Args:
        folder_name: string of folder name containing files

    Returns:
        List<String>: all file names loaded by data provider in folder
    '''
    file_names = []
    manager = DataProvidersManager.getInstance()
    string_set = manager.getLoadedDataNames()
    for i in string_set:
        if folder_name in i:
            file_names.append(i.rsplit('/', 1)[1])

    return file_names


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
    accesses in this nested dictonary its corresponding section which futher
    has to have the following two keys:
        - type -> name of the subclass to use
        - settings -> dictionary with settings needed for the build

    If for no subclass is found for corresponding type, no propagtor attributes
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
        self.earth = None

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

        Spacecraft will follow newtonian orbit if type not matching any
        available Factory Class.
        """

        GMFactory = [cls() for cls in GravityFactory.__subclasses__()]
        GM = self.orbSettings['Gravity']
        for model in GMFactory:
            if model.isApplicable(GM['type']):
                [prop, earth, gravField, file] = model.Setup(self.propagator,
                                                             GM['settings'])
                self.propagator = prop
                self.earth = earth
                self.gravField = gravField

                print "  [INFO]: Gravity pertubation added. Using \'%s\' file."\
                      % (file[0])
                break

    def _build_thirdBody(self):
        """
        Adds Third body pertubation of Sun and/or Moon based on settings.
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

            if self.earth is None:
                earth = _build_default_earth('_build_solid_tides')
            else:
                earth = self.earth

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
                ST = SolidTides(earth.getBodyFrame(),
                                gravField.getAe(),
                                gravField.getMu(),
                                gravField.getTideSystem(),
                                conventions,
                                TimeScalesFactory.getUT1(conventions, True),
                                bl)
                self.propagator.addForceModel(ST)

    def _build_ocean_tides(self):
        """
        Adds ocean tide force model to propagation if spceified in settings.

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

            if self.earth is None:
                earth = _build_default_earth('_build_ocean_tides')
            else:
                earth = self.earth

            conventions = IERS.IERS_2010

            OT = OceanTides(earth.getBodyFrame(),
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
        if self.earth is None:
            return _build_default_earth('get_earth')
        else:
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
    def Setup(epoch, state, setup):
        """Build spacecraft state based on type selected"""


class KeplerianEME2000(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "KeplerianEME2000":
            return True
        else:
            return False

    @staticmethod
    def Setup(epoch, state, setup):
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

        SatMass = setup['mass']

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

        initialState = SpacecraftState(initialOrbit, SatMass)

        return [inertialFrame, initialOrbit, initialState]


class CartesianITRF(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "CartesianITRF":
            return True
        else:
            return False

    @staticmethod
    def Setup(epoch, state, setup):
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

        SatMass = setup['mass']

        p = Vector3D(float(state.R[0]),
                     float(state.R[1]),
                     float(state.R[2]))
        v = Vector3D(float(state.V[0]),
                     float(state.V[1]),
                     float(state.V[2]))

        # Inertial frame where the satellite is defined (and earth)
        inertialFrame = FramesFactory.getEME2000()
        # don't ignore tidal effects
        orbitFrame = FramesFactory.getITRF(IERS.IERS_2010, False)
        ITRF2EME = orbitFrame.getTransformTo(inertialFrame, epoch)
        pv_EME = ITRF2EME.transformPVCoordinates(PVCoordinates(p, v))

        initialOrbit = CartesianOrbit(pv_EME,
                                      inertialFrame,
                                      epoch,
                                      Cst.WGS84_EARTH_MU)

        initialState = SpacecraftState(initialOrbit, SatMass)

        return [inertialFrame, initialOrbit, initialState]


class CartesianEME2000(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "CartesianEME2000":
            return True
        else:
            return False

    @staticmethod
    def Setup(epoch, state, setup):
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

        SatMass = setup['mass']

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

        initialState = SpacecraftState(initialOrbit, SatMass)

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
    def Setup(propagator, setup):
        """Create gravity field and add force model to propagator."""


class EigenGravityWGS84(GravityFactory):

    @staticmethod
    def isApplicable(name):
        if name == "EigenGravityWGS84":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, setup):
        """
        Add gravity pertubation using the HolmesFeatherstoneAttractionModel.

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

        earth = ReferenceEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                                   Cst.WGS84_EARTH_FLATTENING,
                                   FramesFactory.getGTOD(IERS.IERS_2010, False),
                                   Cst.WGS84_EARTH_MU,
                                   Cst.WGS84_EARTH_ANGULAR_VELOCITY)

        degree = setup['degree']
        order = setup['order']
        gravField = GravityFieldFactory.getNormalizedProvider(degree, order)
        gravModel = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(),
                                                      gravField)

        propagator.addForceModel(gravModel)

        file_name = _get_name_of_loaded_files('Potential')
        if len(file_name) > 1:
            file_name = file_name[0]  # orekit uses first loaded file
        elif len(file_name) == 0:
            # error should be trhown before this when creating gravModel!
            raise ValueError('No gravity potential file loaded!')

        return [propagator, earth, gravField, file_name]


class EGM96GravityWGS84(GravityFactory):

    @staticmethod
    def isApplicable(name):
        if name == "EGM96GravityWGS84":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, setup):
        """
        Add gravity pertubation using the HolmesFeatherstoneAttractionModel.

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

        earth = ReferenceEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                                   Cst.WGS84_EARTH_FLATTENING,
                                   FramesFactory.getGTOD(IERS.IERS_2010, False),
                                   Cst.WGS84_EARTH_MU,
                                   Cst.WGS84_EARTH_ANGULAR_VELOCITY)

        degree = setup['degree']
        order = setup['order']

        gravField = GravityFieldFactory.getNormalizedProvider(degree, order)
        gravModel = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(),
                                                      gravField)

        propagator.addForceModel(gravModel)

        file_name = _get_name_of_loaded_files('Potential')
        if len(file_name) > 1:
            file_name = file_name[0]  # orekit uses first loaded file
        elif len(file_name) == 0:
            # error should be trhown before this when creating gravModel!
            raise ValueError('No gravity potential file loaded!')

        return [propagator, earth, gravField, file_name]
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


def discretize_inner_body(discSettings):
    """
    Discretize a shoebox-type satellite in cuboids of equal mass.

    Depends on defined number of cuboids in x,y and z
    direction and total size of satellite.

    Args:
        discSettings: dictionary with satellite_dim, and number of cuboids in
                      all 3 directions

    Example of discSettings layout:
        discSettings:
            satellite_dim:
              l_x: 0.793
              l_y: 0.612
              l_z: 0.513
            inner_cuboids:
              numCub_x: 2
              numCub_y: 2
              numCub_z: 2

    Returns:
        inCub: dictionary with center of masss of each cuboid in satellite
               frame and its corresponding mass
    """

    s_l_x = float(discSettings['satellite_dim']['l_x'])
    s_l_y = float(discSettings['satellite_dim']['l_y'])
    s_l_z = float(discSettings['satellite_dim']['l_z'])

    # seperate cuboid into number of smaller cuboids and store
    # coordinates of center of mass in satellite frame
    numC_x = discSettings['inner_cuboids']['numCub_x']
    numC_y = discSettings['inner_cuboids']['numCub_y']
    numC_z = discSettings['inner_cuboids']['numCub_z']

    # dimension of inner cuboids:
    c_l_x = s_l_x / numC_x
    c_l_y = s_l_y / numC_y
    c_l_z = s_l_z / numC_z

    # total number of cuboids
    numC_tot = numC_x * numC_y * numC_z

    # populate satellite with cuboids:
    inCub = dict()
    CoM = []
    MassCub = []
    massFrac = 1.0 / numC_tot

    ##################################
    # delete this once change in attitude prop
    inCub['mass_frac'] = 1.0 / numC_tot  # mass equally distributed
    ####################################

    # compute center of mass for each cuboid starting at
    # top, back, right corner
    for ix in xrange(numC_x):
        CoM_x = 0.5 * c_l_x - 0.5 * s_l_x + ix * c_l_x
        for iy in xrange(numC_y):
            CoM_y = 0.5 * c_l_y - 0.5 * s_l_y + iy * c_l_y
            for iz in xrange(numC_z):
                CoM_z = 0.5 * c_l_z - 0.5 * s_l_z + iz * c_l_z

                CoM.append(Vector3D(float(CoM_x),
                                    float(CoM_y),
                                    float(CoM_z)))
                MassCub.append(massFrac)

    inCub['CoM'] = CoM

    return inCub


def discretize_outer_surface(solarSettings, discSettings):
    """
    Discretize outer surface of a shoebox satellite into planes of equal area.

    Depends on defined number of surfaces in x,y and z direction and total size
    of satellite.

    Args:
        solarSettings: dictionary with settings for solar arrays
        discSettings: dictionary for discretization of outer surface

    Example of solarSettings:
        solarSettings:
              l_x: 1
              l_z: 1
              numSRSolar_x: 2
              numSRSolar_z: 2
              PosAndDir:
                SA1:
                  dCenter: 0 0 0.5
                  normalV: 0 0 1
                SA2:
                  dCenter: 0 0 -0.5
                  normalV: 0 0 1

    Example of discSettings:
        discSettings:
            satellite_dim:
              l_x: 0.793
              l_y: 0.612
              l_z: 0.513
            inner_cuboids:
              numCub_x: 2
              numCub_y: 2
              numCub_z: 2

    Returns:
     mesh_dA: dictionary with center of mass and normal vector of each surface
              in satellite frame and its corresponding area and reflection
              coefficient
    """

    s_l_x = float(discSettings['satellite_dim']['l_x'])
    s_l_y = float(discSettings['satellite_dim']['l_y'])
    s_l_z = float(discSettings['satellite_dim']['l_z'])

    sat_Ca = solarSettings['AbsorbCoeff']
    sat_Cs = solarSettings['ReflectCoeff']
    sol_Ca = solarSettings['SolarArray_AbsorbCoeff']
    sol_Cs = solarSettings['SolarArray_ReflectCoeff']

    numSR_x = discSettings['surface_rectangles']['numSR_x']
    numSR_y = discSettings['surface_rectangles']['numSR_y']
    numSR_z = discSettings['surface_rectangles']['numSR_z']

    c_l_x = s_l_x / numSR_x
    c_l_y = s_l_y / numSR_y
    c_l_z = s_l_z / numSR_z

    area_x = c_l_y * c_l_z  # front and back area
    area_y = c_l_x * c_l_z  # top and bottom area
    area_z = c_l_x * c_l_y  # left and right area

    # Center of all Planes
    front_CoM_x = float(s_l_x * 0.5)
    back_CoM_x = float(-s_l_x * 0.5)
    bottom_CoM_y = float(s_l_y * 0.5)
    top_CoM_y = float(-s_l_y * 0.5)
    left_CoM_z = float(s_l_z * 0.5)
    right_CoM_z = float(-s_l_z * 0.5)

    # define normal vectors:
    front_Normal = Vector3D.PLUS_I
    back_Normal = Vector3D.MINUS_I
    bottom_Normal = Vector3D.PLUS_J
    top_Normal = Vector3D.MINUS_J
    left_Normal = Vector3D.PLUS_K
    right_Normal = Vector3D.MINUS_K

    CoM = []
    Normal = []
    Area = []
    Coefs = []
    mesh_dA = dict()

    # front/back from left to right, top to bottom:
    for iy in xrange(numSR_y):
        CoM_y = 0.5 * c_l_y - 0.5 * s_l_y + iy * c_l_y
        for iz in xrange(numSR_z):
            CoM_z = 0.5 * c_l_z - 0.5 * s_l_z + iz * c_l_z

            CoM.append(Vector3D(front_CoM_x,
                                CoM_y,
                                CoM_z))
            CoM.append(Vector3D(back_CoM_x,
                                CoM_y,
                                CoM_z))

            Normal.append(front_Normal)
            Normal.append(back_Normal)
            Area.extend([area_x]*2)
            Coefs.extend([np.array([sat_Ca, sat_Cs])]*2)

    # top/bottom from left to right, back to front:
    for ix in xrange(numSR_x):
        CoM_x = 0.5 * c_l_x - 0.5 * s_l_x + ix * c_l_x
        for iz in xrange(numSR_z):
            CoM_z = 0.5 * c_l_z - 0.5 * s_l_z + iz * c_l_z

            CoM.append(Vector3D(CoM_x,
                                bottom_CoM_y,
                                CoM_z))
            CoM.append(Vector3D(CoM_x,
                                top_CoM_y,
                                CoM_z))
            Normal.append(bottom_Normal)
            Normal.append(top_Normal)
            Area.extend([area_y]*2)
            Coefs.extend([np.array([sat_Ca, sat_Cs])]*2)

    # left/right from top to bottom, back to front:
    for ix in xrange(numSR_x):
        CoM_x = 0.5 * c_l_x - 0.5 * s_l_x + ix * c_l_x
        for iy in xrange(numSR_y):
            CoM_y = 0.5 * c_l_y - 0.5 * s_l_y + iy * c_l_y

            CoM.append(Vector3D(CoM_x,
                                CoM_y,
                                left_CoM_z))
            CoM.append(Vector3D(CoM_x,
                                top_CoM_y,
                                right_CoM_z))

            Normal.append(left_Normal)
            Normal.append(right_Normal)
            Area.extend([area_z]*2)
            Coefs.extend([np.array([sat_Ca, sat_Cs])]*2)

    # discretization of 2D solar arrays
    if 'SolarArrays' in discSettings:
        solarSettings = discSettings['SolarArrays']

        dCList = []
        normalList = []

        for SolarArray in solarSettings['PosAndDir'].values():
            # deviation of Solar Array from Satellite CoM:
            dC = [float(x) for x in SolarArray['dCenter'].split(" ")]
            normal = [float(x) for x in SolarArray['normalV'].split(" ")]
            dCList.append(np.array(dC))
            normalList.append(np.array(normal))

        assert len(dCList) == len(normalList)

        sol_l_x = float(solarSettings['l_x'])
        sol_l_z = float(solarSettings['l_z'])

        numSRSolar_x = solarSettings['numSRSolar_x']
        numSRSolar_z = solarSettings['numSRSolar_z']

        c_l_x = sol_l_x / numSRSolar_x
        c_l_z = sol_l_z / numSRSolar_z

        solArea = c_l_x * c_l_z

        for ix in xrange(numSRSolar_x):
            CoM_x = 0.5 * c_l_x - 0.5 * sol_l_x + ix * c_l_x
            for iz in xrange(numSRSolar_z):
                CoM_z = 0.5 * c_l_z - 0.5 * sol_l_z + iz * c_l_z

                for dC, normal in itertools.izip(dCList, normalList):
                    CoM.append(Vector3D(float(dC[0] + CoM_x),
                                        float(dC[1]),
                                        float(dC[2] + CoM_z)))
                    Normal.append(Vector3D(float(normal[0]),
                                           float(normal[1]),
                                           float(normal[2])))
                    Area.append(solArea)
                    Coefs.append(np.array([sol_Ca, sol_Cs]))

    # fill dictionary with lists:
    mesh_dA['CoM'] = CoM
    mesh_dA['Normal'] = Normal
    mesh_dA['Area'] = Area
    mesh_dA['Coefs'] = Coefs

    return mesh_dA


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
        Adding Attitude Provider aligning z-axis of the Spacecraft and nadir.

        Args:
            builderInstance: Instance of propagator builder

        Returns:
            Propagator: propagator
        """
        propagator = builderInstance.propagator

        if builderInstance.earth is not None:
            earth = builderInstance.earth
        else:
            earth = _build_default_earth('AttNadir')

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

        Args:
            builderInstance: Instance of propagator builder

        Returns:
            Propagator: propagator
        """
        mesg = "\033[91m  [WARN] Attitude Propagation still very buggy and unreliable" + \
               " Use at own risk!\033[0m"
        print mesg
        propagator = builderInstance.propagator
        setup = builderInstance.attSettings['settings']
        if builderInstance.earth is not None:
            earth = builderInstance.earth
        else:
            earth = _build_default_earth('Attitude Propagation')

        iT_dict = setup['inertiaTensor']
        int_dict = setup['integrator']
        discSettings = setup['Discretization']

        gravitySettings = setup['GravityGradient']
        solarSettings = setup['SolarPressure']
        dragSettings = setup['AeroDrag']
        magSettings = setup['MagneticTorque']
        innerCuboids = None
        surfaceMesh = None

        AttitudeFM = dict()

        # add Spacecraft State observer as force model to be able
        # to extract spacecraft state during integration
        StateOb = StateObserver(propagator.getInitialState())
        propagator.addForceModel(StateOb)
        AttitudeFM['StateObserver'] = StateOb

        # Intertia Tensor of Spacecraft:
        Ix = [float(x) for x in iT_dict['Ix'].split(" ")]
        Iy = [float(x) for x in iT_dict['Iy'].split(" ")]
        Iz = [float(x) for x in iT_dict['Iz'].split(" ")]
        inertiaT = np.array([Ix, Iy, Iz])

        # add Gravity Gradient Torque to Attitude Propagation:
        if gravitySettings['add']:
            innerCuboids = discretize_inner_body(discSettings)

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

            surfaceMesh = discretize_outer_surface(solarSettings, discSettings)
            sun = CelestialBodyFactory.getSun()

            if solarSettings['add']:
                AttitudeFM['Sun'] = sun
                # dummy spacecraft which enables access to methods
                #  needed to create SolarRadiationPressure objet
                dummy = IsotropicRadiationClassicalConvention(
                                                1.0,
                                                solarSettings['AbsorbCoeff'],
                                                solarSettings['ReflectCoeff'])

                # Force Model needed to get Lighting ratio:
                eqRad = earth.getEquatorialRadius()
                AttitudeFM['SolarModel'] = SolarRadiationPressure(sun,
                                                                  eqRad,
                                                                  dummy)

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
            gmLoader = GeoMagneticModelLoader()
            manager = DataProvidersManager.getInstance()
            manager.feed('WMM15.COF', gmLoader)

            # get item from Collection and transform model to year in sim.
            GM = gmLoader.getModels().iterator().next()
            GM = GM.transformModel(float(builderInstance.refDate
                                                        .getDate()
                                                        .toString()[:4]))

            AttitudeFM['MagneticModel'] = GM
            AttitudeFM['Earth'] = earth

        provider = AttitudePropagation(builderInstance.initialState.getAttitude(),
                                       builderInstance.refDate,
                                       inertiaT,
                                       builderInstance.tol,
                                       int_dict,
                                       innerCuboids,
                                       surfaceMesh,
                                       AttitudeFM)

        # for now assume constand dipole vector:
        x = [float(x) for x in magSettings['Dipole'].split(" ")]
        dipole = Vector3D(magSettings['Area'], Vector3D(x[0], x[1], x[2]))
        provider.setDipoleVector(dipole)

        propagator.setAttitudeProvider(provider)

        return propagator
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
        JArray('object'): Filled with facet list for orekit to create
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
            propagator: Propgator
            atmosphere: DTM2000 model of atmosphere
        """

        propagator = builderInstance.propagator
        if builderInstance.earth is not None:
            earth = builderInstance.earth
        else:
            earth = _build_default_earth('DragDTM2000')
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
            propagator: Propgator
            atmosphere: DTM2000 model of atmosphere
        """
        propagator = builderInstance.propagator
        if builderInstance.earth is not None:
            earth = builderInstance.earth
        else:
            earth = _build_default_earth('DragDTM2000')
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
        if builderInstance.earth is not None:
            earth = builderInstance.earth
        else:
            earth = _build_default_earth('SolarPressureBoxModel')

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
