# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import abc

from ThrustModel import ThrustModel

import orekit
from orekit import JArray_double, JArray
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants as Cst
from org.orekit.utils import IERSConventions as IERS
from org.orekit.time import TimeScalesFactory
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.bodies import CelestialBody
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngle
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.forces import BoxAndSolarArraySpacecraft
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.drag import DragForce
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.gravity import SolidTides, OceanTides, Relativity
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.drag.atmosphere import DTM2000
from org.orekit.forces.drag.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.attitudes import NadirPointing
from org.orekit.data import DataProvidersManager

from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D


def build_default_gravity_Field(methodName):
    """
    Build gravity field using Normalized Provider with degree and order of 5.

    Args:
        methodName: name of method calling this function (for printing warning)

    Returns:
        NormalizedSphericalHarmonicsProvider: provids norm. spherical harmonics

    """
    mesg = "\033[93m  [WARN] [Builder." + methodName\
           + "]: No Gravity Field defined. Creating default using"\
           + " NormalizedProvider of degree and order 5.\033[0m"

    print mesg

    return GravityFieldFactory.getNormalizedProvider(5, 5)


def build_default_earth(methodName):
    '''
    Build earth object in EME2000 Frame using OneAxisElliposoid as Earth model.

    Uses Constants based on WGS84 Standard from Orekit library.

    Args:
        methodName: name of method calling this function (for printing warning)

    Returns:
        OneAxisEllipsoid: non-rotating Earth Body

    '''
    mesg = "\033[93m  [WARN] [Builder." + methodName \
           + "]: No earth defined. Creating default Earth using" \
           + " OneAxisElliposoid in EME2000 Frame and constants based on"\
           + " the WGS84 Standard.\033[0m"
    print mesg

    return OneAxisEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                            Cst.WGS84_EARTH_FLATTENING,
                            FramesFactory.getEME2000())


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

    Here is an fictional example for the _build_state method:
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
            raise ValueError("No existing state class defined. \
                            State could not be build!")

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
            DormandPrince853Integrator(minStep, maxStep,
                                       JArray_double.cast_(self.tol[0]),
                                       # Double array of doubles
                                       # needs to be casted in Python
                                       JArray_double.cast_(self.tol[1]))

        self.integrator.setInitialStepSize(initStep)

    def _build_propagator(self):
        """
        Build of propagator object and add integrator and initial State to it.
        """

        self.propagator = NumericalPropagator(self.integrator)
        self.propagator.setOrbitType(self.orbitType)
        self.propagator.setInitialState(self.initialState)

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
                [prop, earth, gravField] = model.Setup(self.propagator,
                                                       GM['settings'])
                self.propagator = prop
                self.earth = earth
                self.gravField = gravField
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
                gravField = build_default_gravity_Field('_build_solid_tides')
            else:
                gravField = self.gravField

            if self.earth is None:
                earth = build_default_earth('_build_solid_tides')
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
                gravField = build_default_gravity_Field('_build_ocean_tides')
            else:
                gravField = self.gravField

            if self.earth is None:
                earth = build_default_earth('_build_ocean_tides')
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
                gravField = build_default_gravity_Field('_build_relativity')
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
        '''
        Call after build has been completed.
        Returns finished propagator object.
        '''

        # Print all force models which are being integrated
        print "[%s]: added Force models: \n%s"\
              % (self.SatType, str(self.propagator.getAllForceModels()))

        return self.propagator

    def get_earth(self):
        '''
        Returns Earth object which was created during gravity build.
        '''

        return self.earth

    def get_thrust_model(self):
        '''
        Returns thrust model if build and added to propagator.
        Otherwise returns None.
        '''

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


class Keplerian(StateFactory):

    @staticmethod
    def isApplicable(name):

        if name == "Keplerian":
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


class GravityPert(GravityFactory):

    @staticmethod
    def isApplicable(name):
        if name == "GravityPert":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, setup):
        """
        Add gravity pertubation using the HolmesFeatherstoneAttractionModel.

        As Earth Model a OneAxisEllipsoid is used specified in the same
        inertial Frame as the satellite state. The Earth is not rotating.

        Uses WGS84 norm for the equatorial radius and flattening.

        Args:
            propagator: propagator object
            setup: settings defined in dictionary and needed for build

        Returns:
            Propagator: propagator
            OneAxisEllipsoid: Earth body
            NormalizedSphericalHarmonicsProvider: gravity field
        """

        earth = OneAxisEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                                 Cst.WGS84_EARTH_FLATTENING,
                                 propagator.getFrame())

        degree = setup['degree']
        order = setup['order']
        gravField = GravityFieldFactory.getNormalizedProvider(degree, order)
        gravModel = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(),
                                                      gravField)

        propagator.addForceModel(gravModel)

        return [propagator, earth, gravField]


class GravityPertAndRefEllipsoid(GravityFactory):

    @staticmethod
    def isApplicable(name):
        if name == "GravityPertAndRefEllipsoid":
            return True
        else:
            return False

    @staticmethod
    def Setup(propagator, setup):
        """
        Add gravity pertubation using the HolmesFeatherstoneAttractionModel.

        As Earth model a ReferenceEllipsoid is used specified in the same
        inertial Frame as the satellite state.

        Uses WGS84 norm for the equatorial radius, flattening, standard
        grav. parameter and angular velocity. Earth is rotating at a constant
        rate.

        Returns:
            Propagator: propagator
            ReferenceEllipsoid: Earth body
            NormalizedSphericalHarmonicsProvider: gravity field
        """

        earth = ReferenceEllipsoid(Cst.WGS84_EARTH_EQUATORIAL_RADIUS,
                                   Cst.WGS84_EARTH_FLATTENING,
                                   propagator.getFrame(),
                                   Cst.WGS84_EARTH_MU,
                                   Cst.WGS84_EARTH_ANGULAR_VELOCITY)

        degree = setup['degree']
        order = setup['order']
        gravField = GravityFieldFactory.getNormalizedProvider(degree, order)
        gravModel = HolmesFeatherstoneAttractionModel(earth.getBodyFrame(),
                                                      gravField)

        propagator.addForceModel(gravModel)

        return [propagator, earth, gravField]
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
            earth = build_default_earth('AttNadir')

        attitude = NadirPointing(builderInstance.inertialFrame,
                                 earth)

        propagator.setAttitudeProvider(attitude)

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


class DragDTM2000(DragFactory):

    @staticmethod
    def isApplicable(name):
        if name == "DragDTM2000":
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
            earth = build_default_earth('DragDTM2000')
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
            earth = build_default_earth('SolarPressureBoxModel')

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
