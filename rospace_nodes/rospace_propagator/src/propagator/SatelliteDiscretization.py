# @copyright Copyright (c) 2018, Christian Lanegger (lanegger.christian@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import itertools
import numpy as np
import abc

import orekit

from org.hipparchus.geometry.euclidean.threed import Vector3D


class DiscretizationInterface(object):
    '''
    Base class for discretization of satellite.
    '''

    __metaclass__ = abc.ABCMeta

    @staticmethod
    @abc.abstractmethod
    def discretize_inner_body(sat_mass, discSettings):
        pass

    @staticmethod
    @abc.abstractmethod
    def discretize_outer_surface(solarSettings, discSettings):
        pass


class BoxWingModel(DiscretizationInterface):

    @staticmethod
    def discretize_inner_body(discSettings):
        """
        Discretized a shoebox-type satellite in cuboid of equal mass.

        Depends on defined number of cuboid in x,y and z
        direction and total size of satellite.

        Args:
            discSettings: dictionary with satellite_dim, and number of cuboid in
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
            inCub: dictionary with center of mass of each cuboid in satellite
                   frame and its corresponding mass
        """

        s_l_x = float(discSettings['satellite_dim']['l_x'])
        s_l_y = float(discSettings['satellite_dim']['l_y'])
        s_l_z = float(discSettings['satellite_dim']['l_z'])

        if s_l_x <= 0. or s_l_y <= 0. or s_l_z <= 0.:
            raise ValueError("Dimensions of satellite must be bigger than zero!")

        # separate cuboid into number of smaller cuboid and store
        # coordinates of center of mass in satellite frame
        numC_x = discSettings['inner_cuboids']['numCub_x']
        numC_y = discSettings['inner_cuboids']['numCub_y']
        numC_z = discSettings['inner_cuboids']['numCub_z']

        if numC_x <= 0 or numC_y <= 0 or numC_z <= 0:
            raise ValueError("Number of cuboid must be bigger than zero!")

        # dimension of inner cuboid:
        c_l_x = s_l_x / numC_x
        c_l_y = s_l_y / numC_y
        c_l_z = s_l_z / numC_z

        # total number of cuboid
        numC_tot = numC_x * numC_y * numC_z

        # populate satellite with cuboid:
        inCub = dict()
        CoM = []
        CoM_np = np.empty([0, 3])
        dm_Cub = np.empty([0, 1])
        massFrac = 1.0 / numC_tot

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
                    CoM_np = np.append(CoM_np, [[CoM_x, CoM_y, CoM_z]], axis=0)  # for numpy disturbance torque
                    dm_Cub = np.append(dm_Cub, [[massFrac]], axis=0)  # uniform distribution for satellite atm

        # add booms to list:
        if 'Booms' in discSettings:
            dm_boom = np.empty([0, 1])

            for boom in discSettings['Booms'].items():
                boom_dir = np.asarray([float(x) for x in boom[1]['dir'].split(" ")])
                norm_dir = np.linalg.norm(boom_dir)
                if norm_dir > 1.:
                    # normalize if necessary
                    boom_dir = boom_dir / norm_dir

                CoM_np = np.append(CoM_np, [float(boom[1]['length']) * boom_dir], axis=0)
                dm_boom = np.append(dm_boom, [[float(boom[1]['mass'])]], axis=0)

            inCub['dm_boom'] = dm_boom

        # inCub['CoM'] = CoM
        inCub['CoM_np'] = CoM_np
        inCub['dm'] = dm_Cub

        return inCub

    @staticmethod
    def discretize_outer_surface(solarSettings, discSettings):
        """
        Discretization of outer surface of a shoebox satellite into planes of equal area.

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

        if s_l_x <= 0. or s_l_y <= 0. or s_l_z <= 0.:
            raise ValueError("Dimensions of satellite must be bigger than zero!")

        sat_Ca = solarSettings['AbsorbCoeff']
        sat_Cs = solarSettings['ReflectCoeff']
        sol_Ca = solarSettings['SolarArray_AbsorbCoeff']
        sol_Cs = solarSettings['SolarArray_ReflectCoeff']

        numSR_x = int(discSettings['surface_rectangles']['numSR_x'])
        numSR_y = int(discSettings['surface_rectangles']['numSR_y'])
        numSR_z = int(discSettings['surface_rectangles']['numSR_z'])

        if numSR_x <= 0 or numSR_y <= 0 or numSR_z <= 0:
            raise ValueError("Number of cuboid must be bigger than zero!")

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
        CoM_np = np.empty([0, 3])
        Normal = []
        Normal_np = np.empty([0, 3])
        Area = []
        Coefs = []
        mesh_dA = dict()

        # front/back from left to right, bottom to top:
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

                CoM_np = np.append(CoM_np, [[front_CoM_x, CoM_y, CoM_z]], axis=0)
                CoM_np = np.append(CoM_np, [[back_CoM_x, CoM_y, CoM_z]], axis=0)

                Normal.append(front_Normal)
                Normal.append(back_Normal)

                Normal_np = np.append(Normal_np, [[1.0, 0.0, 0.0]], axis=0)
                Normal_np = np.append(Normal_np, [[-1.0, 0.0, 0.0]], axis=0)

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

                CoM_np = np.append(CoM_np, [[CoM_x, bottom_CoM_y, CoM_z]], axis=0)
                CoM_np = np.append(CoM_np, [[CoM_x, top_CoM_y, CoM_z]], axis=0)

                Normal.append(bottom_Normal)
                Normal.append(top_Normal)

                Normal_np = np.append(Normal_np, [[0.0, 1.0, 0.0]], axis=0)
                Normal_np = np.append(Normal_np, [[0.0, -1.0, 0.0]], axis=0)

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
                                    CoM_y,
                                    right_CoM_z))

                CoM_np = np.append(CoM_np, [[CoM_x, CoM_y, left_CoM_z]], axis=0)
                CoM_np = np.append(CoM_np, [[CoM_x, CoM_y, right_CoM_z]], axis=0)

                Normal.append(left_Normal)
                Normal.append(right_Normal)

                Normal_np = np.append(Normal_np, [[0.0, 0.0, 1.0]], axis=0)
                Normal_np = np.append(Normal_np, [[0.0, 0.0, -1.0]], axis=0)

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

            if sol_l_x <= 0. or sol_l_z <= 0.:
                raise ValueError("Dimensions of solar panels must be bigger than zero!")

            numSRSolar_x = solarSettings['numSRSolar_x']
            numSRSolar_z = solarSettings['numSRSolar_z']

            if numSRSolar_x <= 0 or numSRSolar_z <= 0:
                raise ValueError("Number of rectangles must be bigger than zero!")

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
                        CoM_np = np.append(CoM_np, [[dC[0] + CoM_x, dC[1], dC[2] + CoM_z]], axis=0)

                        Normal.append(Vector3D(float(normal[0]),
                                               float(normal[1]),
                                               float(normal[2])))
                        Normal_np = np.append(Normal_np, [[normal[0], normal[1], normal[2]]], axis=0)

                        Area.append(solArea)
                        Coefs.append(np.array([sol_Ca, sol_Cs]))

        # fill dictionary with lists:
        mesh_dA['CoM'] = CoM
        mesh_dA['CoM_np'] = CoM_np.astype(float)
        mesh_dA['Normal'] = Normal
        mesh_dA['Normal_np'] = Normal_np.astype(float)
        mesh_dA['Area'] = Area
        mesh_dA['Area_np'] = np.asarray(Area)
        mesh_dA['Coefs'] = Coefs

        # add to coefs diffusive reflection coefficient
        Coefs_np = np.asarray(Coefs)
        col = np.array([1 - np.sum(Coefs_np, axis=1)])
        Coefs_np = np.concatenate((Coefs_np, col.T), axis=1)
        try:
            assert(any(Coefs_np[:, 2] >= 0))
        except AssertionError:
            raise AssertionError(
                "Negative diffuse reflection coefficient not possible!")
        mesh_dA['Coefs_np'] = Coefs_np

        return mesh_dA
