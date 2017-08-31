# Note: these tests are quite preliminary....

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../src/") #hack...
import space_tf

class ConversionTest(unittest.TestCase):

    def test_OrbitalElementsToCartesian(self):
        # set up data
        i_tle = np.deg2rad(98.5061)
        omega_tle = np.deg2rad(21.2475)
        e_tle = 0.0008182
        w_tle = np.deg2rad(34.7001)
        m_tle = np.deg2rad(325.4728)
        motion_tle = 14.56040586

        # convert OE => TEME cartesian
        orb = space_tf.OrbitalElements()
        orb.fromTLE(i_tle, omega_tle, e_tle, m_tle, w_tle, motion_tle)

        cartTEME = space_tf.CartesianTEME()

        space_tf.Converter.convert(orb, cartTEME)

        # convert back
        orb2 = space_tf.OrbitalElements()
        space_tf.Converter.convert(cartTEME, orb2)

        # compare
        assert(np.linalg.norm(orb.asArray()-orb2.asArray())<1e-10)


    def test_TEMEtoITRF(self):
        # set up data
        cartTEME = space_tf.CartesianTEME()
        cartTEME.R = np.array([7000,0,0]) # x = celestial equinox

        cartITRF = space_tf.CartesianITRF()
        cartITRF.epochJD = 2449069 #equinox of 1993 -seems to be the proper direction?!
        space_tf.Converter.convert(cartTEME, cartITRF)

        #TODO: This test is weird (see also comment in Converter, TEME not completely clear)
        assert(np.linalg.norm(cartITRF.R -  cartTEME.R.reshape(3,1))<1.0)



if __name__ == '__main__':
    unittest.main()
