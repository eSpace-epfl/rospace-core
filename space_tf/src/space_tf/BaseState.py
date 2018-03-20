# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

import Constants

class BaseState(object):
    """
    Base Class for all Coordinate elements.
    Contains coordinate frame and time.
    """

    def __init__(self):
        self.time = float(0)
        '''Timestamp. Format to be defined.'''

        self.frame = ""
        ''' Coordinate frame identifier, to be defined.'''

    def get_jd_time(self):
        """
        Function to calculate julian date (based on J2000 epoch) according to timestamp.
        See https://en.wikipedia.org/wiki/Julian_day for more information

        Returns:
            float: Julian date

        """
        delta = self.time - Constants.Constants.J2000_date
        return Constants.Constants.J2000_jd + delta.total_seconds() / (60.0 * 60 * 24)
