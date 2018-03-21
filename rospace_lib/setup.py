#!/usr/bin/env python
#
# @copyright Copyright (c) 2017, Michael Pantic (michael.pantic@gmail.com)
#
# @license zlib license
#
# This file is licensed under the terms of the zlib license.
# See the LICENSE.md file in the root of this repository
# for complete details.

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    # #  don't do this unless you want a globally visible script
    packages=['rospace_lib'],
    package_dir={'': 'src'}
)

setup(**d)
