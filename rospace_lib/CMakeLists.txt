cmake_minimum_required(VERSION 2.8.3)
project(space_tf)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_python_setup()
cs_install()

cs_export()
