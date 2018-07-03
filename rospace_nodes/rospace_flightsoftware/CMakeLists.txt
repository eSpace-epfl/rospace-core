cmake_minimum_required(VERSION 2.8.3)
project(rospace_flightsoftware)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

cs_install()
if(CATKIN_ENABLE_TESTING)
	find_package(rostest REQUIRED)
	add_rostest(test/launch_FlightSoftwareNode_integration.test)
endif()

cs_export()
