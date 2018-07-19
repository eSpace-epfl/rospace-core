cmake_minimum_required(VERSION 2.8.3)
project(rospace_imu)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
catkin_python_setup()
cs_install()
if(CATKIN_ENABLE_TESTING)
	find_package(rostest REQUIRED)
	add_rostest(test/launch_IMUnode.test)
	add_rostest(test/launch_IMUnode_integration.test)
endif()

cs_export()
