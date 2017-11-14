cmake_minimum_required(VERSION 2.8.3)
project(orekit_wrapper)

## Uses the precompiled version of orekit 9.0 downloaded from anaconda
## Works only on 64-bit linux
find_package(catkin REQUIRED)

## Needed for ExternalProject
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)


set(PKG_DIR ${CMAKE_SOURCE_DIR}/orbdyn-tools/orekit_wrapper)
set(WRAPPER_DIR ${CMAKE_BINARY_DIR}/orbdyn-tools/orekit_wrapper)
set(LIB_DIR https://anaconda.org/conda-forge/orekit/9.0/download/linux-64/orekit-9.0-py27_0.tar.bz2)

## Check if library already in loaded if not copy from build space or
## download from anaconda
if(EXISTS "${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit")
	message(STATUS "Orekit library already loaded in devel space.")

elseif(EXISTS "${WRAPPER_DIR}/src/Orekit_Library/lib/python2.7/site-packages/orekit")
	message(STATUS "Orekit library alredy downloaded, but not in devel space. Library will be copied from build to devel directory.")

	execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${WRAPPER_DIR}/src/Orekit_Library/lib/python2.7/site-packages/orekit ${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit)

else(EXISTS "${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit")
	message(STATUS "Cannot find Orekit library. Library will be downloaded and put in devel space")

	ExternalProject_Add(Orekit_Library
	PREFIX ${WRAPPER_DIR}
	URL ${LIB_DIR}
	UPDATE_COMMAND ""
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${WRAPPER_DIR}/src/Orekit_Library/lib/python2.7/site-packages/orekit ${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit COMMAND ${CMAKE_COMMAND} -E remove ${WRAPPER_DIR}/src/orekit-9.0-py27_0.tar.bz2
	TEST_COMMAND ""
	LOG_DOWNLOAD 1
	LOG_UPDATE 0
	LOG_CONFIGURE 0
	LOG_BUILD 0
	LOG_TEST 0
	LOG_INSTALL 1)

	## The tarball file is being deleted and only the library kept in the
	## devel space
endif(EXISTS "${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit")
