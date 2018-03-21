cmake_minimum_required(VERSION 2.8.3)
project(orekit_wrapper)

## Uses the precompiled version of orekit 9.0 downloaded from anaconda
## Works only on 64-bit linux
find_package(catkin REQUIRED)

## Needed for ExternalProject
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)


set(BUILD_PKG_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(BUILD_WRAPPER_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LIB_DIR https://espace-open.gitlab.io/orekit-wrapper/orekit-wrapper.zip)
set(DATA_ZIP_DIR https://espace-open.gitlab.io/orekit-wrapper/orekit-data.zip)

## Check if library already in loaded if not copy from build space or
## download from anaconda
if(EXISTS "${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit/_orekit.so")
	message(STATUS "Orekit library already loaded in devel space.")

elseif(EXISTS "${BUILD_WRAPPER_DIR}/_orekit.so" AND EXISTS "${BUILD_WRAPPER_DIR}/orekit-data.zip")
	message(STATUS "Orekit library already downloaded, but not in devel space. Library will be copied from build to devel directory.")

	execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${BUILD_WRAPPER_DIR}/src/orekit ${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit)
	execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${BUILD_WRAPPER_DIR}/orekit-data.zip ${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit)

else(EXISTS "${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit")
	message(STATUS "Cannot find Orekit library. Library will be downloaded and put in devel space")

	ExternalProject_Add(orekit
	PREFIX 				${BUILD_WRAPPER_DIR}
	URL 				${LIB_DIR}

	UPDATE_COMMAND 		""
	CONFIGURE_COMMAND 	""
	BUILD_COMMAND 		""

	INSTALL_COMMAND 	${CMAKE_COMMAND} -E copy_directory ${BUILD_WRAPPER_DIR}/src/orekit ${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit 
						COMMAND ${CMAKE_COMMAND} -E remove ${BUILD_WRAPPER_DIR}/orekit-wrapper.zip

	TEST_COMMAND 		wget -O ${BUILD_WRAPPER_DIR}/orekit-data.zip "${DATA_ZIP_DIR}" 
				 		COMMAND ${CMAKE_COMMAND} -E copy ${BUILD_WRAPPER_DIR}/orekit-data.zip ${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit

	LOG_DOWNLOAD 		1
	LOG_UPDATE 			0
	LOG_CONFIGURE 		0
	LOG_BUILD 			0
	LOG_TEST 			0
	LOG_INSTALL 		1
	)

	## The tarball file is being deleted and only the library kept in the
	## devel space
endif(EXISTS "${CATKIN_DEVEL_PREFIX}/lib/python2.7/dist-packages/orekit/_orekit.so")

