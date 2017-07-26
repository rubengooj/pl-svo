################################################################################
# user build settings
SET(TRACE TRUE)
SET(HAVE_G2O FALSE)
SET(HAVE_MRPT TRUE)

################################################################################

SET(DEFAULT_HAS_MRPT ON)
SET(HAS_MRPT ${DEFAULT_HAS_MRPT} CACHE BOOL "Build the representation api which uses the MRPT library")

SET(PROJECT_NAME plsvo)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo -> Set in cmake-gui option
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Set definitions
IF(HAVE_MRPT)
  ADD_DEFINITIONS(-DHAS_MRPT)
ENDIF()

# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas -Wno-reorder")
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(OpenCV 3 REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(fast REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system filesystem)

IF(HAVE_MRPT)
    SET(MRPT_DONT_USE_DBG_LIBS 1) #use release libraries for linking even if "Debug" CMake build
    FIND_PACKAGE(MRPT REQUIRED base opengl gui)
ENDIF()

FIND_LIBRARY(YAML_CPP_LIBRARIES yaml-cpp)
if(NOT YAML_CPP_LIBRARIES)
  # If yaml-cpp not found in the system, try finding it as a user CMake-generated project
  FIND_PACKAGE(yaml-cpp REQUIRED)
  INCLUDE_DIRECTORIES(${YAML_CPP_INCLUDE_DIRS})
endif(NOT YAML_CPP_LIBRARIES)

FIND_PACKAGE(vikit_common REQUIRED)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Include dirs
INCLUDE_DIRECTORIES(
  include
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${fast_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/3rdparty/line_descriptor/include/
)

# Set link libraries
LIST(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  #${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${fast_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ~/libs/Sophus/build/libSophus.so
  ${PROJECT_SOURCE_DIR}/3rdparty/line_descriptor/lib/liblinedesc.so
)

INCLUDE_DIRECTORIES(${vikit_common_INCLUDE_DIRS})
LIST(APPEND LINK_LIBS ${vikit_common_LIBRARIES})

# Set sourcefiles
LIST(APPEND SOURCEFILES
  src/frame_handler_mono.cpp
  src/frame_handler_base.cpp
  src/frame.cpp
  src/feature.cpp
  src/feature3D.cpp
  src/feature3D_impl.cpp
  src/map.cpp
  src/pose_optimizer.cpp
  src/initialization.cpp
  src/matcher.cpp
  src/reprojector.cpp
  src/feature_alignment.cpp
  src/feature_detection.cpp
  src/depth_filter.cpp
  src/config.cpp
  src/sparse_img_align.cpp)

IF(HAVE_MRPT)
    LIST(APPEND SOURCEFILES src/sceneRepresentation.cpp)
ENDIF()

# Code below is just for accessibility issues in IDE editor
# List all files (headers) contained by svo library
FILE(GLOB_RECURSE all_include_files RELATIVE "${CMAKE_SOURCE_DIR}" *.h *.hpp)

# Visualize the files of this directory in IDE creating an custom empty target
add_custom_target( plsvo_includes DEPENDS ${all_include_files} SOURCES ${all_include_files} )

# Create svo library
ADD_LIBRARY(plsvo SHARED ${SOURCEFILES})
TARGET_LINK_LIBRARIES(plsvo ${LINK_LIBS} ${MRPT_LIBS})

################################################################################

IF(HAVE_MRPT)
    ADD_EXECUTABLE( run_pipeline  	    app/run_pipeline.cpp )
    TARGET_LINK_LIBRARIES( run_pipeline	        plsvo)
ENDIF()


